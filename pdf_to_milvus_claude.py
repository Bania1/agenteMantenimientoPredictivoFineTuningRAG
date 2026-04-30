"""
pdf_to_milvus.py
================
Extrae averías del manual Bosch Microondas Serie 6, las enriquece con la
API de Claude (análisis semántico), genera embeddings locales con TF-IDF + LSA
y las almacena en una base de datos vectorial Milvus Lite (fichero local .db).

┌─────────────────────────────────────────────────────────────────────┐
│  ARQUITECTURA DEL PIPELINE                                          │
│                                                                     │
│  PDF ──fitz──► bloques de texto                                     │
│       ──regex──► fragmentos por avería                              │
│       ──Claude API──► JSON estructurado (+ fallback heurístico)     │
│       ──TF-IDF + LSA──► vectores 384d                               │
│       ──Milvus Lite──► colección "averias" con índice AUTOINDEX     │
└─────────────────────────────────────────────────────────────────────┘

Dependencias:
    pip install pymupdf "pymilvus[milvus_lite]" scikit-learn

Uso:
    # Procesar el PDF e indexar en Milvus
    python pdf_to_milvus.py

    # PDF personalizado
    python pdf_to_milvus.py --pdf ruta/manual.pdf

    # Solo busqueda semantica (sin reprocesar el PDF)
    python pdf_to_milvus.py --query "el plato giratorio hace ruido" --solo-buscar

    # Busqueda con filtro de peligrosidad maxima
    python pdf_to_milvus.py --query "no arranca" --peligro-max 3

Salidas:
    averias_extraidas.json   -> JSON con todos los registros extraidos
    averias_microondas.db    -> Base de datos vectorial Milvus Lite
"""

import argparse
import json
import re
import time
import uuid
import warnings
from pathlib import Path

import fitz                                                      # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from pymilvus import MilvusClient, DataType

warnings.filterwarnings("ignore")

# ==================================================================
# CONFIGURACION GLOBAL
# ==================================================================

PDF_PATH        = "/home/alvaro/CATEDRA/microondasboschserie6.pdf"
MILVUS_DB       = "averias_microondas.db"
COLLECTION_NAME = "averias"
OUTPUT_JSON     = "averias_extraidas.json"
EMBED_DIM       = 384

# Paginas del manual en espanol que contienen averias (indice base-0)
# Paginas 15-17 del PDF = indices 14, 15, 16
FAULT_PAGES = list(range(14, 17))

# Patron de inicio de averia extraido de la tabla del manual
FAULT_START_PATTERN = re.compile(
    r"(?="
    r"El aparato no funciona\.|"
    r"Los alimentos tardan|"
    r"El plato giratorio|"
    r"El funcionamiento del microon|"
    r"El aparato no est[aá] en funciona|"
    r"En el display aparecen tres ce|"
    r"En el display aparece\b|"
    r"Aparece el mensaje"
    r")"
)

# ==================================================================
# 1 - EXTRACCION DE TEXTO CON FITZ (PyMuPDF)
# ==================================================================

def extract_fault_blocks(pdf_path: str) -> list:
    """
    Lee las paginas de averias del PDF usando PyMuPDF y devuelve
    una lista de bloques de texto con su numero de pagina.
    """
    doc = fitz.open(pdf_path)
    blocks = []
    for page_num in FAULT_PAGES:
        page = doc[page_num]
        for b in page.get_text("blocks"):
            text = b[4].strip()
            if len(text) > 20:
                blocks.append({"page": page_num + 1, "text": text.replace("\n", " ")})
    doc.close()
    return blocks


def segment_faults(blocks: list) -> list:
    """
    Agrupa los bloques de texto en entradas de averia individuales.
    Cada averia se corresponde con una fila de la tabla del manual.
    """
    full_text = "\n".join(b["text"] for b in blocks)

    start = full_text.find("Averias de funcionamiento")
    # El manual puede tener el acento
    if start == -1:
        start = full_text.find("Aver\u00edas de funcionamiento")
    end = full_text.find("Eliminaci\u00f3n")
    if start == -1:
        # Fallback: usar todo el texto
        start = 0

    section = full_text[start : end if end > start else None]
    parts   = FAULT_START_PATTERN.split(section)
    faults  = [p.strip() for p in parts if len(p.strip()) > 40]
    return faults


# ==================================================================
# 2 - ANALISIS SEMANTICO CON CLAUDE API
# ==================================================================

def analyze_fault_with_claude(fault_text: str) -> dict:
    """
    Envia el fragmento de averia a la API de Claude para extraer el JSON
    estructurado. Si la llamada falla (sin clave API, sin red, etc.)
    se usa automaticamente el extractor heuristico local.
    """
    import urllib.request

    prompt = (
        "Eres un tecnico especialista en electrodomesticos. "
        "Analiza el siguiente fragmento del manual de un microondas"
        "y extrae la informacion de la averia en formato JSON estructurado.\n\n"
        f"TEXTO DEL MANUAL:\n{fault_text}\n\n"
        "Devuelve UNICAMENTE un objeto JSON valido (sin markdown ni backticks) "
        "con exactamente estas claves:\n"
        '  "aparato"           : nombre/tipo del electrodomestico (string)\n'
        '  "sintoma"           : problema fisico o fallo reportado (string)\n'
        '  "codigo_error"      : codigo alfanumerico si existe, si no "No especificado" (string)\n'
        '  "causa_probable"    : motivo tecnico que causa el fallo (string)\n'
        '  "pasos_reparacion"  : instrucciones secuenciales para solucionarlo (string)\n'
        '  "porcentaje_certeza": confianza 0-100 sobre precision y completitud (integer)\n'
        '  "grado_peligrosidad": 1=mas peligroso ... 5=menos peligroso (integer)\n\n'
        "Responde SOLO con el JSON."
    )

    payload = json.dumps({
        "model"     : "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "messages"  : [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data    = payload,
        headers = {
            "Content-Type"     : "application/json",
            "anthropic-version": "2023-06-01",
        },
        method = "POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw = "".join(
            blk["text"] for blk in data.get("content", []) if blk.get("type") == "text"
        )
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        return json.loads(raw)

    except Exception as exc:
        print(f"    [!] Claude API no disponible ({type(exc).__name__}). Usando extractor heuristico.")
        return _heuristic_parse(fault_text)


def _heuristic_parse(text: str) -> dict:
    """
    Extractor de respaldo cuando la API no esta disponible.
    Analiza el texto usando reglas basadas en el formato del manual.
    """
    lines = [ln.strip() for ln in text.split(".") if len(ln.strip()) > 5]

    sintoma        = lines[0] if lines else text[:100]
    causa_probable = lines[1] if len(lines) > 1 else "No determinada"
    pasos          = ". ".join(lines[2:]) if len(lines) > 2 else "Consultar servicio tecnico."

    # Detectar codigo de error visible en pantalla
    codigo = "No especificado"
    if re.search(r"display|indicador|panel", text, re.IGNORECASE):
        m = re.search(r"\b([A-Z]\d{1,2}|\d{3})\b", text)
        if m:
            codigo = m.group(1)

    # Peligrosidad heuristica basada en palabras clave
    peligro = 5
    if re.search(r"descarga el[eé]ctrica|electrocuci[oó]n|alta tensi[oó]n", text, re.IGNORECASE):
        peligro = 1
    elif re.search(r"incendio|fuego|arder|llama|explotar", text, re.IGNORECASE):
        peligro = 2
    elif re.search(r"quemadura|vapor caliente|quemar", text, re.IGNORECASE):
        peligro = 3
    elif re.search(r"da[nñ]o|aver[ií]a|reparaci[oó]n", text, re.IGNORECASE):
        peligro = 4

    return {
        "aparato"           : "Microondas",
        "sintoma"           : sintoma,
        "codigo_error"      : codigo,
        "causa_probable"    : causa_probable,
        "pasos_reparacion"  : pasos,
        "porcentaje_certeza": 70,
        "grado_peligrosidad": peligro,
    }


# ==================================================================
# 3 - EMBEDDINGS LOCALES (TF-IDF + LSA, sin dependencias de red)
# ==================================================================

class LocalEmbedder:
    """
    Genera embeddings de 384 dimensiones sin conexion a internet usando
    TF-IDF (bigramas) + SVD (Latent Semantic Analysis).

    Si sentence-transformers esta disponible y hay acceso a HuggingFace,
    puedes sustituir esta clase por SentenceTransformer("all-MiniLM-L6-v2")
    para embeddings semanticos de mayor calidad.
    """

    def __init__(self, dim: int = 384):
        self.dim     = dim
        self.tfidf   = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.svd     = None
        self._fitted = False

    def fit(self, texts: list):
        X = self.tfidf.fit_transform(texts)
        n = min(self.dim - 1, X.shape[1] - 1)
        self.svd     = TruncatedSVD(n_components=n, random_state=42)
        self._fitted = True
        self.svd.fit(X)

    def encode(self, texts: list) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Llama a fit() antes de encode().")
        X    = self.tfidf.transform(texts)
        vecs = self.svd.transform(X)
        # Relleno hasta dim si hace falta
        if vecs.shape[1] < self.dim:
            pad  = np.zeros((vecs.shape[0], self.dim - vecs.shape[1]))
            vecs = np.hstack([vecs, pad])
        # Normalizacion L2
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-9)


# ==================================================================
# 4 - MILVUS: COLECCION E INSERCION
# ==================================================================

def setup_milvus(db_path: str, force_recreate: bool = False) -> MilvusClient:
    """Crea o abre la coleccion Milvus Lite con el schema de averias."""
    client = MilvusClient(db_path)

    if client.has_collection(COLLECTION_NAME):
        if force_recreate:
            client.drop_collection(COLLECTION_NAME)
            print(f"  [+] Coleccion '{COLLECTION_NAME}' eliminada (force_recreate=True).")
        else:
            print(f"  [ok] Coleccion '{COLLECTION_NAME}' ya existe, reutilizando.")
            return client

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    # Campos del esquema principal
    schema.add_field("id",                 DataType.VARCHAR,      max_length=64,   is_primary=True)
    schema.add_field("aparato",            DataType.VARCHAR,      max_length=256)
    schema.add_field("sintoma",            DataType.VARCHAR,      max_length=1024)
    schema.add_field("codigo_error",       DataType.VARCHAR,      max_length=64)
    schema.add_field("causa_probable",     DataType.VARCHAR,      max_length=2048)
    schema.add_field("pasos_reparacion",   DataType.VARCHAR,      max_length=4096)
    schema.add_field("porcentaje_certeza", DataType.INT64)
    schema.add_field("grado_peligrosidad", DataType.INT64)
    # Metadatos adicionales para filtrado rapido
    schema.add_field("fuente_pdf",         DataType.VARCHAR,      max_length=256)
    schema.add_field("pagina",             DataType.INT64)
    schema.add_field("fecha",              DataType.VARCHAR,      max_length=10)  # YYYY-MM-DD
    schema.add_field("hora",               DataType.VARCHAR,      max_length=8)   # HH:MM:SS
    # Vector semantico
    schema.add_field("embedding",          DataType.FLOAT_VECTOR, dim=EMBED_DIM)

    # AUTOINDEX: unico tipo soportado por milvus_lite local.
    # En un Milvus remoto/cluster puedes usar HNSW para mayor rendimiento.
    idx = client.prepare_index_params()
    idx.add_index("embedding", "AUTOINDEX", "COSINE")

    client.create_collection(COLLECTION_NAME, schema=schema, index_params=idx)
    print(f"  [ok] Coleccion '{COLLECTION_NAME}' creada con indice AUTOINDEX + similitud COSINE.")
    return client


def insert_records(
    client    : MilvusClient,
    records   : list,
    embeddings: list,
    pdf_path  : str = "",
) -> int:
    """Inserta los registros junto con sus embeddings y metadatos."""
    rows = []
    for rec, emb in zip(records, embeddings):
        row = {
            "id"                : str(uuid.uuid4()),
            "aparato"           : str(rec.get("aparato", ""))[:256],
            "sintoma"           : str(rec.get("sintoma", ""))[:1024],
            "codigo_error"      : str(rec.get("codigo_error", "No especificado"))[:64],
            "causa_probable"    : str(rec.get("causa_probable", ""))[:2048],
            "pasos_reparacion"  : str(rec.get("pasos_reparacion", ""))[:4096],
            "porcentaje_certeza": int(rec.get("porcentaje_certeza", 0)),
            "grado_peligrosidad": int(rec.get("grado_peligrosidad", 5)),
            "fuente_pdf"        : Path(pdf_path).name[:256],
            "pagina"            : int(rec.get("_pagina", 0)),
            "embedding"         : emb,
            "fecha"             : str(rec.get("fecha", ""))[:10],
            "hora"              : str(rec.get("hora", ""))[:8],
        }
        rows.append(row)

    result = client.insert(collection_name=COLLECTION_NAME, data=rows)
    return result.get("insert_count", len(rows))


# ==================================================================
# 5 - BUSQUEDA SEMANTICA
# ==================================================================

def search_similar(
    client     : MilvusClient,
    embedder   : LocalEmbedder,
    query      : str,
    top_k      : int = 3,
    peligro_max: int = 5,
) -> list:
    """
    Busca las averias mas similares a la consulta.
    Soporta filtrado por grado_peligrosidad para mostrar solo las
    averias con nivel de peligro menor o igual al umbral indicado.
    """
    qvec = embedder.encode([query])[0].tolist()

    filter_expr = f"grado_peligrosidad <= {peligro_max}" if peligro_max < 5 else ""

    results = client.search(
        collection_name = COLLECTION_NAME,
        data            = [qvec],
        limit           = top_k,
        filter          = filter_expr,
        output_fields   = [
            "aparato", "sintoma", "codigo_error",
            "causa_probable", "pasos_reparacion",
            "porcentaje_certeza", "grado_peligrosidad",
            "fuente_pdf", "pagina", "fecha", "hora"
        ],
    )

    hits = []
    for hit in results[0]:
        e = hit["entity"]
        hits.append({
            "similitud_coseno"  : round(hit["distance"], 4),
            "aparato"           : e["aparato"],
            "sintoma"           : e["sintoma"],
            "codigo_error"      : e["codigo_error"],
            "causa_probable"    : e["causa_probable"],
            "pasos_reparacion"  : e["pasos_reparacion"],
            "porcentaje_certeza": e["porcentaje_certeza"],
            "grado_peligrosidad": e["grado_peligrosidad"],
            "fuente_pdf"        : e["fuente_pdf"],
            "pagina"            : e["pagina"],
            "fecha"             : e["fecha"],
            "hora"              : e["hora"],
        })
    return hits


# ==================================================================
# 6 - PIPELINE PRINCIPAL
# ==================================================================

def run_pipeline(pdf_path: str):
    print("\n==========================================================")
    print("  PIPELINE: PDF -> Claude -> Embeddings -> Milvus Lite")
    print("==========================================================\n")

    # PASO 1: Extraccion de texto
    print("[1/5] Extrayendo bloques de texto del PDF con PyMuPDF...")
    blocks = extract_fault_blocks(pdf_path)
    faults = segment_faults(blocks)
    print(f"  -> {len(faults)} fragmentos de averia detectados.\n")

    # PASO 2: Analisis con Claude API
    print("[2/5] Analizando cada averia con Claude API...")
    records = []
    for i, fault_text in enumerate(faults, 1):
        print(f"  [{i:02d}/{len(faults):02d}] {fault_text[:55]}...")
        rec = analyze_fault_with_claude(fault_text)
        rec.setdefault("aparato",           "Microondas Bosch Serie 6")
        rec.setdefault("sintoma",           fault_text[:100])
        rec.setdefault("codigo_error",      "No especificado")
        rec.setdefault("causa_probable",    "No determinada")
        rec.setdefault("pasos_reparacion",  "Consultar servicio tecnico.")
        rec.setdefault("porcentaje_certeza", 70)
        rec.setdefault("grado_peligrosidad", 5)
        rec["_pagina"] = FAULT_PAGES[0] + 1
        records.append(rec)
        time.sleep(0.25)

    # Guardar JSON limpio (sin campo privado _pagina)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    print(f"\n  -> JSON guardado en '{OUTPUT_JSON}'.\n")

    # PASO 3: Embeddings locales
    print(f"[3/5] Generando embeddings locales (TF-IDF + LSA, dim={EMBED_DIM})...")
    embed_texts = [
        f"{r['sintoma']} {r['causa_probable']} {r['pasos_reparacion']}"
        for r in records
    ]
    embedder = LocalEmbedder(dim=EMBED_DIM)
    embedder.fit(embed_texts)
    embeddings = embedder.encode(embed_texts).tolist()
    print(f"  -> {len(embeddings)} vectores generados.\n")

    # PASO 4: Configurar Milvus
    print(f"[4/5] Configurando Milvus Lite ('{MILVUS_DB}')...")
    client = setup_milvus(MILVUS_DB, force_recreate=True)

    # PASO 5: Insertar
    print(f"[5/5] Insertando registros en la coleccion '{COLLECTION_NAME}'...")
    n_inserted = insert_records(client, records, embeddings, pdf_path)
    print(f"  -> {n_inserted} registros insertados.\n")

    print("==========================================================")
    print("  Pipeline completado con exito.")
    print(f"  JSON   : {OUTPUT_JSON}")
    print(f"  Milvus : {MILVUS_DB}")
    print("==========================================================\n")

    return records, client, embedder


# ==================================================================
# 7 - ENTRY POINT
# ==================================================================

def print_records_summary(records: list):
    print("\n=== AVERIAS EXTRAIDAS ===\n")
    for i, r in enumerate(records, 1):
        danger_bar = "#" * (6 - r["grado_peligrosidad"]) + "-" * (r["grado_peligrosidad"] - 1)
        print(f"  [{i:02d}] {r['sintoma'][:70]}")
        print(f"       Codigo     : {r['codigo_error']}")
        print(f"       Peligro    : [{danger_bar}] {r['grado_peligrosidad']}/5  (1=max, 5=min)")
        print(f"       Certeza    : {r['porcentaje_certeza']}%")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrae averias de un manual PDF y las indexa en Milvus Lite."
    )
    parser.add_argument("--pdf",         default=PDF_PATH,
                        help="Ruta al PDF del manual")
    parser.add_argument("--query",       default=None,
                        help="Consulta de busqueda semantica")
    parser.add_argument("--topk",        type=int, default=3,
                        help="Numero de resultados (default: 3)")
    parser.add_argument("--peligro-max", type=int, default=5, dest="peligro_max",
                        help="Filtrar: solo averias con grado_peligrosidad <= N (1-5, default: 5)")
    parser.add_argument("--solo-buscar", action="store_true",
                        help="Solo ejecutar busqueda, sin reprocesar el PDF")
    args = parser.parse_args()

    if args.solo_buscar and args.query:
        # Modo solo busqueda: reconstruir el embedder desde el JSON guardado
        print(f"\n[BUSQUEDA] Cargando datos desde '{OUTPUT_JSON}'...")
        if not Path(OUTPUT_JSON).exists():
            raise FileNotFoundError(
                f"No se encontro '{OUTPUT_JSON}'. Ejecuta primero sin --solo-buscar."
            )
        with open(OUTPUT_JSON) as f:
            records = json.load(f)

        embed_texts = [
            f"{r['sintoma']} {r['causa_probable']} {r['pasos_reparacion']}"
            for r in records
        ]
        embedder = LocalEmbedder(dim=EMBED_DIM)
        embedder.fit(embed_texts)
        client = MilvusClient(MILVUS_DB)
    else:
        records, client, embedder = run_pipeline(args.pdf)
        print_records_summary(records)

    # Busqueda semantica (demo automatica o consulta explicita)
    query = args.query or "el aparato no arranca ni da senal"
    label = "CONSULTA" if args.query else "DEMO DE BUSQUEDA"

    print(f"\n[{label}]: \"{query}\"")
    if args.peligro_max < 5:
        print(f"  Filtro activo: grado_peligrosidad <= {args.peligro_max}\n")

    hits = search_similar(client, embedder, query,
                          top_k=args.topk, peligro_max=args.peligro_max)

    if not hits:
        print("  Sin resultados con los filtros aplicados.")
    else:
        for i, h in enumerate(hits, 1):
            payload = {k: v for k, v in h.items() if k != "similitud_coseno"}
            print(f"\n  --- Resultado #{i}  (similitud coseno: {h['similitud_coseno']}) ---")
            print(json.dumps(payload, ensure_ascii=False, indent=4))
