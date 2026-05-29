#!/usr/bin/env python3
"""
Backend Flask con RAG integrado sobre Milvus.

Flujo de cada petición /api/chat:
  1. Recibe el mensaje del usuario
  2. Busca en Milvus los fragmentos más relevantes del manual (RAG)
  3. Inyecta ese contexto en el prompt del extractor (qwen-fusionado)
  4. El extractor devuelve el JSON estructurado
  5. El modelo de respuesta (llama3.2:1b) genera la respuesta final

Dependencias:
    pip install flask requests pymilvus sentence-transformers

Variables de entorno (o modifica las constantes de abajo):
    MILVUS_DB            ruta al fichero .db  (default: ./milvus.db)
    COLLECTION_NAME      nombre de la colección (default: reparaciones)
    OLLAMA_GENERATE_URL  URL de Ollama (default: http://127.0.0.1:11434/api/generate)
    EXTRACTOR_MODEL      modelo extractor (default: qwen-fusionado)
    RESPONSE_MODEL       modelo de respuesta (default: llama3.2:1b)
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, jsonify, request

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_GENERATE_URL = os.getenv("OLLAMA_GENERATE_URL", "http://127.0.0.1:11434/api/generate")
EXTRACTOR_MODEL     = os.getenv("EXTRACTOR_MODEL", "qwen-fusionado")
RESPONSE_MODEL      = os.getenv("RESPONSE_MODEL", "llama3.2:1b")
REQUEST_TIMEOUT     = int(os.getenv("REQUEST_TIMEOUT", "600"))
DEFAULT_MODE        = os.getenv("DEFAULT_MODE", "assistant")
DOMAIN_CATALOG_PATH = Path(__file__).with_name("domain_catalog.json")

# ── Milvus ───────────────────────────────────────────────────────────────────
MILVUS_DB       = os.getenv("MILVUS_DB", "./milvus.db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "reparaciones")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K           = int(os.getenv("TOP_K", "4"))   # fragmentos a recuperar por consulta

# ─────────────────────────────────────────────────────────────────────────────
# MILVUS — cliente y embedding (singletons, se cargan una vez al arrancar)
# ─────────────────────────────────────────────────────────────────────────────

_milvus_client   = None
_embedding_fn    = None
_milvus_ok       = False   # flag: False si Milvus no está disponible


def _init_milvus() -> None:
    """Intenta inicializar Milvus y el modelo de embeddings.
    Si falla (fichero no existe, librería no instalada…) deja _milvus_ok=False
    y el sistema sigue funcionando SIN RAG."""
    global _milvus_client, _embedding_fn, _milvus_ok
    try:
        from pymilvus import MilvusClient                                     # type: ignore
        from pymilvus.model.dense import SentenceTransformerEmbeddingFunction # type: ignore

        if not Path(MILVUS_DB).exists():
            print(f"[WARN] Milvus DB no encontrada en '{MILVUS_DB}'. RAG desactivado.")
            return

        print(f"[INFO] Cargando modelo de embeddings: {EMBED_MODEL} ...")
        _embedding_fn  = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL, device="cpu")
        _milvus_client = MilvusClient(MILVUS_DB)
        _milvus_ok     = True
        print(f"[OK]  Milvus listo. Colección: {COLLECTION_NAME}")
    except Exception as exc:
        print(f"[WARN] No se pudo inicializar Milvus: {exc}. RAG desactivado.")


def buscar_en_milvus(query: str, subject: str = "") -> List[Dict[str, Any]]:
    """Busca los TOP_K fragmentos más relevantes en Milvus para la query dada.
    Devuelve lista vacía si Milvus no está disponible."""
    if not _milvus_ok or _embedding_fn is None or _milvus_client is None:
        return []
    try:
        # encode_queries para preguntas (≠ encode_documents que se usa al indexar)
        vector = _embedding_fn.encode_queries([query])[0].tolist()
        filtro = f"subject == '{subject}'" if subject else ""
        resultados = _milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[vector],
            filter=filtro,
            limit=TOP_K,
            output_fields=["text", "source", "page", "subject"],
        )
        hits = []
        for hit in resultados[0]:
            entity = hit.get("entity", hit)
            hits.append({
                "score":   round(float(hit.get("distance", 0.0)), 4),
                "text":    entity.get("text", ""),
                "source":  entity.get("source", ""),
                "page":    entity.get("page", 0),
                "subject": entity.get("subject", ""),
            })
        return hits
    except Exception as exc:
        print(f"[WARN] Error buscando en Milvus: {exc}")
        return []


def construir_contexto_rag(hits: List[Dict[str, Any]]) -> str:
    """Convierte los fragmentos de Milvus en texto de contexto para el prompt."""
    if not hits:
        return ""
    bloques = []
    for i, h in enumerate(hits, 1):
        bloques.append(
            f"[Manual — fragmento {i}, pág. {h['page']}, relevancia {h['score']}]\n"
            f"{h['text'].strip()}"
        )
    return "\n\n".join(bloques)


# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP Y LÓGICA ORIGINAL (sin cambios salvo donde se indica con ← RAG)
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

REQUIRED_KEYS = {
    "aparato", "sintoma", "codigo_error", "causa_probable",
    "pasos_reparacion", "porcentaje_certeza", "grado_peligrosidad",
}

# ← RAG: el SYSTEM_PROMPT ahora incluye una sección {contexto_manual} opcional
SYSTEM_PROMPT_BASE = """[GOAL]
Eres un motor de extracción de datos de alta precisión especializado en soporte técnico de electrodomésticos. Tu tarea es analizar descripciones de fallos o manuales y extraer la información técnica clave con objetividad.

[OUTPUT FORMAT]
Responde ÚNICAMENTE con un objeto JSON válido y plano que contenga exactamente estas 7 claves:
"aparato", "sintoma", "codigo_error", "causa_probable", "pasos_reparacion", "porcentaje_certeza", "grado_peligrosidad".

[ESCALA DE PELIGROSIDAD]
Usa esta escala para el campo "grado_peligrosidad":
- 1: Peligro Crítico (Riesgo de Incendio o Arqueo Eléctrico) — chispas, olor a quemado, sobrecalentamiento extremo.
- 2: Peligro Alto (Sistemas de Potencia e Inverter) — fallos de magnetrón, inverter, sobrecalentamiento intermitente.
- 3: Peligro Medio (Componentes Internos de Tensión) — fusibles, diodos, sensores de temperatura/humedad.
- 4: Peligro Bajo (Interface y Mecánica de Puerta) — panel de control, mecanismos de cierre, resistencias de grill.
- 5: Peligro Mínimo (Configuración y Accesorios) — bloqueos de seguridad, accesorios externos, errores de software."""

BRAND_KEYWORDS = [
    "bosch", "balay", "siemens", "lg", "samsung", "whirlpool",
    "beko", "teka", "indesit", "electrolux", "aeg", "zanussi",
    "haier", "hisense", "miele",
]


def load_domain_catalog() -> Dict[str, Any]:
    with DOMAIN_CATALOG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


DOMAIN_CATALOG  = load_domain_catalog()
DEVICE_CATALOG  = DOMAIN_CATALOG.get("devices", {})


def detect_output_language(user_message: str) -> str:
    lowered = user_message.lower()
    english_markers = [
        " the ", " freezer", " fridge", " refrigerator", " washing machine",
        " dishwasher", " oven", " dryer", " microwave",
        " not ", " doesn't ", " does not ",
    ]
    if any(marker in f" {lowered} " for marker in english_markers):
        return "English"
    return "Spanish"


def get_domain_keywords() -> List[str]:
    keywords: List[str] = []
    for device_data in DEVICE_CATALOG.values():
        keywords.extend(device_data.get("aliases", []))
    return keywords


DOMAIN_KEYWORDS = get_domain_keywords()


def is_in_domain(user_message: str) -> bool:
    lowered = user_message.lower()
    return any(keyword in lowered for keyword in DOMAIN_KEYWORDS)


def detect_device_category(*texts: str) -> Optional[str]:
    lowered_text = " ".join(texts).lower()
    for device_name, device_data in DEVICE_CATALOG.items():
        aliases = device_data.get("aliases", [])
        if any(alias in lowered_text for alias in aliases):
            return device_name
    return None


# ← RAG: build_prompt ahora acepta contexto opcional del manual
def build_prompt(user_message: str, contexto_rag: str = "") -> str:
    if contexto_rag:
        contexto_bloque = (
            "\n\n[MANUAL CONTEXT — usa esta información para enriquecer la extracción]\n"
            + contexto_rag
        )
    else:
        contexto_bloque = ""

    input_text = f"[INPUT TEXT]\n{user_message.strip()}\n\n[OUTPUT]\n"
    return f"{SYSTEM_PROMPT_BASE}{contexto_bloque}\n\n{input_text}"


def extract_first_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start: index + 1]
    return None


def canonicalize_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in data.items():
        key_norm = re.sub(r"\s+", "_", str(key).strip().lower())
        normalized[key_norm] = value
    return normalized


def normalize_percentage(value: Any) -> Any:
    if value is None:
        return "No especificado"
    text = str(value).strip()
    if not text:
        return "No especificado"
    match = re.search(r"\d+", text)
    if not match:
        return "No especificado"
    return max(0, min(100, int(match.group())))


def normalize_severity(value: Any) -> Any:
    if value is None:
        return "No especificado"
    text = str(value).strip()
    if not text:
        return "No especificado"
    match = re.search(r"[1-5]", text)
    if not match:
        return "No especificado"
    return int(match.group())


def fill_missing_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    filled = dict(data)
    filled.setdefault("aparato", "No especificado")
    filled.setdefault("sintoma", "No especificado")
    filled.setdefault("codigo_error", "No especificado")
    filled.setdefault("causa_probable", "No especificado")
    filled.setdefault("pasos_reparacion", "No especificado")
    filled["porcentaje_certeza"]  = normalize_percentage(filled.get("porcentaje_certeza"))
    filled["grado_peligrosidad"]  = normalize_severity(filled.get("grado_peligrosidad"))
    return filled


def validate_extraction_json(json_text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not json_text:
        return None
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None
    data = canonicalize_keys(data)
    if not isinstance(data, dict):
        return None
    if not {"aparato", "sintoma", "causa_probable", "pasos_reparacion"}.issubset(data.keys()):
        return None
    filled = fill_missing_fields(data)
    if set(filled.keys()) != REQUIRED_KEYS:
        return None
    return filled


def detect_brand(user_message: str, extracted: Dict[str, Any]) -> Optional[str]:
    aparato = str(extracted.get("aparato", ""))
    lowered = f"{user_message} {aparato}".lower()
    for brand in BRAND_KEYWORDS:
        if brand in lowered:
            return brand.capitalize()
    return None


def assess_extraction_consistency(user_message: str, extracted: Dict[str, Any]) -> List[str]:
    device_category = detect_device_category(user_message, str(extracted.get("aparato", "")))
    if not device_category:
        return []
    device_data = DEVICE_CATALOG.get(device_category, {})
    text_to_check = " ".join([
        str(extracted.get("aparato", "")),
        str(extracted.get("sintoma", "")),
        str(extracted.get("causa_probable", "")),
        json.dumps(extracted.get("pasos_reparacion", ""), ensure_ascii=False),
    ]).lower()
    issues: List[str] = []
    for term in device_data.get("incompatible_terms", []):
        if term.lower() in text_to_check:
            issues.append(
                f"Incompatibilidad aparente: el término '{term}' no encaja bien con el aparato '{device_category}'."
            )
    return issues


def build_consistency_notice(issues: List[str]) -> str:
    if not issues:
        return ""
    bullet_lines = "".join(f"<li>{html.escape(issue)}</li>" for issue in issues)
    return (
        "<div style='margin-top:12px;padding:12px;border:1px solid #f59e0b;"
        "background:#fffbeb;border-radius:10px;color:#92400e;'>"
        "<strong>Extracción dudosa:</strong> se detectaron incoherencias en los datos del extractor."
        f"<ul style='margin:8px 0 0 18px;'>{bullet_lines}</ul>"
        "</div>"
    )


def build_severity_warning(user_message: str, extracted: Dict[str, Any]) -> str:
    severity = extracted.get("grado_peligrosidad")
    if severity != 1:
        return ""
    brand = detect_brand(user_message, extracted)
    if brand:
        return (
            "<div style='margin-top:12px;padding:12px;border:1px solid #fecaca;"
            "background:#fff1f2;border-radius:10px;color:#9f1239;'>"
            "<strong>Aviso de seguridad:</strong> El caso se ha clasificado "
            "como peligro crítico. No se recomienda seguir manipulando el "
            f"equipo. Contacta con el soporte técnico oficial de {html.escape(brand)}."
            "</div>"
        )
    return (
        "<div style='margin-top:12px;padding:12px;border:1px solid #fecaca;"
        "background:#fff1f2;border-radius:10px;color:#9f1239;'>"
        "<strong>Aviso de seguridad:</strong> El caso se ha clasificado como "
        "peligro crítico. No se recomienda seguir manipulando el equipo. "
        "Contacta con el soporte técnico oficial de la marca del electrodoméstico."
        "</div>"
    )


def render_json_response(user_message: str, data: Dict[str, Any], issues: List[str]) -> str:
    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    return (
        "<strong>Modo validación del extractor</strong><br>"
        "Respuesta JSON generada por el modelo:<br><br>"
        f"<pre>{html.escape(pretty)}</pre>"
        f"{build_consistency_notice(issues)}"
        f"{build_severity_warning(user_message, data)}"
    )


def build_inconsistent_extraction_response(
    user_message: str, extracted: Dict[str, Any], issues: List[str]
) -> str:
    language   = detect_output_language(user_message)
    appliance  = str(extracted.get("aparato", "el electrodoméstico"))
    symptom    = str(extracted.get("sintoma", "el problema descrito"))
    error_code = str(extracted.get("codigo_error", "No especificado"))
    code_text  = ""
    if error_code and error_code != "No especificado":
        if language == "English":
            code_text = f" The extracted error code was {error_code}."
        else:
            code_text = f" El código extraído fue {error_code}."
    if language == "English":
        return (
            f"We detected an issue affecting the {appliance}: {symptom}.{code_text} "
            "However, some extracted diagnostic details do not fit this type of appliance, "
            "so the automatic diagnosis may be unreliable. For now, the safest option is to avoid "
            "assuming the extracted cause is correct and to continue with conservative checks only. "
            "If the problem persists, contact the brand's official technical support."
        )
    return (
        f"Hemos detectado un problema en {appliance}: {symptom}.{code_text} "
        "Sin embargo, algunos detalles del diagnóstico extraído no encajan bien con este tipo de "
        "electrodoméstico, así que la causa automática puede no ser fiable. Por ahora, lo más prudente "
        "es no dar por buena esa causa y limitarse a comprobaciones seguras. Si el problema persiste, "
        "contacta con el soporte técnico oficial de la marca."
    )


def normalize_steps(steps: Any) -> List[str]:
    if isinstance(steps, list):
        return [str(step).strip() for step in steps if str(step).strip()]
    if isinstance(steps, str) and steps.strip():
        return [steps.strip()]
    return []


def build_structured_assistant_response(user_message: str, extracted: Dict[str, Any]) -> str:
    language   = detect_output_language(user_message)
    appliance  = str(extracted.get("aparato", "el electrodoméstico"))
    symptom    = str(extracted.get("sintoma", "el problema descrito"))
    cause      = str(extracted.get("causa_probable", "No especificado"))
    certainty  = extracted.get("porcentaje_certeza", "No especificado")
    error_code = str(extracted.get("codigo_error", "No especificado"))
    steps      = normalize_steps(extracted.get("pasos_reparacion"))

    if language == "English":
        parts = [f"We detected a problem affecting the {appliance}: {symptom}."]
        if error_code != "No especificado":
            parts.append(f"The extracted error code is {error_code}.")
        if cause != "No especificado":
            parts.append(f"The most likely extracted cause is: {cause}.")
        if certainty != "No especificado":
            parts.append(f"Estimated confidence: {certainty}%.")
        if steps:
            joined = " ".join(f"{i+1}. {s}." for i, s in enumerate(steps[:3]))
            parts.append(f"Suggested next steps: {joined}")
        else:
            parts.append("No safe repair steps were extracted with enough confidence.")
        return " ".join(parts)

    parts = [f"Hemos detectado un problema en {appliance}: {symptom}."]
    if error_code != "No especificado":
        parts.append(f"El código extraído es {error_code}.")
    if cause != "No especificado":
        parts.append(f"La causa probable extraída es: {cause}.")
    if certainty != "No especificado":
        parts.append(f"Certeza estimada: {certainty}%.")
    if steps:
        joined = " ".join(f"{i+1}. {s}." for i, s in enumerate(steps[:3]))
        parts.append(f"Pasos sugeridos: {joined}")
    else:
        parts.append("No se han extraído pasos de reparación seguros con suficiente confianza.")
    return " ".join(parts)


def call_ollama_generate(model: str, prompt: str, num_predict: int = 350) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "top_p": 0.9, "num_predict": num_predict},
    }
    response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return str(data.get("response", ""))


def build_assistant_prompt(
    user_message: str,
    extracted: Dict[str, Any],
    issues: List[str],
    contexto_rag: str = "",      # ← RAG: contexto del manual
) -> str:
    extracted_json = json.dumps(extracted, ensure_ascii=False, indent=2)
    target_language = detect_output_language(user_message)
    issues_block = "\n".join(f"- {issue}" for issue in issues) or "- No issues detected."

    # ← RAG: añade el contexto del manual si está disponible
    rag_block = ""
    if contexto_rag:
        rag_block = f"\n[MANUAL CONTEXT — fragmentos relevantes del manual oficial]\n{contexto_rag}\n"

    return f"""You are a multilingual appliance support assistant.
Answer only in {target_language}. Do not switch languages.
Be concise, practical, and safety-aware.
Do not invent facts not grounded in the extracted data or the manual context.
If the extraction contains suspicious or inconsistent details, do not present them as confirmed facts.
If the extracted severity is 1, explicitly warn the user to stop manipulating the appliance.
{rag_block}
[USER_MESSAGE]
{user_message}

[EXTRACTED_JSON]
{extracted_json}

[CONSISTENCY_CHECK]
{issues_block}

[TASK]
Write a clear support reply for the end user in plain text.
If key fields are missing or unreliable, say so plainly instead of inventing details.
Do not output JSON."""


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat() -> Any:
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    mode    = str(payload.get("mode", DEFAULT_MODE)).strip().lower() or DEFAULT_MODE
    if mode not in {"extractor", "assistant"}:
        mode = DEFAULT_MODE

    if not message:
        return jsonify({"error": "Falta el campo 'message'"}), 400

    if not is_in_domain(message):
        return jsonify({
            "response": (
                "<strong>Fuera de dominio.</strong><br>"
                "Este prototipo solo está preparado ahora mismo para "
                "consultas sobre averías y soporte técnico de electrodomésticos."
            ),
            "mode": mode,
        })

    # ── 1. RAG: buscar contexto en Milvus ────────────────────────────────────
    subject_hint = detect_device_category(message) or ""
    hits         = buscar_en_milvus(message, subject=subject_hint)
    contexto_rag = construir_contexto_rag(hits)
    rag_usado    = bool(hits)   # para incluirlo en la respuesta de debug

    # ── 2. Llamar al extractor con el contexto enriquecido ───────────────────
    try:
        raw_response = call_ollama_generate(
            EXTRACTOR_MODEL,
            build_prompt(message, contexto_rag),   # ← RAG: prompt enriquecido
            num_predict=350,
        )
    except requests.RequestException as exc:
        return jsonify({"error": f"Error al consultar Ollama: {exc}"}), 502

    json_text = extract_first_json(raw_response)
    extracted = validate_extraction_json(json_text)

    if extracted is None:
        fallback = (
            "<strong>El extractor no devolvió un JSON válido.</strong><br>"
            "Salida bruta del modelo para diagnóstico:<br><br>"
            f"<pre>{html.escape(raw_response)}</pre>"
        )
        return jsonify({"response": fallback, "mode": mode})

    issues = assess_extraction_consistency(message, extracted)

    # ── 3. Modo extractor: devuelve el JSON directamente ─────────────────────
    if mode == "extractor":
        return jsonify({
            "response":    render_json_response(message, extracted, issues),
            "mode":        mode,
            "rag_used":    rag_usado,
            "rag_sources": [{"source": h["source"], "page": h["page"], "score": h["score"]} for h in hits],
        })

    # ── 4. Modo assistant: genera respuesta en lenguaje natural ──────────────
    if issues:
        assistant_text = build_inconsistent_extraction_response(message, extracted, issues)
    elif extracted.get("grado_peligrosidad") == 1:
        assistant_text = build_structured_assistant_response(message, extracted)
    else:
        try:
            assistant_text = call_ollama_generate(
                RESPONSE_MODEL,
                build_assistant_prompt(message, extracted, issues, contexto_rag),  # ← RAG
                num_predict=260,
            )
        except requests.RequestException:
            assistant_text = build_structured_assistant_response(message, extracted)

    # ── 5. Construir HTML de respuesta ────────────────────────────────────────
    # Badge RAG opcional (indica al usuario que se usó el manual)
    rag_badge = ""
    if rag_usado:
        fuentes = ", ".join(
            f"{h['source']} pág.{h['page']}" for h in hits[:2]
        )
        rag_badge = (
            "<div style='margin-top:10px;padding:8px 12px;border:1px solid #a7f3d0;"
            "background:#ecfdf5;border-radius:8px;color:#065f46;font-size:0.85em;'>"
            f"📄 Respuesta enriquecida con el manual — fuentes: {html.escape(fuentes)}"
            "</div>"
        )

    response_html = (
        f"<div>{html.escape(assistant_text).replace(chr(10), '<br>')}</div>"
        f"{build_consistency_notice(issues)}"
        f"{build_severity_warning(message, extracted)}"
        f"{rag_badge}"
    )

    return jsonify({
        "response":      response_html,
        "mode":          mode,
        "extractor_json": extracted,
        "rag_used":      rag_usado,
        "rag_sources":   [{"source": h["source"], "page": h["page"], "score": h["score"]} for h in hits],
    })


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({
        "status":          "ok",
        "extractor_model": EXTRACTOR_MODEL,
        "response_model":  RESPONSE_MODEL,
        "default_mode":    DEFAULT_MODE,
        "milvus_ok":       _milvus_ok,
        "milvus_db":       MILVUS_DB,
        "collection":      COLLECTION_NAME,
    })


# ─────────────────────────────────────────────────────────────────────────────
# ARRANQUE
# ─────────────────────────────────────────────────────────────────────────────

# Inicializar Milvus al arrancar (no bloquea si falla)
_init_milvus()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
