#!/usr/bin/env python3
"""
Carga un JSON de averías (generado por pdf_to_json.py) en una base vectorial Milvus Lite.

Uso:
    python json_to_milvus.py averias_reparacion.json
    python json_to_milvus.py averias_reparacion.json --milvus-db milvus.db
    python json_to_milvus.py averias_reparacion.json --milvus-db milvus.db --append
    python json_to_milvus.py averias_reparacion.json --embed-dim 384

Dependencias:
    pip install "pymilvus[milvus-lite]"

Notas:
    - Milvus Lite guarda la base de datos en un único fichero .db local.
    - Los vectores se generan con un hash determinista (sin APIs externas).
    - Por defecto borra y recrea la colección; usa --append para añadir registros.
    - Solo se insertan campos presentes en el registro; nunca se inventa un valor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

VECTOR_FIELD = "vector"
DEFAULT_COLLECTION = "reparaciones"
DEFAULT_EMBED_DIM = 384
MAX_MILVUS_VARCHAR = 8192

# Campos opcionales y sus tipos/límites.
# Solo se añaden al esquema y al payload si aparecen en al menos un registro.
OPTIONAL_VARCHAR_FIELDS: dict[str, int] = {
    "aparato":          512,
    "causa_probable":   2048,
    "codigo_error":     128,
    "fuente_pdf":       2048,
    "pasos_reparacion": 4096,
    "sintoma":          1024,
    "marca":            128,
    "modelo":           256,
}
OPTIONAL_INT_FIELDS: tuple[str, ...] = (
    "grado_peligrosidad",
    "pagina",
    "porcentaje_certeza",
)


# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------

def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def norm(value: str) -> str:
    return strip_accents(value or "").casefold()


def clean_field(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def truncate(value: str, max_len: int = MAX_MILVUS_VARCHAR) -> str:
    value = clean_field(value)
    if len(value) <= max_len:
        return value
    cut = value[: max_len - 1].rsplit(" ", 1)[0]
    return cut if cut else value[: max_len - 1]


# ---------------------------------------------------------------------------
# Embedding por hash (sin APIs externas)
# ---------------------------------------------------------------------------

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_/-]{2,}")


def hash_embedding(text: str, dim: int = DEFAULT_EMBED_DIM) -> list[float]:
    tokens = TOKEN_PATTERN.findall(norm(text))
    vector = [0.0] * dim
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, byteorder="little", signed=False)
        index = number % dim
        sign = -1.0 if number & 1 else 1.0
        vector[index] += sign
    length = math.sqrt(sum(v * v for v in vector))
    if not length:
        return vector
    return [v / length for v in vector]


# ---------------------------------------------------------------------------
# Detección de campos presentes en el dataset
# ---------------------------------------------------------------------------

def detect_present_fields(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """
    Recorre todos los registros y devuelve solo los campos que tienen
    al menos un valor no nulo / no vacío.
    Retorna (varchar_fields, int_fields) en el orden de OPTIONAL_*_FIELDS.
    """
    present_varchar: list[str] = []
    present_int: list[str] = []

    for field in OPTIONAL_VARCHAR_FIELDS:
        for r in records:
            if clean_field(r.get(field)):
                present_varchar.append(field)
                break

    for field in OPTIONAL_INT_FIELDS:
        for r in records:
            val = r.get(field)
            if val is not None:
                present_int.append(field)
                break

    return present_varchar, present_int


# ---------------------------------------------------------------------------
# Payload Milvus — sin valores inventados
# ---------------------------------------------------------------------------

def milvus_payload(
    record: dict[str, Any],
    embed_dim: int,
    present_varchar: list[str],
    present_int: list[str],
) -> dict[str, Any]:
    # El vector se construye con los campos de texto disponibles
    text_for_vector = " ".join(
        clean_field(record.get(field, ""))
        for field in ("aparato", "sintoma", "codigo_error", "causa_probable", "pasos_reparacion")
    )

    # El id se genera solo si el registro no lo trae
    record_id = record.get("id")
    if not record_id:
        record_id = hashlib.sha1(
            text_for_vector.encode("utf-8", errors="ignore")
        ).hexdigest()[:24]

    payload: dict[str, Any] = {
        "id": truncate(str(record_id), 64),
        VECTOR_FIELD: hash_embedding(text_for_vector, embed_dim),
    }

    # MilvusLite no soporta nullable: los campos del esquema deben estar siempre
    # en el payload. VARCHAR ausente → ""  (cadena vacía, sin inventar contenido)
    # INT64 ausente → -1  (centinela "no informado"; nunca es valor legítimo)
    for field in present_varchar:
        val = clean_field(record.get(field))
        payload[field] = truncate(val, OPTIONAL_VARCHAR_FIELDS[field])

    for field in present_int:
        val = record.get(field)
        if val is None:
            payload[field] = -1
        else:
            try:
                payload[field] = int(val)
            except (TypeError, ValueError):
                payload[field] = -1

    return payload


# ---------------------------------------------------------------------------
# Milvus: creación de colección e inserción
# ---------------------------------------------------------------------------

def import_milvus():
    try:
        from pymilvus import DataType, MilvusClient  # type: ignore
        return MilvusClient, DataType
    except ImportError as exc:
        raise RuntimeError(
            'Falta pymilvus. Instala con: pip install "pymilvus[milvus-lite]"'
        ) from exc


def ensure_collection(
    client: Any,
    DataType: Any,
    collection_name: str,
    embed_dim: int,
    append: bool,
    present_varchar: list[str],
    present_int: list[str],
) -> None:
    if client.has_collection(collection_name):
        if append:
            print(f"[INFO] Colección '{collection_name}' ya existe — modo append activado.")
            return
        print(f"[INFO] Borrando colección existente '{collection_name}'...")
        client.drop_collection(collection_name)

    print(f"[INFO] Creando colección '{collection_name}' con dim={embed_dim}...")
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

    # Campos obligatorios
    schema.add_field("id",        DataType.VARCHAR,      is_primary=True, max_length=64)
    schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=embed_dim)

    # Solo los campos realmente presentes en los datos
    # MilvusLite no soporta nullable: los VARCHAR usan "" y los INT64 usan -1 como centinela
    for field in present_varchar:
        schema.add_field(field, DataType.VARCHAR, max_length=OPTIONAL_VARCHAR_FIELDS[field])

    for field in present_int:
        schema.add_field(field, DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name=VECTOR_FIELD, index_type="FLAT", metric_type="COSINE")
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    print(f"[OK] Colección '{collection_name}' creada.")
    if present_varchar:
        print(f"     Campos VARCHAR: {', '.join(present_varchar)}")
    if present_int:
        print(f"     Campos INT64:   {', '.join(present_int)}")


def record_has_real_data(record: dict[str, Any]) -> bool:
    """Devuelve True si el registro contiene al menos un campo con valor útil."""
    for field in ("aparato", "sintoma", "codigo_error", "causa_probable", "pasos_reparacion"):
        if clean_field(record.get(field)):
            return True
    return False


def write_milvus(
    records: list[dict[str, Any]],
    db_path: Path,
    collection_name: str,
    embed_dim: int,
    append: bool,
    batch_size: int = 100,
) -> None:
    if not records:
        print("[WARN] No hay registros que insertar.")
        return

    present_varchar, present_int = detect_present_fields(records)
    print(f"[INFO] Campos VARCHAR detectados: {present_varchar or '(ninguno)'}")
    print(f"[INFO] Campos INT64   detectados: {present_int or '(ninguno)'}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    MilvusClient, DataType = import_milvus()
    client = MilvusClient(str(db_path))
    try:
        ensure_collection(
            client, DataType, collection_name, embed_dim, append,
            present_varchar, present_int,
        )
        total = len(records)
        inserted = 0
        for start in range(0, total, batch_size):
            batch = [
                milvus_payload(r, embed_dim, present_varchar, present_int)
                for r in records[start: start + batch_size]
            ]
            client.insert(collection_name=collection_name, data=batch)
            inserted += len(batch)
            print(f"    Insertados {inserted}/{total} registros...", end="\r")
        print()
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Carga un JSON de averías en una base vectorial Milvus Lite (.db)."
    )
    parser.add_argument(
        "input_json",
        help="Ruta al fichero JSON generado por pdf_to_json.py.",
    )
    parser.add_argument(
        "--milvus-db",
        default="milvus.db",
        help="Ruta del fichero Milvus Lite de salida. Por defecto: milvus.db",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Nombre de la colección en Milvus. Por defecto: {DEFAULT_COLLECTION}",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=DEFAULT_EMBED_DIM,
        help=f"Dimensión del vector hash. Por defecto: {DEFAULT_EMBED_DIM}",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="No borra la colección si ya existe; inserta registros nuevos.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Registros por lote de inserción. Por defecto: 100",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_json = Path(args.input_json)
    milvus_db = Path(args.milvus_db)

    if not input_json.exists():
        print(f"[ERROR] Fichero JSON no encontrado: {input_json}", file=sys.stderr)
        return 2

    print(f"[INFO] Leyendo JSON: {input_json}")
    try:
        records: list[dict[str, Any]] = json.loads(input_json.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERROR] No se pudo leer el JSON: {exc}", file=sys.stderr)
        return 2

    if not isinstance(records, list):
        print("[ERROR] El JSON debe ser una lista de registros.", file=sys.stderr)
        return 2

    original_count = len(records)
    records = [r for r in records if record_has_real_data(r)]
    print(f"[INFO] Registros leídos:          {original_count}")
    print(f"[INFO] Registros con datos reales: {len(records)}")

    if not records:
        print("[WARN] Ningún registro tiene datos útiles. No se escribe nada.")
        return 0

    try:
        write_milvus(
            records,
            db_path=milvus_db,
            collection_name=args.collection,
            embed_dim=args.embed_dim,
            append=args.append,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"[ERROR] Fallo al escribir en Milvus: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] Milvus DB guardado en: {milvus_db.resolve()}")
    print(f"[OK] Colección: {args.collection}")
    print(f"[OK] Registros insertados: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())