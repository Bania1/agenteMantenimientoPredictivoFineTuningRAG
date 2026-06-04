from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, request

try:
    from pymilvus import MilvusClient
except Exception:
    MilvusClient = None

REPO_ROOT = Path(__file__).resolve().parents[3]
DOMAIN_CATALOG_PATH = REPO_ROOT / "config" / "domain" / "domain_catalog.json"
MILVUS_DB_PATH = Path(
    os.getenv(
        "MILVUS_DB",
        str(REPO_ROOT / "data" / "milvus-lite" / "reparaciones-lite" / "milvus.db"),
    )
)
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "reparaciones")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
OLLAMA_GENERATE_URL = os.getenv("OLLAMA_GENERATE_URL", "http://127.0.0.1:11434/api/generate")
EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "qwen-fusionado")
RESPONSE_MODEL = os.getenv("RESPONSE_MODEL", "llama3.2:1b")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "600"))
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "assistant")

app = Flask(__name__)

REQUIRED_KEYS = {
    "aparato",
    "sintoma",
    "codigo_error",
    "causa_probable",
    "pasos_reparacion",
    "porcentaje_certeza",
    "grado_peligrosidad",
}

SYSTEM_PROMPT = """[GOAL]
Eres un motor de extraccion de datos de alta precision especializado en soporte tecnico de electrodomesticos. Tu tarea es analizar descripciones de fallos o manuales y extraer la informacion tecnica clave con objetividad.

[OUTPUT FORMAT]
Responde UNICAMENTE con un objeto JSON valido y plano que contenga exactamente estas 7 claves:
"aparato", "sintoma", "codigo_error", "causa_probable", "pasos_reparacion", "porcentaje_certeza", "grado_peligrosidad".

[ESCALA DE PELIGROSIDAD]
Usa esta escala para el campo "grado_peligrosidad":
- 1: Peligro Critico (Riesgo de Incendio o Arqueo Electrico) - chispas, olor a quemado, sobrecalentamiento extremo.
- 2: Peligro Alto (Sistemas de Potencia e Inverter) - fallos de magnetron, inverter, sobrecalentamiento intermitente.
- 3: Peligro Medio (Componentes Internos de Tension) - fusibles, diodos, sensores de temperatura/humedad.
- 4: Peligro Bajo (Interfaz y mecanica de puerta) - panel de control, mecanismos de cierre, resistencias de grill.
- 5: Peligro Minimo (Configuracion y accesorios) - bloqueos de seguridad, accesorios externos, errores de software."""

BRAND_KEYWORDS = [
    "bosch",
    "balay",
    "siemens",
    "lg",
    "samsung",
    "whirlpool",
    "beko",
    "teka",
    "indesit",
    "electrolux",
    "aeg",
    "zanussi",
    "haier",
    "hisense",
    "miele",
]

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_/-]{2,}")

_milvus_client: MilvusClient | None = None
_milvus_ok = False
_milvus_error = "No inicializado"


def append_follow_up_question(text: str, language: str) -> str:
    closing = "Do you need help with anything else?" if language == "English" else "¿Necesitas que te ayude con algo mas?"
    text = text.strip()
    if not text:
        return closing
    normalized_text = norm(text)
    normalized_closings = {norm(closing)}
    if language != "English":
        normalized_closings.add(norm("¿Necesitas ayuda con algo más?"))
        normalized_closings.add(norm("¿Necesitas que te ayude con algo más?"))
    if any(option in normalized_text for option in normalized_closings):
        return text
    separator = " " if text.endswith((".", "!", "?")) else ". "
    return f"{text}{separator}{closing}"


def load_domain_catalog() -> dict[str, Any]:
    with DOMAIN_CATALOG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


DOMAIN_CATALOG = load_domain_catalog()
DEVICE_CATALOG = DOMAIN_CATALOG.get("devices", {})


def detect_output_language(user_message: str) -> str:
    lowered = user_message.lower()
    english_markers = [
        " the ",
        " freezer",
        " fridge",
        " refrigerator",
        " washing machine",
        " dishwasher",
        " oven",
        " dryer",
        " microwave",
        " not ",
        " doesn't ",
        " does not ",
    ]
    if any(marker in f" {lowered} " for marker in english_markers):
        return "English"
    return "Spanish"


def get_domain_keywords() -> list[str]:
    keywords: list[str] = []
    for device_data in DEVICE_CATALOG.values():
        keywords.extend(device_data.get("aliases", []))
    return keywords


DOMAIN_KEYWORDS = get_domain_keywords()


def is_in_domain(user_message: str) -> bool:
    lowered = user_message.lower()
    return any(keyword in lowered for keyword in DOMAIN_KEYWORDS)


def detect_device_category(*texts: str) -> str | None:
    lowered_text = " ".join(texts).lower()
    for device_name, device_data in DEVICE_CATALOG.items():
        aliases = device_data.get("aliases", [])
        if any(alias in lowered_text for alias in aliases):
            return device_name
    return None


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


def hash_embedding(text: str, dim: int = 384) -> list[float]:
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


def inicializar_milvus() -> None:
    global _milvus_client, _milvus_ok, _milvus_error
    if MilvusClient is None:
        _milvus_ok = False
        _milvus_error = "pymilvus no esta instalado en el entorno actual"
        return
    try:
        if not MILVUS_DB_PATH.exists():
            _milvus_ok = False
            _milvus_error = f"No existe la base vectorial en {MILVUS_DB_PATH}"
            return
        _milvus_client = MilvusClient(uri=str(MILVUS_DB_PATH))
        _milvus_client.load_collection(MILVUS_COLLECTION)
        _milvus_ok = True
        _milvus_error = "ok"
    except Exception as exc:
        _milvus_ok = False
        _milvus_error = str(exc)
        _milvus_client = None


def get_milvus_client() -> MilvusClient | None:
    if _milvus_client is None and not _milvus_ok:
        inicializar_milvus()
    return _milvus_client


def hit_encaja_con_categoria(hit_aparato: str, categoria: str | None) -> bool:
    if not categoria:
        return True
    aliases = DEVICE_CATALOG.get(categoria, {}).get("aliases", [])
    aparato_norm = norm(hit_aparato)
    return any(norm(alias) in aparato_norm for alias in aliases)


def rerank_hits_por_aparato(hits: list[dict[str, Any]], aparato_hint: str | None) -> list[dict[str, Any]]:
    if not aparato_hint:
        return hits

    hits_compatibles = [
        hit for hit in hits if hit_encaja_con_categoria(str(hit.get("aparato", "")), aparato_hint)
    ]
    if not hits_compatibles:
        return []

    return sorted(
        hits_compatibles,
        key=lambda hit: norm(str(hit.get("aparato", ""))),
    )


def buscar_fragmentos_rag(consulta: str, aparato_hint: str | None = None, top_k: int = RAG_TOP_K) -> list[dict[str, Any]]:
    global _milvus_ok, _milvus_error
    client = get_milvus_client()
    if not _milvus_ok or client is None:
        return []
    try:
        vector = hash_embedding(consulta)
        raw_results = client.search(
            collection_name=MILVUS_COLLECTION,
            data=[vector],
            limit=top_k,
            output_fields=[
                "aparato",
                "causa_probable",
                "codigo_error",
                "pasos_reparacion",
                "grado_peligrosidad",
                "porcentaje_certeza",
            ],
        )
        hits: list[dict[str, Any]] = []
        for hit in raw_results[0]:
            entity = hit.get("entity", hit)
            hits.append(
                {
                    "score": round(float(hit.get("distance", 0.0)), 4),
                    "aparato": entity.get("aparato", ""),
                    "causa_probable": entity.get("causa_probable", ""),
                    "codigo_error": entity.get("codigo_error", ""),
                    "pasos_reparacion": entity.get("pasos_reparacion", ""),
                    "grado_peligrosidad": entity.get("grado_peligrosidad", -1),
                    "porcentaje_certeza": entity.get("porcentaje_certeza", -1),
                }
            )
        return rerank_hits_por_aparato(hits, aparato_hint)
    except Exception as exc:
        _milvus_ok = False
        _milvus_error = str(exc)
        return []


def construir_contexto_rag(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return ""
    bloques = []
    for index, hit in enumerate(hits, 1):
        bloques.append(
            f"[CASO RECUPERADO {index}]\n"
            f"Aparato: {hit.get('aparato', 'No especificado')}\n"
            f"Causa probable: {hit.get('causa_probable', 'No especificado')}\n"
            f"Codigo de error: {hit.get('codigo_error', 'No especificado')}\n"
            f"Pasos de reparacion: {hit.get('pasos_reparacion', 'No especificado')}\n"
            f"Grado de peligrosidad: {hit.get('grado_peligrosidad', 'No especificado')}\n"
            f"Porcentaje de certeza: {hit.get('porcentaje_certeza', 'No especificado')}\n"
            f"Similitud: {hit.get('score', 0.0)}"
        )
    return "\n\n".join(bloques)


def build_prompt(user_message: str, contexto_rag: str = "") -> str:
    contexto_bloque = ""
    if contexto_rag:
        contexto_bloque = (
            "\n\n[CONTEXTO RECUPERADO DE LA BASE DE AVERIAS]\n"
            f"{contexto_rag}"
        )
    input_text = f"[INPUT TEXT]\n{user_message.strip()}\n\n[OUTPUT]\n"
    return f"{SYSTEM_PROMPT}{contexto_bloque}\n\n{input_text}"


def extract_first_json(text: str) -> str | None:
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
                return text[start : index + 1]
    return None


def sanitize_extraction_json_text(json_text: str) -> str:
    text = json_text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = re.sub(r"-\n\s+", "", text)

    pieces: list[str] = []
    inside_string = False
    escape = False
    for char in text:
        if inside_string:
            if escape:
                pieces.append(char)
                escape = False
                continue
            if char == "\\":
                pieces.append(char)
                escape = True
                continue
            if char == '"':
                pieces.append(char)
                inside_string = False
                continue
            if char == "\n":
                pieces.append(" ")
                continue
            pieces.append(char)
            continue
        pieces.append(char)
        if char == '"':
            inside_string = True

    text = "".join(pieces)
    text = re.sub(r'[ \t]{2,}', " ", text)
    return text


def loads_json_lenient(json_text: str) -> dict[str, Any] | None:
    sanitized = sanitize_extraction_json_text(json_text)
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        return None


def canonicalize_keys(data: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in data.items():
        key_norm = re.sub(r"\s+", "_", str(key).strip().lower())
        normalized[key_norm] = value
    return normalized


def normalize_percentage(value: Any) -> int | str:
    if value is None:
        return "No especificado"
    text = str(value).strip()
    if not text:
        return "No especificado"
    match = re.search(r"\d+", text)
    if not match:
        return "No especificado"
    return max(0, min(100, int(match.group())))


def normalize_severity(value: Any) -> int | str:
    if value is None:
        return "No especificado"
    text = str(value).strip()
    if not text:
        return "No especificado"
    match = re.search(r"[1-5]", text)
    if not match:
        return "No especificado"
    return int(match.group())


def normalize_steps_value(value: Any) -> list[str] | str:
    if value is None:
        return "No especificado"
    if isinstance(value, list):
        cleaned = [clean_field(step) for step in value if clean_field(step)]
        return cleaned or "No especificado"
    if isinstance(value, str):
        cleaned = clean_field(value)
        if not cleaned:
            return "No especificado"
        parts = [segment.strip(" -") for segment in re.split(r"\s*(?:\d+\.\s+|;\s+|\|\s+)\s*", cleaned) if segment.strip(" -")]
        if len(parts) > 1:
            return parts
        return cleaned
    cleaned = clean_field(value)
    return cleaned or "No especificado"


def fill_missing_fields(data: dict[str, Any]) -> dict[str, Any]:
    filled = dict(data)
    filled.setdefault("aparato", "No especificado")
    filled.setdefault("sintoma", "No especificado")
    filled.setdefault("codigo_error", "No especificado")
    filled.setdefault("causa_probable", "No especificado")
    filled.setdefault("pasos_reparacion", "No especificado")
    filled["aparato"] = clean_field(filled.get("aparato"))
    filled["sintoma"] = clean_field(filled.get("sintoma"))
    filled["codigo_error"] = clean_field(filled.get("codigo_error")) or "No especificado"
    filled["causa_probable"] = clean_field(filled.get("causa_probable")) or "No especificado"
    filled["pasos_reparacion"] = normalize_steps_value(filled.get("pasos_reparacion"))
    filled["porcentaje_certeza"] = normalize_percentage(filled.get("porcentaje_certeza"))
    filled["grado_peligrosidad"] = normalize_severity(filled.get("grado_peligrosidad"))
    return filled


def validate_extraction_json(json_text: str | None) -> dict[str, Any] | None:
    if not json_text:
        return None
    data = loads_json_lenient(json_text)
    if data is None:
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


def detect_brand(user_message: str, extracted: dict[str, Any]) -> str | None:
    aparato = str(extracted.get("aparato", ""))
    lowered = f"{user_message} {aparato}".lower()
    for brand in BRAND_KEYWORDS:
        if brand in lowered:
            return brand.capitalize()
    return None


def assess_extraction_consistency(user_message: str, extracted: dict[str, Any]) -> list[str]:
    device_category = detect_device_category(user_message, str(extracted.get("aparato", "")))
    if not device_category:
        return []
    device_data = DEVICE_CATALOG.get(device_category, {})
    text_to_check = " ".join(
        [
            str(extracted.get("aparato", "")),
            str(extracted.get("sintoma", "")),
            str(extracted.get("causa_probable", "")),
            json.dumps(extracted.get("pasos_reparacion", ""), ensure_ascii=False),
        ]
    ).lower()
    issues: list[str] = []
    for term in device_data.get("incompatible_terms", []):
        if term.lower() in text_to_check:
            issues.append(
                f"Incompatibilidad aparente: el termino '{term}' no encaja bien con el aparato '{device_category}'."
            )
    return issues


def build_consistency_notice(issues: list[str]) -> str:
    if not issues:
        return ""
    bullet_lines = "".join(f"<li>{html.escape(issue)}</li>" for issue in issues)
    return (
        "<div style='margin-top:12px;padding:12px;border:1px solid #f59e0b;"
        "background:#fffbeb;border-radius:10px;color:#92400e;'>"
        "<strong>Extraccion dudosa:</strong> se detectaron incoherencias en los datos del extractor."
        f"<ul style='margin:8px 0 0 18px;'>{bullet_lines}</ul>"
        "</div>"
    )


def build_severity_warning(user_message: str, extracted: dict[str, Any]) -> str:
    severity = extracted.get("grado_peligrosidad")
    if severity != 1:
        return ""
    brand = detect_brand(user_message, extracted)
    if brand:
        return (
            "<div style='margin-top:12px;padding:12px;border:1px solid #fecaca;"
            "background:#fff1f2;border-radius:10px;color:#9f1239;'>"
            "<strong>Aviso de seguridad:</strong> El caso se ha clasificado "
            "como peligro critico. No se recomienda seguir manipulando el "
            f"equipo. Contacta con el soporte tecnico oficial de {html.escape(brand)}."
            "</div>"
        )
    return (
        "<div style='margin-top:12px;padding:12px;border:1px solid #fecaca;"
        "background:#fff1f2;border-radius:10px;color:#9f1239;'>"
        "<strong>Aviso de seguridad:</strong> El caso se ha clasificado como "
        "peligro critico. No se recomienda seguir manipulando el equipo. "
        "Contacta con el soporte tecnico oficial de la marca del electrodomestico."
        "</div>"
    )


def build_rag_notice(rag_hits: list[dict[str, Any]]) -> str:
    if not rag_hits:
        return ""
    elementos = "".join(
        f"<li>{html.escape(str(hit.get('aparato', 'Caso recuperado')))} - similitud {hit.get('score', 0.0)}</li>"
        for hit in rag_hits[:3]
    )
    return (
        "<div style='margin-top:12px;padding:12px;border:1px solid #86efac;"
        "background:#f0fdf4;border-radius:10px;color:#166534;'>"
        "<strong>RAG activo:</strong> se han recuperado averias similares desde Milvus Lite."
        f"<ul style='margin:8px 0 0 18px;'>{elementos}</ul>"
        "</div>"
    )


def build_response_section(title: str, body: str) -> str:
    return (
        "<div style='margin-top:12px;'>"
        f"<div style='font-size:12px;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;color:#64748b;margin-bottom:6px;'>{html.escape(title)}</div>"
        f"<div style='color:#0f172a;line-height:1.6;'>{body}</div>"
        "</div>"
    )


def render_json_response(
    user_message: str,
    data: dict[str, Any],
    issues: list[str],
    rag_hits: list[dict[str, Any]],
) -> str:
    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    return (
        "<strong>Modo validacion del extractor</strong><br>"
        "Respuesta JSON generada por el modelo:<br><br>"
        f"<pre>{html.escape(pretty)}</pre>"
        f"{build_consistency_notice(issues)}"
        f"{build_severity_warning(user_message, data)}"
    )


def build_inconsistent_extraction_response(
    user_message: str,
    extracted: dict[str, Any],
    issues: list[str],
) -> str:
    language = detect_output_language(user_message)
    appliance = str(extracted.get("aparato", "el electrodomestico"))
    symptom = str(extracted.get("sintoma", "el problema descrito"))
    error_code = str(extracted.get("codigo_error", "No especificado"))
    code_text = ""
    if error_code and error_code != "No especificado":
        if language == "English":
            code_text = f" The extracted error code was {error_code}."
        else:
            code_text = f" El codigo extraido fue {error_code}."
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
        "Sin embargo, algunos detalles del diagnostico extraido no encajan bien con este tipo de "
        "electrodomestico, asi que la causa automatica puede no ser fiable. Por ahora, lo mas prudente "
        "es no dar por buena esa causa y limitarse a comprobaciones seguras. Si el problema persiste, "
        "contacta con el soporte tecnico oficial de la marca."
    )


def normalize_steps(steps: Any) -> list[str]:
    if isinstance(steps, list):
        return [str(step).strip() for step in steps if str(step).strip()]
    if isinstance(steps, str) and steps.strip():
        return [steps.strip()]
    return []


def build_structured_assistant_response(user_message: str, extracted: dict[str, Any]) -> str:
    language = detect_output_language(user_message)
    appliance = str(extracted.get("aparato", "el electrodomestico"))
    symptom = str(extracted.get("sintoma", "el problema descrito"))
    cause = str(extracted.get("causa_probable", "No especificado"))
    certainty = extracted.get("porcentaje_certeza", "No especificado")
    error_code = str(extracted.get("codigo_error", "No especificado"))
    steps = normalize_steps(extracted.get("pasos_reparacion"))
    if language == "English":
        parts = [f"We detected a problem affecting the {appliance}: {symptom}."]
        if error_code != "No especificado":
            parts.append(f"The extracted error code is {error_code}.")
        if cause != "No especificado":
            parts.append(f"The most likely extracted cause is: {cause}.")
        if certainty != "No especificado":
            parts.append(f"Estimated confidence: {certainty}%.")
        if steps:
            joined_steps = " ".join(f"{index + 1}. {step}." for index, step in enumerate(steps[:3]))
            parts.append(f"Suggested next steps: {joined_steps}")
        else:
            parts.append("No safe repair steps were extracted with enough confidence.")
        parts.append("If the issue persists or something does not match what you observe, contact the brand's official technical support.")
        return append_follow_up_question(" ".join(parts), language)
    parts = [f"Hemos detectado un problema en {appliance}: {symptom}."]
    if error_code != "No especificado":
        parts.append(f"El codigo extraido es {error_code}.")
    if cause != "No especificado":
        parts.append(f"La causa probable extraida es: {cause}.")
    if certainty != "No especificado":
        parts.append(f"Certeza estimada: {certainty}%.")
    if steps:
        joined_steps = " ".join(f"{index + 1}. {step}." for index, step in enumerate(steps[:3]))
        parts.append(f"Pasos sugeridos: {joined_steps}")
    else:
        parts.append("No se han extraido pasos de reparacion seguros con suficiente confianza.")
    parts.append("Si el problema sigue igual o no encaja con lo que ves en el equipo, lo mas recomendable es contactar con el soporte tecnico oficial de la marca.")
    return append_follow_up_question(" ".join(parts), language)


def build_rag_supported_fallback(user_message: str, rag_hits: list[dict[str, Any]]) -> str:
    language = detect_output_language(user_message)
    if not rag_hits:
        if language == "English":
            return append_follow_up_question(
                "I could not extract a reliable diagnosis from the model and I do not have enough supporting context to answer safely. The most sensible option is to contact the brand's official technical support.",
                language,
            )
        return append_follow_up_question(
            "No he podido extraer un diagnostico fiable del modelo y tampoco tengo contexto suficiente para responder con seguridad. Lo mas recomendable es contactar con el soporte tecnico oficial de la marca.",
            language,
        )

    top_hit = rag_hits[0]
    aparato = clean_field(top_hit.get("aparato")) or "el electrodomestico"
    causa = clean_field(top_hit.get("causa_probable")) or "No especificado"
    pasos = normalize_steps(top_hit.get("pasos_reparacion"))
    if language == "English":
        parts = [
            f"I could not validate the extractor output cleanly, but I did recover a similar case related to {aparato}.",
            f"The closest likely cause in the knowledge base is: {causa}.",
        ]
        if pasos:
            joined_steps = " ".join(f"{index + 1}. {step}." for index, step in enumerate(pasos[:3]))
            parts.append(f"Conservative next steps: {joined_steps}")
        parts.append("If the symptoms do not match your appliance exactly, avoid forcing a repair and contact the brand's official technical support.")
        return append_follow_up_question(" ".join(parts), language)

    parts = [
        f"No he podido validar del todo la salida del extractor, pero si he recuperado un caso parecido relacionado con {aparato}.",
        f"La causa mas cercana encontrada en la base es: {causa}.",
    ]
    if pasos:
        joined_steps = " ".join(f"{index + 1}. {step}." for index, step in enumerate(pasos[:3]))
        parts.append(f"Como orientacion prudente, los pasos sugeridos serian: {joined_steps}")
    parts.append("Si los sintomas no encajan bien con tu equipo, evita forzar la reparacion y contacta con el soporte tecnico oficial de la marca.")
    return append_follow_up_question(" ".join(parts), language)


def build_grounded_assistant_response(
    user_message: str,
    extracted: dict[str, Any],
    rag_hits: list[dict[str, Any]],
) -> str:
    language = detect_output_language(user_message)
    appliance = str(extracted.get("aparato", "el electrodomestico"))
    symptom = str(extracted.get("sintoma", "el problema descrito"))
    cause = str(extracted.get("causa_probable", "No especificado"))
    certainty = extracted.get("porcentaje_certeza", "No especificado")
    error_code = str(extracted.get("codigo_error", "No especificado"))
    severity = extracted.get("grado_peligrosidad", "No especificado")
    steps = normalize_steps(extracted.get("pasos_reparacion"))
    rag_hint = rag_hits[0] if rag_hits else {}

    if language == "English":
        parts = [f"I detected a problem in {appliance}: {symptom}."]
        if error_code != "No especificado":
            parts.append(f"The extracted error code is {error_code}.")
        if cause != "No especificado":
            parts.append(f"The most likely cause right now is: {cause}.")
        if certainty != "No especificado":
            parts.append(f"Estimated confidence: {certainty}%.")
        if rag_hits:
            rag_appliance = clean_field(rag_hint.get("aparato")) or appliance
            parts.append(f"I also found similar registered cases for {rag_appliance}, which reinforces this diagnosis.")
        if steps:
            joined_steps = " ".join(f"{index + 1}. {step}." for index, step in enumerate(steps[:3]))
            parts.append(f"Prudent next steps: {joined_steps}")
        else:
            parts.append("I do not have enough safe repair steps to recommend a reliable intervention.")
        if severity == 1:
            parts.append("This case looks critical, so stop manipulating the appliance and contact the brand's official technical support immediately.")
        else:
            parts.append("If the symptoms do not match what you observe or the appliance keeps failing, contact the brand's official technical support.")
        return append_follow_up_question(" ".join(parts), language)

    parts = [f"He detectado un problema en {appliance}: {symptom}."]
    if error_code != "No especificado":
        parts.append(f"El codigo extraido es {error_code}.")
    if cause != "No especificado":
        parts.append(f"La causa probable mas razonable ahora mismo es: {cause}.")
    if certainty != "No especificado":
        parts.append(f"La certeza estimada es del {certainty}%.")
    if rag_hits:
        rag_appliance = clean_field(rag_hint.get("aparato")) or appliance
        parts.append(f"Ademas, he encontrado averias parecidas registradas para {rag_appliance}, lo que refuerza esta orientacion.")
    if steps:
        joined_steps = " ".join(f"{index + 1}. {step}." for index, step in enumerate(steps[:3]))
        parts.append(f"Como siguientes pasos prudentes, te sugiero: {joined_steps}")
    else:
        parts.append("No tengo pasos de reparacion suficientemente fiables como para recomendar una intervencion con seguridad.")
    if severity == 1:
        parts.append("Este caso parece critico, asi que deja de manipular el aparato y contacta cuanto antes con el soporte tecnico oficial de la marca.")
    else:
        parts.append("Si los sintomas no encajan bien con lo que ves en tu equipo o el fallo persiste, lo mas recomendable es contactar con el soporte tecnico oficial de la marca.")
    return append_follow_up_question(" ".join(parts), language)


def render_assistant_response_html(
    user_message: str,
    extracted: dict[str, Any],
    rag_hits: list[dict[str, Any]],
    issues: list[str],
) -> str:
    language = detect_output_language(user_message)
    appliance = html.escape(str(extracted.get("aparato", "el electrodomestico")))
    symptom = html.escape(str(extracted.get("sintoma", "el problema descrito")))
    cause = html.escape(str(extracted.get("causa_probable", "No especificado")))
    certainty = extracted.get("porcentaje_certeza", "No especificado")
    error_code = html.escape(str(extracted.get("codigo_error", "No especificado")))
    severity = extracted.get("grado_peligrosidad", "No especificado")
    steps = normalize_steps(extracted.get("pasos_reparacion"))

    summary_lines: list[str] = [f"He detectado un problema en <strong>{appliance}</strong>: {symptom}."]
    if error_code != "No especificado":
        summary_lines.append(f"El codigo extraido es <strong>{error_code}</strong>.")
    if certainty != "No especificado":
        summary_lines.append(f"La certeza estimada es del <strong>{certainty}%</strong>.")
    if rag_hits:
        summary_lines.append("Ademas, he encontrado averias parecidas en la base vectorial, lo que ayuda a reforzar esta orientacion.")
    if issues:
        summary_lines.append("Aun asi, hay incoherencias en la extraccion y conviene interpretar el diagnostico con prudencia.")

    if steps:
        steps_html = "<ul style='margin:8px 0 0 18px;padding:0;'>" + "".join(
            f"<li style='margin-bottom:6px;'>{html.escape(step)}</li>" for step in steps[:4]
        ) + "</ul>"
    else:
        if language == "English":
            steps_html = "<p style='margin:0;'>I do not have enough safe repair steps to recommend a reliable intervention.</p>"
        else:
            steps_html = "<p style='margin:0;'>No tengo pasos de reparacion suficientemente fiables como para recomendar una intervencion con seguridad.</p>"

    recommendation = (
        "Este caso parece critico, asi que deja de manipular el aparato y contacta cuanto antes con el soporte tecnico oficial de la marca."
        if severity == 1
        else "Si los sintomas no encajan bien con lo que ves en tu equipo o el fallo persiste, lo mas recomendable es contactar con el soporte tecnico oficial de la marca."
    )

    closing = "Do you need help with anything else?" if language == "English" else "¿Necesitas que te ayude con algo mas?"

    return (
        "<div style='display:flex;flex-direction:column;gap:10px;'>"
        f"{build_response_section('Resumen del caso', ' '.join(summary_lines))}"
        f"{build_response_section('Causa probable', cause)}"
        f"{build_response_section('Pasos sugeridos', steps_html)}"
        f"{build_response_section('Recomendacion', html.escape(recommendation))}"
        f"{build_response_section('Siguiente paso', html.escape(closing))}"
        "</div>"
    )


def call_ollama_generate(model: str, prompt: str, num_predict: int = 350) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": num_predict,
        },
    }
    response = requests.post(
        OLLAMA_GENERATE_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return str(data.get("response", ""))


def build_assistant_prompt(
    user_message: str,
    extracted: dict[str, Any],
    issues: list[str],
    contexto_rag: str = "",
) -> str:
    extracted_json = json.dumps(extracted, ensure_ascii=False, indent=2)
    target_language = detect_output_language(user_message)
    issues_block = "\n".join(f"- {issue}" for issue in issues) or "- No se han detectado incoherencias."
    rag_block = ""
    if contexto_rag:
        rag_block = f"\n[CONTEXTO RAG]\n{contexto_rag}\n"
    return f"""Eres un asistente tecnico multilingue especializado en electrodomesticos.
Responde solo en {target_language}. No cambies de idioma.
Escribe una respuesta de longitud corta-media, algo mas desarrollada que una sola frase, pero sin hacerse pesada visualmente.
Se practico, claro y cuidadoso con la seguridad.
Convierte el diagnostico extraido en una respuesta natural pensada para el usuario final.
No inventes hechos que no esten respaldados por los datos extraidos ni por el contexto RAG.
Si la extraccion contiene detalles sospechosos o incoherentes, no los presentes como hechos confirmados.
Si la peligrosidad extraida es 1, advierte de forma explicita que el usuario debe dejar de manipular el aparato y contactar con el soporte tecnico oficial si hace falta.
No menciones etiquetas internas, nombres de bloques, numeraciones de casos recuperados ni detalles de implementacion.
No digas cosas como "caso recuperado", "contexto RAG", "transcripcion" o frases parecidas.
Si no tienes informacion suficiente para recomendar una accion fiable, dilo con claridad y sugiere contactar con el soporte tecnico oficial.
{rag_block}
[USER_MESSAGE]
{user_message}

[EXTRACTED_JSON]
{extracted_json}

[CONSISTENCY_CHECK]
{issues_block}

[TASK]
Escribe una respuesta clara para el usuario final en texto plano.
Debe incluir, cuando sea posible:
1. un resumen breve del problema detectado,
2. la causa probable o la incertidumbre si no esta clara,
3. uno o varios pasos siguientes prudentes,
4. una pregunta final pidiendo si necesita algo mas.
Si faltan campos importantes o no son fiables, dilo de forma clara en vez de inventar detalles.
No devuelvas JSON."""


@app.post("/api/chat")
def chat() -> tuple[Any, int] | Any:
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    mode = str(payload.get("mode", DEFAULT_MODE)).strip().lower() or DEFAULT_MODE
    use_rag = bool(payload.get("use_rag", True))
    if mode not in {"extractor", "assistant"}:
        mode = DEFAULT_MODE

    if not message:
        return jsonify({"error": "Falta el campo 'message'"}), 400

    if not is_in_domain(message):
        return jsonify(
            {
                "response": (
                    "<strong>Fuera de dominio.</strong><br>"
                    "Este prototipo solo esta preparado ahora mismo para "
                    "consultas sobre averias y soporte tecnico de electrodomesticos. "
                    "Si necesitas algo relacionado con este dominio, dime el aparato y el sintoma principal."
                ),
                "mode": mode,
            }
        )

    aparato_hint = detect_device_category(message)
    rag_hits = buscar_fragmentos_rag(message, aparato_hint=aparato_hint, top_k=RAG_TOP_K) if use_rag else []
    contexto_rag = construir_contexto_rag(rag_hits)

    try:
        raw_response = call_ollama_generate(
            EXTRACTOR_MODEL,
            build_prompt(message, contexto_rag),
            num_predict=500,
        )
    except requests.RequestException as exc:
        return jsonify({"error": f"Error al consultar Ollama: {exc}"}), 502

    json_text = extract_first_json(raw_response)
    extracted = validate_extraction_json(json_text)

    if extracted is None:
        if mode == "assistant":
            assistant_text = build_rag_supported_fallback(message, rag_hits)
            response_html = (
                f"<div>{html.escape(assistant_text).replace(chr(10), '<br>')}</div>"
            )
            return jsonify({"response": response_html, "mode": mode, "rag_used": bool(rag_hits), "rag_hits": rag_hits, "use_rag": use_rag})

        fallback = (
            "<strong>El extractor no devolvio un JSON valido.</strong><br>"
            "Salida bruta del modelo para diagnostico:<br><br>"
            f"<pre>{html.escape(raw_response)}</pre>"
        )
        return jsonify({"response": fallback, "mode": mode, "rag_used": bool(rag_hits), "rag_hits": rag_hits, "use_rag": use_rag})

    issues = assess_extraction_consistency(message, extracted)

    if mode == "extractor":
        return jsonify(
            {
                "response": render_json_response(message, extracted, issues, rag_hits),
                "mode": mode,
                "rag_used": bool(rag_hits),
                "rag_hits": rag_hits,
                "extractor_json": extracted,
                "use_rag": use_rag,
            }
        )

    if issues:
        assistant_html = (
            f"<div>{html.escape(build_inconsistent_extraction_response(message, extracted, issues)).replace(chr(10), '<br>')}</div>"
        )
    else:
        assistant_html = render_assistant_response_html(message, extracted, rag_hits, issues)

    response_html = (
        f"{assistant_html}"
        f"{build_consistency_notice(issues)}"
        f"{build_severity_warning(message, extracted)}"
    )
    return jsonify(
        {
            "response": response_html,
            "mode": mode,
            "extractor_json": extracted,
            "rag_used": bool(rag_hits),
            "rag_hits": rag_hits,
            "use_rag": use_rag,
        }
    )


@app.get("/health")
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "extractor_model": EXTRACTOR_MODEL,
            "response_model": RESPONSE_MODEL,
            "default_mode": DEFAULT_MODE,
            "milvus_ok": _milvus_ok,
            "milvus_error": _milvus_error,
            "milvus_db": str(MILVUS_DB_PATH),
            "collection": MILVUS_COLLECTION,
            "rag_top_k": RAG_TOP_K,
        }
    )


inicializar_milvus()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
