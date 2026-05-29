from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, request

OLLAMA_GENERATE_URL = os.getenv("OLLAMA_GENERATE_URL", "http://127.0.0.1:11434/api/generate")
EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "qwen-fusionado")
RESPONSE_MODEL = os.getenv("RESPONSE_MODEL", "llama3.2:1b")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "600"))
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "assistant")
DOMAIN_CATALOG_PATH = Path(__file__).with_name("domain_catalog.json")

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


def build_prompt(user_message: str) -> str:
    input_text = f"[INPUT TEXT]\n{user_message.strip()}\n\n[OUTPUT]\n"
    return f"{SYSTEM_PROMPT}\n\n{input_text}"


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


def fill_missing_fields(data: dict[str, Any]) -> dict[str, Any]:
    filled = dict(data)
    filled.setdefault("aparato", "No especificado")
    filled.setdefault("sintoma", "No especificado")
    filled.setdefault("codigo_error", "No especificado")
    filled.setdefault("causa_probable", "No especificado")
    filled.setdefault("pasos_reparacion", "No especificado")
    filled["porcentaje_certeza"] = normalize_percentage(filled.get("porcentaje_certeza"))
    filled["grado_peligrosidad"] = normalize_severity(filled.get("grado_peligrosidad"))
    return filled


def validate_extraction_json(json_text: str | None) -> dict[str, Any] | None:
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
                f"Incompatibilidad aparente: el término '{term}' no encaja bien con el aparato '{device_category}'."
            )
    return issues


def build_consistency_notice(issues: list[str]) -> str:
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


def render_json_response(user_message: str, data: dict[str, Any], issues: list[str]) -> str:
    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    return (
        "<strong>Modo validación del extractor</strong><br>"
        "Respuesta JSON generada por el modelo:<br><br>"
        f"<pre>{html.escape(pretty)}</pre>"
        f"{build_consistency_notice(issues)}"
        f"{build_severity_warning(user_message, data)}"
    )


def build_inconsistent_extraction_response(user_message: str, extracted: dict[str, Any], issues: list[str]) -> str:
    language = detect_output_language(user_message)
    appliance = str(extracted.get("aparato", "el electrodoméstico"))
    symptom = str(extracted.get("sintoma", "el problema descrito"))
    error_code = str(extracted.get("codigo_error", "No especificado"))
    code_text = ""
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


def normalize_steps(steps: Any) -> list[str]:
    if isinstance(steps, list):
        return [str(step).strip() for step in steps if str(step).strip()]
    if isinstance(steps, str) and steps.strip():
        return [steps.strip()]
    return []


def build_structured_assistant_response(user_message: str, extracted: dict[str, Any]) -> str:
    language = detect_output_language(user_message)
    appliance = str(extracted.get("aparato", "el electrodoméstico"))
    symptom = str(extracted.get("sintoma", "el problema descrito"))
    cause = str(extracted.get("causa_probable", "No especificado"))
    certainty = extracted.get("porcentaje_certeza", "No especificado")
    error_code = str(extracted.get("codigo_error", "No especificado"))
    steps = normalize_steps(extracted.get("pasos_reparacion"))

    if language == "English":
        parts = [
            f"We detected a problem affecting the {appliance}: {symptom}.",
        ]
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
        return " ".join(parts)

    parts = [
        f"Hemos detectado un problema en {appliance}: {symptom}.",
    ]
    if error_code != "No especificado":
        parts.append(f"El código extraído es {error_code}.")
    if cause != "No especificado":
        parts.append(f"La causa probable extraída es: {cause}.")
    if certainty != "No especificado":
        parts.append(f"Certeza estimada: {certainty}%.")
    if steps:
        joined_steps = " ".join(f"{index + 1}. {step}." for index, step in enumerate(steps[:3]))
        parts.append(f"Pasos sugeridos: {joined_steps}")
    else:
        parts.append("No se han extraído pasos de reparación seguros con suficiente confianza.")
    return " ".join(parts)


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


def build_assistant_prompt(user_message: str, extracted: dict[str, Any], issues: list[str]) -> str:
    extracted_json = json.dumps(extracted, ensure_ascii=False, indent=2)
    target_language = detect_output_language(user_message)
    issues_block = "\n".join(f"- {issue}" for issue in issues) or "- No issues detected."
    return f"""You are a multilingual appliance support assistant.
Answer only in {target_language}. Do not switch languages. Support at least Spanish and English.
Be concise, practical, and safety-aware. Your job is to turn a structured appliance diagnosis into a natural user-facing answer.
Do not invent facts not grounded in the extracted data.
If the extraction contains suspicious or inconsistent details, do not present them as confirmed facts. Say that some extracted details may be unreliable and focus on safe, conservative guidance.

If the extracted severity is 1, explicitly warn the user to stop manipulating the appliance and contact the official technical support of the brand if known.

[USER_MESSAGE]
{user_message}

[EXTRACTED_JSON]
{extracted_json}

[CONSISTENCY_CHECK]
{issues_block}

[TASK]
Write a clear support reply for the end user in plain text.
If key fields are missing or unreliable, say so plainly instead of inventing details.
If there are consistency issues, explain that the diagnostic extraction may be inconsistent for this appliance and avoid repeating the suspicious component as if it were certainly correct.
Do not output JSON."""


@app.post("/api/chat")
def chat() -> tuple[Any, int] | Any:
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    mode = str(payload.get("mode", DEFAULT_MODE)).strip().lower() or DEFAULT_MODE
    if mode not in {"extractor", "assistant"}:
        mode = DEFAULT_MODE

    if not message:
        return jsonify({"error": "Falta el campo 'message'"}), 400

    if not is_in_domain(message):
        return jsonify(
            {
                "response": (
                    "<strong>Fuera de dominio.</strong><br>"
                    "Este prototipo solo está preparado ahora mismo para "
                    "consultas sobre averías y soporte técnico de electrodomésticos."
                ),
                "mode": mode,
            }
        )

    try:
        raw_response = call_ollama_generate(EXTRACTOR_MODEL, build_prompt(message), num_predict=350)
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

    if mode == "extractor":
        return jsonify({"response": render_json_response(message, extracted, issues), "mode": mode})

    if issues:
        assistant_text = build_inconsistent_extraction_response(message, extracted, issues)
    elif extracted.get("grado_peligrosidad") == 1:
        assistant_text = build_structured_assistant_response(message, extracted)
    else:
        try:
            assistant_text = call_ollama_generate(
                RESPONSE_MODEL,
                build_assistant_prompt(message, extracted, issues),
                num_predict=260,
            )
        except requests.RequestException as exc:
            assistant_text = build_structured_assistant_response(message, extracted)

    response_html = (
        f"<div>{html.escape(assistant_text).replace(chr(10), '<br>')}</div>"
        f"{build_consistency_notice(issues)}"
        f"{build_severity_warning(message, extracted)}"
    )
    return jsonify(
        {
            "response": response_html,
            "mode": mode,
            "extractor_json": extracted,
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
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
