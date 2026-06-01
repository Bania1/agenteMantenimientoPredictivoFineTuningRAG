#!/usr/bin/env python3
"""
PASO 2 — Servidor RAG con FastAPI + Milvus + embeddings reales.

Arranque:
    uvicorn 2_rag_server:app --host 0.0.0.0 --port 8000 --reload

Dependencias:
    pip install "pymilvus[model]" fastapi uvicorn httpx python-dotenv

Variables de entorno (.env):
    MODEL_API_URL   URL de tu modelo  (ej: http://localhost:11434/v1/chat/completions)
    MODEL_API_KEY   API key opcional
    MODEL_NAME      nombre del modelo (ej: llama3, gpt-4o)
    ALLOWED_ORIGINS CORS origins separados por coma (default: *)
"""

from __future__ import annotations
from typing import Any, Optional, Tuple, List

import os
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymilvus import MilvusClient
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from pydantic import BaseModel

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURACIÓN — debe coincidir con 1_indexar_pdfs.py
# ─────────────────────────────────────────────
# Cámbialo para que use una carpeta clara, así Milvus se gestiona solo
MILVUS_DB = "/home/alvaro/milvus_local.db"
COLLECTION_NAME = "reparaciones"
EMBED_MODEL     = "all-MiniLM-L6-v2"           # ← MISMO modelo que al indexar
TOP_K           = int(os.getenv("TOP_K", "5"))

# CORREGIDO: Se añaden comillas para que Python lo interprete correctamente como string
MODEL_API_URL   = "http://localhost:11434/v1/chat/completions"
MODEL_API_KEY   = os.getenv("MODEL_API_KEY", "")
MODEL_NAME      = "qwen-fusionado:latest"

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
# ─────────────────────────────────────────────
# ── Singletons (se cargan una vez al arrancar) ──────────────────────────────

_embedding_fn: Optional[SentenceTransformerEmbeddingFunction] = None
_milvus_client: Optional[MilvusClient] = None
def get_embedding_fn() -> SentenceTransformerEmbeddingFunction:
    global _embedding_fn
    if _embedding_fn is None:
        print(f"[INFO] Cargando modelo embeddings: {EMBED_MODEL}")
        _embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL, device="cpu")
    return _embedding_fn
def get_milvus_client() -> MilvusClient:
    global _milvus_client
    if _milvus_client is None:
        print(f"[INFO] Conectando a Milvus Lite local en: {MILVUS_DB}")
        # Para Milvus Lite local NO usamos ni client_properties ni connection_args
        _milvus_client = MilvusClient(uri=MILVUS_DB)
    return _milvus_client
# ── Búsqueda vectorial ───────────────────────────────────────────────────────

def buscar_en_milvus(
    query: str,
    top_k: int = TOP_K,
    subject: str = "",
) -> List[dict[str, Any]]:
    """
    Convierte la query en vector y busca en Milvus.
    Si se pasa `subject`, filtra por tipo de aparato.
    """
    fn = get_embedding_fn()
    client = get_milvus_client()

    # encode_queries para preguntas, encode_documents para indexar — ¡no mezclar!
    vector = fn.encode_queries([query])[0].tolist()

    filtro = f"subject == '{subject}'" if subject else ""

    resultados = client.search(
        collection_name=COLLECTION_NAME,
        data=[vector],
        filter=filtro,
        limit=top_k,
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
# ── Construcción del prompt ──────────────────────────────────────────────────

def construir_prompt(query: str, hits: List[dict]) -> Tuple[str, str]:
    if hits:
        bloques = []
        for i, h in enumerate(hits, 1):
            bloques.append(
                f"[Fragmento {i} — {h['source']}, pág. {h['page']}, similitud {h['score']}]\n"
                f"{h['text']}"
            )
        contexto = "\n\n".join(bloques)
    else:
        contexto = "No se encontraron fragmentos relevantes en la base de datos."

    system = (
        "Eres un técnico experto en reparación de electrodomésticos. "
        "Responde SIEMPRE en español, de forma clara y estructurada. "
        "Usa ÚNICAMENTE la información del contexto proporcionado. "
        "Si el contexto no es suficiente, indícalo explícitamente. "
        "Advierte sobre riesgos eléctricos o de seguridad cuando sea necesario."
    )

    user = (
        f"CONTEXTO EXTRAÍDO DEL MANUAL:\n{contexto}\n\n"
        f"PREGUNTA: {query}\n\n"
        f"Responde con:\n"
        f"1. Diagnóstico probable\n"
        f"2. Pasos de reparación o solución\n"
        f"3. Advertencias de seguridad si aplican"
    )

    return system, user
# ── Llamada al LLM ───────────────────────────────────────────────────────────

async def llamar_llm(system: str, user: str) -> str:
    if not MODEL_API_URL:
        return (
            "⚠ MODEL_API_URL no configurada.\n"
            "Define la variable de entorno en tu .env y reinicia el servidor.\n\n"
            "Ejemplo para Ollama:\n  MODEL_API_URL=http://localhost:11434/v1/chat/completionsn"
            "Ejemplo para OpenAI:\n  MODEL_API_URL=https://api.openai.com/v1/chat/completions"
        )

    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY:
        headers["Authorization"] = f"Bearer {MODEL_API_KEY}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    async with httpx.AsyncClient(timeout=60.0) as http:
        resp = await http.post(MODEL_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return str(data)
# ── FastAPI ──────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Reparaciones", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Precarga el modelo al arrancar (evita latencia en la primera petición)
@app.on_event("startup")
async def startup():
    get_embedding_fn()
    get_milvus_client()
    print("[OK] Servidor listo.")
class ChatRequest(BaseModel):
    query: str
    subject: Optional[str] = None    # filtro opcional: "microondas", "lavadora"…
    top_k: Optional[int] = None
    session_id: Optional[str] = None
class FragmentoFuente(BaseModel):
    score: float
    text: str
    source: str
    page: int
    subject: str
class ChatResponse(BaseModel):
    answer: str
    sources: List[FragmentoFuente]
    query: str
@app.get("/health")
async def health():
    """Estado del servidor y de Milvus."""
    try:
        client = get_milvus_client()
        colecciones = client.list_collections()
        return {"status": "ok", "milvus_db": MILVUS_DB, "colecciones": colecciones}
    except Exception as exc:
        raise HTTPException(503, f"Milvus no disponible: {exc}")
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Endpoint principal RAG: busca en Milvus → construye prompt → llama al LLM."""
    if not req.query.strip():
        raise HTTPException(400, "La query no puede estar vacía.")

    top_k = req.top_k or TOP_K

    # 1. Recuperar fragmentos relevantes de Milvus
    try:
        hits = buscar_en_milvus(req.query, top_k=top_k, subject=req.subject or "")
    except Exception as exc:
        raise HTTPException(503, f"Error consultando Milvus: {exc}")

    # 2. Construir prompt enriquecido con el contexto
    system, user = construir_prompt(req.query, hits)

    # 3. Llamar al LLM
    try:
        answer = await llamar_llm(system, user)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(502, f"Error del modelo: {exc.response.status_code}")
    except Exception as exc:
        raise HTTPException(502, f"Error llamando al modelo: {exc}")

    return ChatResponse(
        answer=answer,
        sources=[FragmentoFuente(**h) for h in hits],
        query=req.query,
    )
@app.get("/buscar")
async def buscar(q: str, subject: str = "", top_k: int = TOP_K):
    """Solo búsqueda vectorial, sin LLM. Útil para depurar el retrieval."""
    if not q:
        raise HTTPException(400, "Parámetro 'q' requerido.")
    hits = buscar_en_milvus(q, top_k=top_k, subject=subject)
    return {"query": q, "subject": subject, "resultados": hits}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("2_rag_server:app", host="0.0.0.0", port=8000, reload=True)