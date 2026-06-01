# HTTP Endpoints And Ports

## Public Entry Points
- `http://31.220.95.11/` -> frontend served by nginx in Docker on port `80`
- `ssh angel@31.220.95.11` -> SSH on port `22`

## Current Backend Endpoints
- `POST /api/chat` -> current Flask backend behind nginx proxy
- `GET /health` on backend service -> `http://127.0.0.1:5000/health`

## Candidate RAG Endpoints Found In Code
- `POST /chat` in `apps/rag/api/rag_server.py`
- `GET /health` in `apps/rag/api/rag_server.py`
- `GET /buscar` in `apps/rag/api/rag_server.py`

## Current Port Exposure
- `80/tcp` public via Docker nginx
- `22/tcp` public via SSH
- `5000/tcp` public directly from Flask backend
- `11434/tcp` local only for Ollama

## Recommendation
Restrict backend port `5000` to localhost once the repo-based deployment path is validated.
