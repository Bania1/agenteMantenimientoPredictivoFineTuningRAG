# Endpoints HTTP y puertos

## Puntos de entrada publicos
- `http://31.220.95.11/` -> frontend servido por `nginx` en Docker sobre el puerto `80`
- `ssh angel@31.220.95.11` -> acceso SSH sobre el puerto `22`

## Endpoints del backend actual
- `POST /api/chat` -> endpoint principal del backend Flask accesible a traves del proxy de `nginx`
- `GET /health` -> comprobacion de estado del backend en `http://127.0.0.1:5000/health`

## Endpoints RAG encontrados en el codigo
- `POST /chat` en `apps/rag/api/rag_server.py`
- `GET /health` en `apps/rag/api/rag_server.py`
- `GET /buscar` en `apps/rag/api/rag_server.py`

## Exposicion actual de puertos
- `80/tcp` publico por Docker + `nginx`
- `22/tcp` publico por SSH
- `5000/tcp` expuesto directamente por el backend Flask
- `11434/tcp` solo en local para `Ollama`

## Recomendacion
Cuando la migracion al repositorio quede validada, conviene restringir el puerto `5000` a `127.0.0.1` para que el acceso publico pase unicamente por `nginx`.
