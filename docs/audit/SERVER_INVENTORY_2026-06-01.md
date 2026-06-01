# Server Inventory 2026-06-01

## Host
- Hostname: `vmi2901112`
- OS: `Ubuntu 20.04.6 LTS`
- User reviewed: `angel`

## Live Services
- `nginx` in Docker publishing `:80`
- `mvp-backend.service` publishing `:5000`
- `ollama.service` listening on `127.0.0.1:11434`

## Active Runtime Paths
- Frontend HTML: `/opt/containerd/web/index.html`
- Nginx config: `/opt/containerd/nginx/conf.d/default.conf`
- Docker compose: `/opt/containerd/docker-compose.yml`
- Backend service unit: `/etc/systemd/system/mvp-backend.service`
- Active backend code: `/home/angel/mvp_backend/app.py`
- Active backend catalog: `/home/angel/mvp_backend/domain_catalog.json`
- Active local vector store artifact: `/home/angel/mvp_backend/milvus.db`
- Ollama models folder: `/home/angel/models`

## RAG Work Found But Not Deployed
- Candidate Flask+RAG backend: `/home/angel/mvp_backend/app1.0.py`
- Standalone FastAPI RAG server: `/home/alvaro/rag_server.py`
- Indexing script: `/home/alvaro/CATEDRA/nuevo/json_to_milvus.py`
- Additional sample/vector data: `/home/alvaro/CATEDRA/nuevo/*`

## Security Notes
- Backend currently binds to `0.0.0.0:5000`
- This bypasses the reverse proxy boundary and should be narrowed later to `127.0.0.1:5000`
