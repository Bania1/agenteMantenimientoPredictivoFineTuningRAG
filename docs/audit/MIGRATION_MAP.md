# Migration Map

## Current Live Paths -> Repository Paths
- `/opt/containerd/web/index.html` -> `apps/frontend/web/index.html`
- `/opt/containerd/nginx/conf.d/default.conf` -> `infra/nginx/conf.d/default.conf`
- `/opt/containerd/docker-compose.yml` -> `infra/docker/frontend/docker-compose.yml`
- `/home/angel/mvp_backend/app.py` -> `apps/backend/current/app.py`
- `/home/angel/mvp_backend/requirements.txt` -> `apps/backend/current/requirements.txt`
- `/home/angel/mvp_backend/app1.0.py` -> `apps/backend/candidates/app_rag_candidate.py`
- `/home/angel/mvp_backend/domain_catalog.json` -> `config/domain/domain_catalog.json`
- `/home/angel/mvp_backend/milvus.db` -> `data/milvus-lite/reparaciones-lite`
- `/home/alvaro/rag_server.py` -> `apps/rag/api/rag_server.py`
- `/home/alvaro/CATEDRA/nuevo/json_to_milvus.py` -> `apps/rag/indexers/json_to_milvus.py`
- `/etc/systemd/system/mvp-backend.service` -> `infra/systemd/mvp-backend.service`

## Promotion Strategy
1. Keep production pointing to the current live paths until the repo structure is validated.
2. Review and unify the active backend and RAG candidate.
3. Install missing RAG dependencies into the chosen runtime.
4. Switch the systemd service to point to the canonical repo path.
5. Restrict backend exposure to localhost and keep public access only through nginx.
