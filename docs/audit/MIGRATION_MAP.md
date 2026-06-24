# Mapa de migracion

## Rutas activas actuales -> rutas canonicas del repositorio
- `/opt/containerd/web/index.html` -> `apps/frontend/web/index.html`
- `/opt/containerd/nginx/conf.d/default.conf` -> `infra/nginx/conf.d/default.conf`
- `/opt/containerd/docker-compose.yml` -> `infra/docker/frontend/docker-compose.yml`
- `/home/angel/mvp_backend/app.py` -> `apps/backend/current/app.py`
- `/home/angel/mvp_backend/requirements.txt` -> `apps/backend/current/requirements.txt`
- `/home/angel/mvp_backend/app1.0.py` -> `apps/backend/candidates/app_rag_candidate.py`
- `/home/angel/mvp_backend/domain_catalog.json` -> `config/domain/domain_catalog.json`
- `/home/angel/mvp_backend/milvus.db` -> `data/milvus-lite/reparaciones-lite/milvus.db`
- `/home/alvaro/rag_server.py` -> `apps/rag/api/rag_server.py`
- `/home/alvaro/CATEDRA/nuevo/json_to_milvus.py` -> `apps/rag/indexers/json_to_milvus.py`
- `/etc/systemd/system/mvp-backend.service` -> `infra/systemd/mvp-backend.service`

## Estrategia de promocion
1. Mantener produccion apuntando a las rutas actuales mientras validamos el codigo dentro del repositorio.
2. Unificar la logica del backend activo y del candidato con RAG en una sola implementacion principal.
3. Instalar en el entorno correcto las dependencias minimas necesarias para Milvus Lite.
4. Cambiar la unidad `systemd` para que ejecute la ruta canonica del repositorio.
5. Cerrar la exposicion publica del puerto `5000` y dejar el trafico externo unicamente detras de `nginx`.
