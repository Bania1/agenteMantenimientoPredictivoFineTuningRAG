# Inventario del servidor 2026-06-01

## Host
- Hostname: `vmi2901112`
- Sistema operativo: `Ubuntu 20.04.6 LTS`
- Usuario revisado: `angel`

## Servicios activos
- `nginx` en Docker publicando `:80`
- `mvp-backend.service` publicando `:5000`
- `ollama.service` escuchando en `127.0.0.1:11434`

## Rutas activas verificadas
- Frontend HTML: `/opt/containerd/web/index.html`
- Configuracion de `nginx`: `/opt/containerd/nginx/conf.d/default.conf`
- `docker-compose` del frontend: `/opt/containerd/docker-compose.yml`
- Unidad del backend: `/etc/systemd/system/mvp-backend.service`
- Codigo activo del backend: `/home/angel/mvp_backend/app.py`
- Catalogo de dominio activo: `/home/angel/mvp_backend/domain_catalog.json`
- Artefacto vectorial local activo: `/home/angel/mvp_backend/milvus.db`
- Carpeta de modelos de `Ollama`: `/home/angel/models`

## Trabajo RAG localizado pero no desplegado
- Backend candidato Flask + RAG: `/home/angel/mvp_backend/app1.0.py`
- Servidor RAG FastAPI independiente: `/home/alvaro/rag_server.py`
- Script de indexacion: `/home/alvaro/CATEDRA/nuevo/json_to_milvus.py`
- Artefactos y muestras adicionales: `/home/alvaro/CATEDRA/nuevo/*`

## Nota de seguridad
- El backend actual escucha en `0.0.0.0:5000`
- Esto permite saltarse el proxy inverso, asi que conviene migrarlo despues a `127.0.0.1:5000`
