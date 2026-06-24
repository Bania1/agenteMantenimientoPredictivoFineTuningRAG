# Estructura del repositorio

- `apps/frontend/web` -> assets del frontend activo
- `apps/backend/current` -> backend principal que debe quedar como fuente de verdad
- `apps/backend/candidates` -> variantes o experimentos que aun no se han promocionado
- `apps/rag/api` -> implementacion alternativa del servidor RAG encontrada en el servidor
- `apps/rag/indexers` -> scripts para construir o actualizar la base vectorial
- `config/domain` -> reglas compartidas del dominio de electrodomesticos
- `data/milvus-lite` -> instantanea local de la base vectorial usada para RAG
- `data/rag-samples` -> datasets y artefactos auxiliares del trabajo previo
- `infra/nginx` -> configuracion de `nginx`
- `infra/docker/frontend` -> `docker-compose` del frontend
- `infra/systemd` -> unidades `systemd`
- `archive/home-angel` -> copia de rutas y ficheros originales del servidor
- `docs` -> auditorias, decisiones de arquitectura y guias de migracion
