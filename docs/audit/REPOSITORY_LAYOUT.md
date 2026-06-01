# Repository Layout

- `apps/frontend/web` -> active frontend assets
- `apps/backend/current` -> backend currently serving production traffic
- `apps/backend/candidates` -> backend variants not yet promoted
- `apps/rag/api` -> standalone RAG API code found on server
- `apps/rag/indexers` -> scripts to create/update vector data
- `config/domain` -> shared domain classification rules
- `data/milvus-lite` -> local vector store snapshot used by backend candidate
- `data/rag-samples` -> sample datasets and auxiliary vector stores from coworker work
- `infra/nginx` -> nginx runtime config
- `infra/docker/frontend` -> compose file for frontend container
- `infra/systemd` -> systemd unit definitions
- `archive/home-angel` -> preserved copies of live files and original home artifacts
- `docs` -> server audit, path maps, architecture decisions
