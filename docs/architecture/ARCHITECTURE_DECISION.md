# Architecture Decision

## Chosen Architecture
A modular layered architecture with one repository and clearly separated concerns:
- `apps/frontend`: web UI
- `apps/backend/current`: current HTTP API used by the frontend
- `apps/backend/candidates`: alternative backend implementations not yet promoted
- `apps/rag/api`: RAG service code
- `apps/rag/indexers`: scripts to build/update the vector store
- `config`: shared configuration such as domain catalogs
- `data`: local vector stores and curated sample datasets
- `infra`: deployment artifacts such as nginx, docker compose, and systemd units
- `docs`: audits, architecture notes, and migration maps

## Why This Is The Best Fit Here
- It separates live code from experiments without losing either.
- It keeps infrastructure and application code versioned together.
- It makes future changes to RAG, models, or deployment boundaries safer.
- It avoids mixing operational files into user home directories.
- It gives one canonical root for onboarding, debugging, and deployment.

## Benefits
- Easier version control and rollback.
- Cleaner migration path from current Flask API to an eventual RAG-backed API.
- Safer promotion flow: candidate code can be reviewed before replacing live code.
- Better maintainability because paths are predictable and grouped by responsibility.
- Easier CI/CD later because app, config, and infra live together.
