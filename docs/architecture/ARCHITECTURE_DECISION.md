# Decision de arquitectura

## Arquitectura elegida
Se adopta una arquitectura modular por capas dentro de un unico repositorio, separando claramente responsabilidades:
- `apps/frontend`: interfaz web
- `apps/backend/current`: API HTTP principal usada por el frontend
- `apps/backend/candidates`: implementaciones alternativas todavia no promocionadas
- `apps/rag/api`: codigo auxiliar del servidor RAG encontrado en el servidor
- `apps/rag/indexers`: scripts para poblar o actualizar la base vectorial
- `config`: configuracion compartida, como el catalogo de dominio
- `data`: bases vectoriales locales y datasets auxiliares
- `infra`: artefactos de despliegue (`nginx`, Docker, `systemd`)
- `docs`: auditorias, notas de arquitectura y guias de migracion

## Por que esta opcion es la mas adecuada aqui
- Separa el codigo activo de los experimentos sin perder ninguna variante util.
- Permite versionar juntos codigo, configuracion e infraestructura.
- Facilita evolucionar el backend clasico hacia una variante con RAG sin romper produccion.
- Evita que el conocimiento del proyecto quede repartido por rutas sueltas del `home` de distintos usuarios.
- Deja un unico punto de entrada claro para depuracion, mantenimiento y futuras migraciones.

## Ventajas
- Mejor control de versiones y rollback mas sencillo.
- Menos riesgo al promocionar cambios grandes, como la integracion de Milvus.
- Onboarding mas facil para cualquier persona que entre al proyecto.
- Base mucho mas limpia para CI/CD o automatizaciones futuras.
- Capacidad de crecer sin mezclar frontend, backend, datos e infraestructura.
