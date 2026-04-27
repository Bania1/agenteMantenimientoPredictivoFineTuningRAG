# Agente de Mantenimiento Predictivo para el hogar y soporte técnico

Proyecto de la Cátedra Atmira UCO 2025/2026. En este repositorio desarrollamos un agente de mantenimiento predictivo para el hogar con soporte técnico, integrando capacidades de MLops y herramientas de Databricks.

## Descripción del proyecto

Este proyecto construye un agente conversacional inteligente orientado a:
- Diagnóstico y soporte técnico para dispositivos del hogar.
- Mantenimiento predictivo basado en datos de sensores, registros y métricas.
- Respuestas contextualizadas mediante técnicas de RAG (Retrieval-Augmented Generation).

El agente combina finetuning de modelos de lenguaje con un flujo de trabajo de monitoreo, experimentación y despliegue en un entorno Databricks.

## Tecnologías principales

- Databricks: Workspaces, Catalog, experimentación y despliegue.
- MLflow: Seguimiento de experimentos, métricas, artefactos y versiones de modelos.
- Python: Desarrollo de pipelines, procesamiento de datos y entrenamiento.
- Librerías de ciencia de datos y MLops: pandas, numpy, scikit-learn, transformers, accelerate, peft, datasets, y otras.
- RAG: Uso de colecciones de documentos y embeddings para mejorar las respuestas del agente.

## Arquitectura general

1. Ingesta y preparación de datos
   - Captura de datos de equipos del hogar, sensores y solicitudes de soporte.
   - Normalización de datos en Databricks y uso de Catálogo para gobernanza.

2. Finetuning del modelo Qwen 2.5 0.5B con LoRA
   - Empleamos una versión ligera de Qwen 2.5 con 0.5B parámetros.
   - Aplicamos LoRA para adaptar el modelo a nuestro dominio sin entrenar todos los pesos.
   - Se diseñaron `system prompts` para orientar el comportamiento del agente hacia mantenimiento y soporte técnico.

3. Entrenamiento y evaluación
   - Los experimentos se ejecutaron en Databricks utilizando notebooks y clusters.
   - Se registraron métricas de entrenamiento y evaluación con MLflow.
   - Evaluamos el desempeño del modelo con métricas de calidad, coherencia y precisión en respuestas técnicas.

4. Generación de respuestas y RAG
   - Integramos un repositorio documental para consultas de soporte técnico.
   - El agente utiliza recuperación de información para enriquecer sus respuestas con datos específicos del dominio.

5. Documentación y control de versiones
   - En este repositorio se aloja la documentación del proyecto.
   - Se documenta el flujo de trabajo, la configuración de Databricks, los experimentos de MLflow y las decisiones de diseño.

## Detalles de finetuning

- Modelo base: Qwen 2.5 0.5B.
- Estrategia: LoRA para un entrenamiento eficiente y ligero.
- Objetivo: adaptar el lenguaje del modelo a escenarios de mantenimiento predictivo y soporte técnico residencial.
- Componentes clave:
  - Prompts de sistema para establecer rol, tono y reglas de respuesta.
  - Conjuntos de datos de entrenamiento/validación construidos con casos reales y simulados.
  - Uso de MLflow para comparar versiones y métricas.

## Uso de Databricks

- Workspaces: desarrollo colaborativo de notebooks y pipelines.
- Catalog: gestión de tablas, datasets y metadatos.
- Experimentos: ejecución de jobs y entrenamiento escalable.
- Integración con MLflow para seguimiento centralizado.

## MLflow y métricas

Registramos las siguientes métricas y artefactos:
- Pérdida de entrenamiento y validación.
- Exactitud y métricas de evaluación de respuesta.
- Versiones del modelo y archivos de checkpoints.
- Configuraciones de entrenamiento, hiperparámetros y metadatos.

## Estructura del repositorio

- `README.md`: documentación general del proyecto.
- `notebooks/`: ejemplos de notebooks de Databricks y experimentos.
- `src/`: código fuente para entrenamiento, ingestión y evaluación.
- `docs/`: documentación adicional, guías y diagramas.

## Objetivos futuros

- Ampliar el agente con más datos de sensores y eventos domésticos.
- Integrar pipeline completo de MLOps en Databricks con despliegue automático.
- Mejorar la base de conocimiento RAG y los prompts para soporte técnico.
- Evaluar el modelo en escenarios reales de mantenimiento predictivo.

## Autores

- Ángel Eduardo Bauste De Nicolais
- Álvaro Ruiz Rivas
