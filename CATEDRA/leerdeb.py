from pymilvus import MilvusClient
import json

client = MilvusClient("averias_microondas.db")

results = client.query(
    collection_name = "averias",
    filter          = "",
    output_fields   = [
        "id",
        "aparato",
        "sintoma",
        "codigo_error",
        "causa_probable",
        "pasos_reparacion",
        "porcentaje_certeza",
        "grado_peligrosidad",
        "fuente_pdf",
        "pagina"
    ],
    limit = 100
)

for r in results:
    print(json.dumps(r, ensure_ascii=False, indent=2))

from pymilvus import MilvusClient

client = MilvusClient("averias_microondas.db")
schema = client.describe_collection("averias")
print(schema)

