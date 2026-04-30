try:
    import milvus_lite
except ImportError:
    # Si por alguna razón no lo encuentra con guion bajo, 
    # esto ayuda a mapear el nombre correctamente en memoria
    import milvus_lite as milvus_lite 

import fitz  # PyMuPDF
from pymilvus import MilvusClient

client = MilvusClient("./demo.db")

# 1. Extraer texto del PDF automáticamente
doc = fitz.open("microondasboschserie6.pdf")
texto_completo = ""
for pagina in doc:
    texto_completo += pagina.get_text()

# 2. Tu lógica para detectar información específica
# (Aquí podrías filtrar solo lo que te interesa)

# 3. Insertar en Milvus (asumiendo que ya tienes el vector)
# Milvus recibe diccionarios de Python, que son como JSONs,
# pero se generan automáticamente en memoria.
data = [
    {"vector": [0.1, 0.2, ...], "text": texto_completo[:500], "source": "pdf_1"}
]
client.insert(collection_name="demo_collection", data=data)
