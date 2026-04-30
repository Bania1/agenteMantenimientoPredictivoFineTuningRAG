import sys

# Forzar que Python entienda que milvus_lite es el paquete que acabas de instalar
try:
    import milvus_lite
except ImportError:
    # Si por alguna razón no lo encuentra con guion bajo, 
    # esto ayuda a mapear el nombre correctamente en memoria
    import milvus_lite as milvus_lite 

import fitz  # PyMuPDF
from pymilvus import MilvusClient

# Tu código sigue igual...
client = MilvusClient("./demo.db")
