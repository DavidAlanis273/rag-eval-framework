# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install sentence-transformers --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 - Prueba de Retrieval
# MAGIC Este notebook prueba la función de búsqueda con preguntas manuales.

# COMMAND ----------

import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in dir() else '/Workspace' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()).rsplit('/notebooks', 1)[0]

sys.path.insert(0, repo_root)

from sentence_transformers import SentenceTransformer
from utils.retrieval import load_embeddings, retrieve_top_k
from config.settings import EMBEDDINGS_FILE

# Cargar modelo y embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_path = os.path.join(repo_root, EMBEDDINGS_FILE)
data = load_embeddings(embeddings_path)

print(f"Modelo cargado")
print(f"Embeddings cargados: {len(data['chunk_id'])} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prueba 1: Pregunta específica sobre un producto

# COMMAND ----------

query = "What data recorder has a 3.5 inch display?"
results = retrieve_top_k(query, model, data, k=5)

print(f"Query: {query}\n")
for i, r in enumerate(results):
    print(f"  #{i+1} | Score: {r['similarity_score']:.4f} | Source: {r['source_document']}")
    print(f"       {r['chunk_text'][:150]}...")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prueba 2: Pregunta sobre protocolos de comunicación

# COMMAND ----------

query = "Which recorder supports EtherNet/IP communication?"
results = retrieve_top_k(query, model, data, k=5)

print(f"Query: {query}\n")
for i, r in enumerate(results):
    print(f"  #{i+1} | Score: {r['similarity_score']:.4f} | Source: {r['source_document']}")
    print(f"       {r['chunk_text'][:150]}...")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prueba 3: Pregunta sobre cumplimiento regulatorio

# COMMAND ----------

query = "Which product complies with FDA 21 CFR Part 11?"
results = retrieve_top_k(query, model, data, k=5)

print(f"Query: {query}\n")
for i, r in enumerate(results):
    print(f"  #{i+1} | Score: {r['similarity_score']:.4f} | Source: {r['source_document']}")
    print(f"       {r['chunk_text'][:150]}...")
    print()