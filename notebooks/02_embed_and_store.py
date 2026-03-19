# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install sentence-transformers --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 - Generar Embeddings y Almacenar
# MAGIC Este notebook lee los chunks del CSV, genera embeddings con 
# MAGIC sentence-transformers y guarda el resultado como pickle.

# COMMAND ----------

import os
import sys
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in dir() else '/Workspace' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()).rsplit('/notebooks', 1)[0]

sys.path.insert(0, repo_root)

from config.settings import CHUNKS_FILE, EMBEDDINGS_FILE

print(f"Repo root: {repo_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1: Cargar los chunks

# COMMAND ----------

chunks_path = os.path.join(repo_root, CHUNKS_FILE)
df_chunks = pd.read_csv(chunks_path)

print(f"Chunks cargados: {len(df_chunks)}")
df_chunks.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2: Cargar el modelo de embeddings

# COMMAND ----------

model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Modelo cargado: all-MiniLM-L6-v2")
print(f"Dimensión del embedding: {model.get_sentence_embedding_dimension()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3: Generar embeddings para cada chunk

# COMMAND ----------

texts = df_chunks['chunk_text'].tolist()

print(f"Generando embeddings para {len(texts)} chunks...")
embeddings = model.encode(texts, show_progress_bar=True)
print(f"Embeddings generados: {embeddings.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4: Combinar chunks con embeddings y guardar

# COMMAND ----------

data = {
    'chunk_id': df_chunks['chunk_id'].tolist(),
    'source_document': df_chunks['source_document'].tolist(),
    'chunk_text': df_chunks['chunk_text'].tolist(),
    'embedding': [emb.tolist() for emb in embeddings]
}

embeddings_path = os.path.join(repo_root, EMBEDDINGS_FILE)
with open(embeddings_path, 'wb') as f:
    pickle.dump(data, f)

print(f"Archivo guardado: {embeddings_path}")
print(f"Total de registros: {len(data['chunk_id'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verificación final

# COMMAND ----------

with open(embeddings_path, 'rb') as f:
    verify = pickle.load(f)

print(f"Registros: {len(verify['chunk_id'])}")
print(f"Dimensión embedding: {len(verify['embedding'][0])}")
print(f"Ejemplo chunk: {verify['chunk_text'][0][:100]}...")
print(f"Ejemplo embedding (primeros 5 valores): {verify['embedding'][0][:5]}")