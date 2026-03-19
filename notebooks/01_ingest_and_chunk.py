# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 - Ingesta y Chunking de Documentos
# MAGIC Este notebook lee los documentos .txt de la carpeta data/, 
# MAGIC los parte en chunks y los guarda como CSV.

# COMMAND ----------

import os
import sys
import pandas as pd

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in dir() else '/Workspace' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()).rsplit('/notebooks', 1)[0]

sys.path.insert(0, repo_root)

from utils.chunking import chunk_document
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR, OUTPUT_DIR, CHUNKS_FILE

print(f"Repo root: {repo_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1: Leer los archivos .txt de la carpeta data/

# COMMAND ----------

data_path = os.path.join(repo_root, DATA_DIR)

txt_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
txt_files.sort()

print(f"Encontrados {len(txt_files)} archivos:")
for f in txt_files:
    print(f"  - {f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2: Leer cada documento y aplicar chunking

# COMMAND ----------

all_chunks = []

for filename in txt_files:
    filepath = os.path.join(data_path, filename)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    source_name = filename.replace('.txt', '')
    chunks = chunk_document(text, source_name, CHUNK_SIZE, CHUNK_OVERLAP)
    all_chunks.extend(chunks)
    
    print(f"  {filename}: {len(text)} caracteres -> {len(chunks)} chunks")

print(f"\nTotal de chunks generados: {len(all_chunks)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3: Crear DataFrame con pandas

# COMMAND ----------

df_chunks = pd.DataFrame(all_chunks)
print(f"Shape: {df_chunks.shape}")
print(f"Columnas: {list(df_chunks.columns)}")
df_chunks.head(10)

# COMMAND ----------

# Estadísticas por documento
stats = df_chunks.groupby('source_document').agg(
    num_chunks=('chunk_id', 'count'),
    avg_chunk_length=('chunk_text', lambda x: x.str.len().mean())
).round(0)
stats

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4: Guardar como CSV

# COMMAND ----------

output_path = os.path.join(repo_root, OUTPUT_DIR)
os.makedirs(output_path, exist_ok=True)

chunks_file = os.path.join(repo_root, CHUNKS_FILE)
df_chunks.to_csv(chunks_file, index=False)

print(f"Archivo guardado: {chunks_file}")
print(f"Total de registros: {len(df_chunks)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verificación final

# COMMAND ----------

df_verify = pd.read_csv(chunks_file)
print(f"Registros: {len(df_verify)}")
print(f"Columnas: {list(df_verify.columns)}")
df_verify.head(5)