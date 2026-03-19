# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 - Ingesta y Chunking de Documentos
# MAGIC Este notebook lee los documentos .txt de la carpeta data/, 
# MAGIC los parte en chunks y los guarda en una tabla Delta.

# COMMAND ----------

# Imports
import os
import sys

# Agregar el root del repo al path para poder importar utils/ y config/
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in dir() else '/Workspace' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()).rsplit('/notebooks', 1)[0]

sys.path.insert(0, repo_root)

from utils.chunking import chunk_document
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR, DATABASE, CHUNKS_TABLE

print(f"Repo root: {repo_root}")
print(f"Data dir: {os.path.join(repo_root, DATA_DIR)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1: Leer los archivos .txt de la carpeta data/

# COMMAND ----------

# Construir el path a la carpeta data/
data_path = os.path.join(repo_root, DATA_DIR)

# Listar solo archivos .txt (ignorar gold_standard.csv y otros)
txt_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
txt_files.sort()

print(f"Encontrados {len(txt_files)} archivos:")
for f in txt_files:
    print(f"  - {f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2: Leer cada documento y aplicar chunking

# COMMAND ----------

# Procesar todos los documentos
all_chunks = []

for filename in txt_files:
    filepath = os.path.join(data_path, filename)
    
    # Leer el contenido del archivo
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Nombre limpio del source (sin extensión)
    source_name = filename.replace('.txt', '')
    
    # Aplicar chunking
    chunks = chunk_document(text, source_name, CHUNK_SIZE, CHUNK_OVERLAP)
    all_chunks.extend(chunks)
    
    print(f"  {filename}: {len(text)} caracteres -> {len(chunks)} chunks")

print(f"\nTotal de chunks generados: {len(all_chunks)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3: Crear DataFrame y guardar como tabla Delta

# COMMAND ----------

# Crear DataFrame de Spark
from pyspark.sql import Row

rows = [Row(**chunk) for chunk in all_chunks]
df_chunks = spark.createDataFrame(rows)

# Mostrar preview
display(df_chunks)

# COMMAND ----------

# Ver estadísticas por documento
from pyspark.sql.functions import count, avg, length

stats = df_chunks.groupBy("source_document").agg(
    count("*").alias("num_chunks"),
    avg(length("chunk_text")).alias("avg_chunk_length")
)
display(stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4: Guardar en tabla Delta

# COMMAND ----------

# Crear la base de datos si no existe
spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")

# Guardar como tabla Delta (sobrescribir si ya existe)
table_name = f"{DATABASE}.{CHUNKS_TABLE}"
df_chunks.write.mode("overwrite").saveAsTable(table_name)

print(f"Tabla guardada: {table_name}")
print(f"Total de registros: {df_chunks.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verificación final

# COMMAND ----------

# Leer la tabla para confirmar que se guardó correctamente
df_verify = spark.table(table_name)
print(f"Registros en tabla: {df_verify.count()}")
print(f"Columnas: {df_verify.columns}")
display(df_verify.limit(5))