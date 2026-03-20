# Databricks notebook source

# COMMAND ----------

# MAGIC 
%pip install sentence-transformers --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 - Evaluación del Retrieval
# MAGIC Este notebook corre las preguntas del gold standard contra el sistema
# MAGIC de búsqueda y calcula métricas de precisión.

# COMMAND ----------

import os
import sys
import pandas as pd

repo_root = '/Workspace/Users/david.alanis@watlow.com/rag-eval-framework'
sys.path.insert(0, repo_root)

from sentence_transformers import SentenceTransformer
from utils.retrieval import load_embeddings, retrieve_top_k
from utils.evaluation import hit_at_k, reciprocal_rank, find_rank
from config.settings import EMBEDDINGS_FILE

# Cargar modelo y embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_path = os.path.join(repo_root, EMBEDDINGS_FILE)
data = load_embeddings(embeddings_path)

print(f"Modelo cargado")
print(f"Embeddings cargados: {len(data['chunk_id'])} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1: Cargar el Gold Standard

# COMMAND ----------

gold_path = os.path.join(repo_root, "data/gold_standard.csv")
df_gold = pd.read_csv(gold_path)

print(f"Preguntas de evaluación: {len(df_gold)}")
df_gold.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2: Correr cada pregunta por el sistema de retrieval

# COMMAND ----------

results_list = []

for _, row in df_gold.iterrows():
    query_id = row['query_id']
    query_text = row['query_text']
    expected = row['expected_source_document']
    
    # Buscar top 5
    results = retrieve_top_k(query_text, model, data, k=5)
    
    # Calcular métricas
    h3 = hit_at_k(results, expected, 3)
    h5 = hit_at_k(results, expected, 5)
    rr = reciprocal_rank(results, expected)
    rank = find_rank(results, expected)
    
    # Guardar resultado
    results_list.append({
        'query_id': query_id,
        'query_text': query_text,
        'expected_source': expected,
        'top1_source': results[0]['source_document'],
        'top1_score': round(results[0]['similarity_score'], 4),
        'rank_of_correct': rank,
        'hit_at_3': h3,
        'hit_at_5': h5,
        'reciprocal_rank': round(rr, 4),
        'top1_text': results[0]['chunk_text'][:100]
    })
    
    status = "PASS" if h3 == 1 else ("TOP5" if h5 == 1 else "FAIL")
    print(f"  {query_id} | {status} | Rank: {rank} | Score: {results[0]['similarity_score']:.4f} | {query_text[:60]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3: Tabla de resultados completa

# COMMAND ----------

df_results = pd.DataFrame(results_list)
df_results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4: Métricas globales

# COMMAND ----------

total = len(df_results)
hit3 = df_results['hit_at_3'].sum()
hit5 = df_results['hit_at_5'].sum()
mrr = df_results['reciprocal_rank'].mean()

print("=" * 50)
print("   RETRIEVAL EVALUATION RESULTS")
print("=" * 50)
print(f"   Total queries:     {total}")
print(f"   Hit@3:             {hit3}/{total} ({hit3/total*100:.1f}%)")
print(f"   Hit@5:             {hit5}/{total} ({hit5/total*100:.1f}%)")
print(f"   MRR:               {mrr:.4f}")
print("=" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5: Análisis de fallas

# COMMAND ----------

df_fails = df_results[df_results['hit_at_5'] == 0]

if len(df_fails) == 0:
    print("No hubo fallas - todos los queries encontraron el documento correcto en top 5")
else:
    print(f"Queries que FALLARON ({len(df_fails)}):\n")
    for _, row in df_fails.iterrows():
        print(f"  {row['query_id']}: {row['query_text']}")
        print(f"    Expected: {row['expected_source']}")
        print(f"    Got:      {row['top1_source']} (score: {row['top1_score']})")
        print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 6: Resultados por documento

# COMMAND ----------

doc_stats = df_results.groupby('expected_source').agg(
    total_queries=('query_id', 'count'),
    hits_at_3=('hit_at_3', 'sum'),
    hits_at_5=('hit_at_5', 'sum'),
    avg_mrr=('reciprocal_rank', 'mean')
).round(4)

doc_stats

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 7: Guardar resultados

# COMMAND ----------

output_path = os.path.join(repo_root, "outputs/evaluation_results.csv")
df_results.to_csv(output_path, index=False)
print(f"Resultados guardados: {output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen para presentación
# MAGIC 
# MAGIC Los números de arriba son tu slide 1.
# MAGIC Las fallas de arriba son tu slide 2.

# COMMAND ----------

print("\n SLIDE 1 - Headline Numbers:")
print(f"   Hit@3 = {hit3/total*100:.0f}%  |  Hit@5 = {hit5/total*100:.0f}%  |  MRR = {mrr:.2f}")

print(f"\n SLIDE 2 - {len(df_fails)} queries failed (not found in top 5)")
for _, row in df_fails.iterrows():
    print(f"   '{row['query_text'][:50]}...'")
    print(f"   Expected: {row['expected_source']} → Got: {row['top1_source']}")