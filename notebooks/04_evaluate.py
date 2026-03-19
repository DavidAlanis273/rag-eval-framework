import os
import pandas as pd
df = pd.read_csv("/Workspace/Users/david.alanis@watlow.com/rag-eval-framework/outputs/chunks.csv")
for _, row in df.iterrows():
    print(f"{row['chunk_id']} | {row['source_document']} | {row['chunk_text'][:80]}")