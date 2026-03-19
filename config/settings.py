#RAG Eval Framework-Configuration

# Chunking parameters
CHUNK_SIZE = 500          # caracteres por chunk
CHUNK_OVERLAP = 50        # caracteres de traslape entre chunks

# Paths
DATA_DIR = "data"         # carpeta con los .txt

# Delta table names
DATABASE = "rag_eval"
CHUNKS_TABLE = "document_chunks"
EMBEDDINGS_TABLE = "chunk_embeddings"