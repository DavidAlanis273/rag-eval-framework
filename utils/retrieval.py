import pickle
import numpy as np

def load_embeddings(embeddings_path):
    """Carga los embeddings desde el archivo pickle."""
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    return data

def cosine_similarity(vec1, vec2):
    """Calcula la similitud coseno entre dos vectores."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def retrieve_top_k(query, model, data, k=5):
    """
    Busca los K chunks más similares a la query.
    
    Args:
        query: texto de la pregunta
        model: modelo de SentenceTransformer
        data: diccionario con chunk_id, source_document, chunk_text, embedding
        k: número de resultados a devolver
    
    Returns:
        Lista de diccionarios con los top K resultados
    """
    # Generar embedding de la query
    query_embedding = model.encode(query)
    
    # Calcular similitud contra todos los chunks
    results = []
    for i in range(len(data['chunk_id'])):
        score = cosine_similarity(query_embedding, data['embedding'][i])
        results.append({
            'chunk_id': data['chunk_id'][i],
            'source_document': data['source_document'][i],
            'chunk_text': data['chunk_text'][i],
            'similarity_score': float(score)
        })
    
    # Ordenar por score descendente y devolver top K
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results[:k]