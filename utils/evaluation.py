def hit_at_k(results, expected_source, k):
    """
    Revisa si el documento correcto aparece en los top K resultados.
    Returns: 1 si aparece, 0 si no.
    """
    top_k_sources = [r['source_document'] for r in results[:k]]
    return 1 if expected_source in top_k_sources else 0

def reciprocal_rank(results, expected_source):
    """
    Calcula 1/posición del documento correcto.
    Si es primero = 1.0, si es tercero = 0.33, si no aparece = 0.0
    """
    for i, r in enumerate(results):
        if r['source_document'] == expected_source:
            return 1.0 / (i + 1)
    return 0.0

def find_rank(results, expected_source):
    """
    Encuentra la posición del documento correcto.
    Returns: posición (1-indexed) o -1 si no aparece.
    """
    for i, r in enumerate(results):
        if r['source_document'] == expected_source:
            return i + 1
    return -1