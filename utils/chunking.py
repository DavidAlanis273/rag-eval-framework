def chunk_document(text, source_name, chunk_size=500, overlap=50):
    """
    Toma un documento completo y lo parte en chunks con traslape.
    
    Args:
        text: texto completo del documento
        source_name: nombre del archivo fuente (ej: '01_nanodac_recorder')
        chunk_size: tamaño máximo de cada chunk en caracteres
        overlap: cantidad de caracteres que se repiten entre chunks
    
    Returns:
        Lista de diccionarios, cada uno con chunk_id, source, y text
    """
    # Limpiar texto
    text = text.strip()
    
    # Si el texto es más corto que el chunk_size, devolver un solo chunk
    if len(text) <= chunk_size:
        return [{
            "chunk_id": f"{source_name}_chunk_000",
            "source_document": source_name,
            "chunk_text": text
        }]
    
    chunks = []
    start = 0
    chunk_num = 0
    
    while start < len(text):
        # Tomar el pedazo de texto
        end = start + chunk_size
        
        # Si no es el último chunk, intentar cortar en un punto natural
        if end < len(text):
            # Buscar el último salto de línea o punto dentro del rango
            last_newline = text.rfind('\n', start, end)
            last_period = text.rfind('. ', start, end)
            
            # Elegir el mejor punto de corte
            cut_point = max(last_newline, last_period)
            if cut_point > start:
                end = cut_point + 1
        
        chunk_text = text[start:end].strip()
        
        if chunk_text:  # Solo agregar si tiene contenido
            chunks.append({
                "chunk_id": f"{source_name}_chunk_{chunk_num:03d}",
                "source_document": source_name,
                "chunk_text": chunk_text
            })
            chunk_num += 1
        
        # Mover el inicio considerando el traslape
        start = end - overlap
    
    return chunks