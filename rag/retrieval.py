"""
rag/retrieval.py
================
Récupération (Retrieval) des chunks les plus pertinents par
similarité cosinus via pgvector.
L'opérateur <#> de pgvector calcule la distance négative cosinus.
"""

from core.models import get_embedding_model
from core.database import get_db_connection
from config import logger


def retrieve_relevant_chunks(job_description, candidat_id, top_k=5):
    """
    Récupère les top_k chunks d'un candidat les plus similaires
    à la description de poste.

    Args:
        job_description (str): Texte de l'offre d'emploi
        candidat_id (str): ID du candidat à interroger
        top_k (int): Nombre de chunks à retourner

    Returns:
        list[dict]: Chunks avec leur score de similarité
    """
    model = get_embedding_model()
    if model is None:
        return []

    # Encode la requête (offre d'emploi)
    query_embedding = model.encode(job_description).tolist()

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # <#> = distance négative cosinus → ORDER BY <#> = plus proches d'abord
        # 1 - (embedding <#> vec) = similarité cosinus entre 0 et 1
        cur.execute("""
            SELECT chunk_text,
                   1 - (embedding <#> %s::vector) as similarity
            FROM rag_cv_chunks
            WHERE candidat_id = %s
            ORDER BY embedding <#> %s::vector
            LIMIT %s
        """, (str(query_embedding), candidat_id, str(query_embedding), top_k))

        results = []
        for row in cur.fetchall():
            results.append({
                "chunk_text": row[0],
                "similarity": round(float(row[1]), 4)
            })

        cur.close()
        conn.close()
        return results

    except Exception as e:
        logger.error(f"❌ Erreur retrieval chunks: {e}")
        return []
