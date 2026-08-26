"""
rag/storage.py
==============
Fonctions de persistance des chunks et embeddings dans PostgreSQL
via l'extension pgvector.
"""

from core.models import get_embedding_model
from core.database import get_db_connection, init_pgvector_tables
from rag.chunking import chunk_cv
from config import logger


def store_cv_chunks(candidat_id, cv_text):
    """
    Pipeline complet : chunk + encode + stocke dans rag_cv_chunks.

    Args:
        candidat_id (str): Identifiant unique du candidat
        cv_text (str): Texte brut du CV

    Returns:
        int: Nombre de chunks stockés (0 en cas d'erreur)
    """
    try:
        model = get_embedding_model()
        if model is None:
            logger.error("❌ Modèle embeddings non disponible, indexation annulée")
            return 0

        # 1. Chunking
        chunks = chunk_cv(cv_text)
        if not chunks:
            return 0

        # 2. Stockage
        conn = get_db_connection()
        cur = conn.cursor()

        # Nettoyage des anciens chunks du candidat
        cur.execute("DELETE FROM rag_cv_chunks WHERE candidat_id = %s", (candidat_id,))

        for i, chunk_text in enumerate(chunks):
            embedding = model.encode(chunk_text)
            cur.execute(
                """INSERT INTO rag_cv_chunks
                   (candidat_id, chunk_text, document_cv_id, index_ordre, embedding)
                   VALUES (%s, %s, %s, %s, %s)""",
                (candidat_id, chunk_text, None, i, embedding.tolist())
            )

        conn.commit()
        cur.close()
        conn.close()

        logger.info(f"✅ {len(chunks)} chunks stockés pour candidat {candidat_id}")
        return len(chunks)

    except Exception as e:
        logger.error(f"❌ Erreur stockage chunks: {e}")
        return 0
