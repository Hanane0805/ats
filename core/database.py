"""
core/database.py
================
Gestion des connexions à PostgreSQL avec psycopg2.
Toutes les fonctions de persistance utilisent get_db_connection().
"""

import psycopg2
from config import DB_CONFIG, logger


def get_db_connection():
    """
    Crée et retourne une connexion PostgreSQL fraîche.
    L'appelant est responsable de fermer la connexion (conn.close()).
    """
    return psycopg2.connect(**DB_CONFIG)


def init_pgvector_tables():
    """
    Initialise les tables nécessaires au RAG avec l'extension pgvector :
    - rag_cv_chunks   : chunks de CV avec leurs embeddings (384d)
    - cv_embeddings   : embedding global du CV par candidat
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Activation de l'extension vectorielle
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()

        # Table des chunks pour le RAG
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_cv_chunks (
                id BIGSERIAL PRIMARY KEY,
                candidat_id VARCHAR(255),
                chunk_text TEXT NOT NULL,
                document_cv_id BIGINT,
                index_ordre INT,
                embedding vector(384)
            )
        """)

        # Index sur candidat_id pour accélérer le retrieval
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_cv_chunks_candidat_id
            ON rag_cv_chunks (candidat_id)
        """)

        # Table d'embedding global par candidat
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cv_embeddings (
                candidat_id VARCHAR(255) PRIMARY KEY,
                embedding vector(384),
                cv_text TEXT
            )
        """)

        conn.commit()
        cur.close()
        conn.close()
        logger.info("✅ Tables pgvector initialisées")
    except Exception as e:
        logger.warning(f"⚠️ Tables pgvector non initialisées: {e}")
