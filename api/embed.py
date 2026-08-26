"""
api/embed.py
============
Endpoint pour stocker l'embedding global d'un CV dans pgvector.
Utilisé pour la recherche sémantique globale (pas le RAG chunké).
"""

from flask import Blueprint, request, jsonify
from core.models import get_embedding_model
from core.database import get_db_connection
from config import logger

embed_bp = Blueprint('embed', __name__)


@embed_bp.route('/api/rag/embed', methods=['POST'])
def embed_cv():
    """
    Encode un CV et stocke son embedding global dans cv_embeddings.
    Si le candidat existe déjà, met à jour l'embedding (UPSERT).
    """
    data = request.get_json()
    cv_text = data.get('cv_text', '')
    candidat_id = data.get('candidat_id')

    if not cv_text or not candidat_id:
        return jsonify({"error": "cv_text et candidat_id sont requis"}), 400

    try:
        model = get_embedding_model()
        if model is None:
            return jsonify({"error": "Modèle embeddings non disponible"}), 503

        embedding = model.encode(cv_text).tolist()

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO cv_embeddings (candidat_id, embedding, cv_text)
            VALUES (%s, %s, %s)
            ON CONFLICT (candidat_id)
            DO UPDATE SET embedding = EXCLUDED.embedding, cv_text = EXCLUDED.cv_text
        """, (candidat_id, str(embedding), cv_text))
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"message": "Embedding stocké", "candidat_id": candidat_id})

    except Exception as e:
        logger.error(f"❌ Erreur embed: {e}")
        return jsonify({"error": str(e)}), 500
