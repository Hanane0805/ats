"""
api/search.py
=============
Endpoint de recherche sémantique globale sur tous les CVs indexés.
Retourne les candidats les plus similaires à une requête textuelle.
"""

from flask import Blueprint, request, jsonify
from core.models import get_embedding_model
from core.database import get_db_connection
from config import logger

search_bp = Blueprint('search', __name__)


@search_bp.route('/api/rag/search', methods=['POST'])
def semantic_search():
    """
    Recherche sémantique : trouve les CVs les plus proches d'une requête.
    Utilise la table cv_embeddings (embedding global, pas chunké).
    """
    data = request.get_json()
    query = data.get('query', '')
    top_k = data.get('top_k', 10)

    if not query:
        return jsonify({"error": "query est requise"}), 400

    try:
        model = get_embedding_model()
        if model is None:
            return jsonify({"error": "Modèle embeddings non disponible"}), 503

        query_embedding = model.encode(query).tolist()

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT candidat_id, cv_text,
                   1 - (embedding <#> %s::vector) as similarity
            FROM cv_embeddings
            ORDER BY embedding <#> %s::vector
            LIMIT %s
        """, (str(query_embedding), str(query_embedding), top_k))

        results = [
            {
                "candidat_id": r[0],
                "cv_text": r[1],
                "similarity": round(float(r[2]), 4)
            }
            for r in cur.fetchall()
        ]

        cur.close()
        conn.close()
        return jsonify({"results": results, "query": query, "total": len(results)})

    except Exception as e:
        logger.error(f"❌ Erreur recherche: {e}")
        return jsonify({"error": str(e)}), 500
