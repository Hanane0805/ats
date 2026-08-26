"""
api/compatibility.py
====================
Endpoint de calcul du score de compatibilité entre un CV et une offre.
Combine similarité sémantique, match de compétences, expérience et mots-clés.
"""

from flask import Blueprint, request, jsonify
from core.models import get_embedding_model
from scoring.compatibility import compute_compatibility_score
from config import logger

compatibility_bp = Blueprint('compatibility', __name__)


@compatibility_bp.route('/api/rag/compatibility', methods=['POST'])
def compatibility_score():
    """
    Calcule le score de compatibilité détaillé CV ↔ Offre.
    """
    data = request.get_json()
    cv_text = data.get('cv_text', '')
    job_description = data.get('job_description', '')

    if not cv_text or not job_description:
        return jsonify({"error": "cv_text et job_description sont requis"}), 400

    try:
        model = get_embedding_model()
        cv_emb = None
        job_emb = None
        if model is not None:
            cv_emb = model.encode(cv_text)
            job_emb = model.encode(job_description)

        result = compute_compatibility_score(cv_text, job_description, cv_emb, job_emb, model)
        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ Erreur compatibilité: {e}")
        return jsonify({"error": str(e)}), 500
