"""
api/health.py
=============
Endpoints de vérification de l'état du service et de ses composants.
"""

from flask import Blueprint, jsonify
from core.models import _ats_model, _embedding_model
from core.database import get_db_connection
from rag.llm import check_ollama_available

health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health():
    """Endpoint de base : vérifie que le service répond."""
    return jsonify({"status": "ok", "service": "ats-rag-service"})


@health_bp.route('/api/rag/health', methods=['GET'])
def rag_health():
    """
    Vérifie l'état de tous les composants du pipeline RAG :
    - Ollama (LLM)
    - PostgreSQL + pgvector
    - Modèle BERT
    - Modèle d'embeddings
    """
    ollama_ok = check_ollama_available()

    try:
        conn = get_db_connection()
        conn.close()
        db_ok = True
    except Exception:
        db_ok = False

    return jsonify({
        "service": "ats-rag-service",
        "status": "ok",
        "components": {
            "ollama_llama32_3b": ollama_ok,
            "postgresql_pgvector": db_ok,
            "bert_model": _ats_model is not None,
            "embedding_model": _embedding_model is not None
        }
    })
