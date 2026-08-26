"""
api/analyze.py
==============
Endpoints d'analyse de CV :
- /api/rag/analyze       : Analyse ATS simple (BERT + heuristique)
- /api/rag/analyze_rag   : Pipeline RAG complet (2 scores separes)
- /api/rag/index_cv      : Indexation d'un CV dans pgvector
"""

from flask import Blueprint, request, jsonify
from core.database import init_pgvector_tables
from core.models import get_embedding_model
from scoring.heuristic import ats_score
from scoring.bert_scorer import predict_ats_score
from rag.storage import store_cv_chunks
from rag.retrieval import retrieve_relevant_chunks
from rag.llm import check_ollama_available, generate_llm_analysis
from scoring.compatibility import compute_compatibility_score
from config import logger
import numpy as np

analyze_bp = Blueprint('analyze', __name__)


@analyze_bp.route('/api/rag/analyze', methods=['POST'])
def analyze_cv():
    """
    Analyse ATS d'un CV (sans offre d'emploi).
    Utilise BERT si disponible, sinon fallback heuristique.
    """
    data = request.get_json()
    cv_text = data.get('cv_text', '')
    job_description = data.get('job_description', None)

    if not cv_text:
        return jsonify({"error": "cv_text est requis"}), 400

    heuristic_result = ats_score(cv_text)

    bert_score_value, method = predict_ats_score(cv_text)
    if bert_score_value is None:
        bert_score_value = heuristic_result["score"]
        method = "heuristic_fallback"

    result = {
        "ats_score": bert_score_value,
        "ats_method": method,
        "is_ats_friendly": bert_score_value >= 55,
        "heuristic_score": heuristic_result["score"],
        "points_forts": heuristic_result["points_forts"],
        "a_ameliorer": heuristic_result["a_ameliorer"],
        "score_details": heuristic_result["details"],
    }

    if job_description:
        try:
            model = get_embedding_model()
            if model is not None:
                emb_cv = model.encode(cv_text)
                emb_job = model.encode(job_description)
                similarity = float(np.dot(emb_cv, emb_job) /
                                  (np.linalg.norm(emb_cv) * np.linalg.norm(emb_job) + 1e-8))
                result["semantic_similarity"] = round(similarity * 100, 1)
        except Exception as e:
            logger.error(f"Erreur similarite: {e}")

    return jsonify(result)


@analyze_bp.route('/api/rag/analyze_rag', methods=['POST'])
def analyze_rag():
    """
    Pipeline RAG COMPLET avec 2 scores separes :
    - ats_score : Qualite du CV (BERT + Heuristique)
    - compatibility_score : Adequation CV <-> Offre (Semantique + Competences + Experience + Mots-cles)
    """
    data = request.get_json()
    cv_text = data.get('cv_text', '')
    job_description = data.get('job_description', '')
    candidat_id = data.get('candidat_id', 'unknown')

    if not cv_text:
        return jsonify({"error": "cv_text est requis"}), 400

    result = {"candidat_id": candidat_id}

    # ================================================================
    # SCORE 1 : Qualite ATS du CV (BERT + Heuristique)
    # ================================================================
    bert_score_value, bert_method = predict_ats_score(cv_text)
    heuristic = ats_score(cv_text)

    ats_final = bert_score_value if bert_score_value is not None else heuristic["score"]
    if ats_final >= 75:
        ats_verdict = "Excellent - CV tres bien structure"
    elif ats_final >= 55:
        ats_verdict = "Bon - CV correctement structure"
    elif ats_final >= 35:
        ats_verdict = "Moyen - CV partiellement structure"
    else:
        ats_verdict = "Faible - CV mal structure"

    result["ats_score"] = {
        "score": ats_final,
        "method": bert_method,
        "heuristic_score": heuristic["score"],
        "points_forts": heuristic["points_forts"],
        "a_ameliorer": heuristic["a_ameliorer"],
        "score_details": heuristic["details"],
        "verdict": ats_verdict
    }

    # ================================================================
    # SCORE 2 : Compatibilite CV <-> Offre (si offre fournie)
    # ================================================================
    if job_description:
        try:
            model = get_embedding_model()
            cv_emb = None
            job_emb = None
            if model is not None:
                cv_emb = model.encode(cv_text)
                job_emb = model.encode(job_description)

            compat = compute_compatibility_score(cv_text, job_description, cv_emb, job_emb, model)

            total = compat["total_score"]
            if total >= 75:
                compat_verdict = "Excellent - Profil tres compatible"
            elif total >= 55:
                compat_verdict = "Bon - Profil compatible avec reserve"
            elif total >= 35:
                compat_verdict = "Moyen - Profil partiellement compatible"
            else:
                compat_verdict = "Faible - Profil peu compatible"

            result["compatibility_score"] = {
                "score": total,
                "semantic_similarity": compat["semantic_similarity"],
                "skill_match": compat["skill_match"],
                "experience_match": compat["experience_match"],
                "keyword_match": compat["keyword_match"],
                "weights": compat["weights"],
                "matched_skills": compat.get("matched_skills", []),
                "missing_skills": compat.get("missing_skills", []),
                "cv_experience_years": compat.get("cv_experience_years", 0),
                "required_experience_years": compat.get("required_experience_years", 0),
                "recommandation": compat.get("recommandation", ""),
                "explication": compat.get("explication", ""),
                "verdict": compat_verdict
            }
        except Exception as e:
            logger.error(f"Erreur compatibilite: {e}")
            result["compatibility_score"] = None

    # ================================================================
    # RAG : Chunking + Retrieval + LLM
    # ================================================================
    llm_analysis = None
    relevant_chunks = []

    if job_description:
        try:
            init_pgvector_tables()
            store_cv_chunks(candidat_id, cv_text)
            relevant_chunks = retrieve_relevant_chunks(job_description, candidat_id, top_k=5)
            result["retrieved_chunks"] = len(relevant_chunks)
        except Exception as e:
            logger.error(f"Erreur RAG retrieval: {e}")
            result["retrieved_chunks"] = 0

    ollama_available = check_ollama_available()
    result["ollama_available"] = ollama_available

    if ollama_available and job_description and relevant_chunks:
        llm_analysis = generate_llm_analysis(
            relevant_chunks=relevant_chunks,
            job_description=job_description,
            cv_text=cv_text,
            ats_score=ats_final,
            heuristic_score=heuristic["score"]
        )
        if llm_analysis:
            result["llm_analysis"] = llm_analysis

    return jsonify(result)


@analyze_bp.route('/api/rag/index_cv', methods=['POST'])
def index_cv():
    """
    Indexe un CV dans le pipeline RAG :
    chunking + embeddings + stockage pgvector.
    """
    data = request.get_json()
    cv_text = data.get('cv_text', '')
    candidat_id = data.get('candidat_id')

    if not cv_text or not candidat_id:
        return jsonify({"error": "cv_text et candidat_id sont requis"}), 400

    try:
        init_pgvector_tables()
        nb_chunks = store_cv_chunks(candidat_id, cv_text)
        return jsonify({
            "message": "CV indexe avec succes",
            "candidat_id": candidat_id,
            "chunks_stored": nb_chunks
        })
    except Exception as e:
        logger.error(f"Erreur indexation: {e}")
        return jsonify({"error": str(e)}), 500