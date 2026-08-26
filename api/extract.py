"""
api/extract.py
==============
Endpoint d'extraction d'informations structurées depuis un CV brut :
email, téléphone, nom, compétences, années d'expérience.
"""

from flask import Blueprint, request, jsonify
from extraction.info_extractor import extract_full_profile

extract_bp = Blueprint('extract', __name__)


@extract_bp.route('/api/rag/extract', methods=['POST'])
def extract_info():
    """
    Extrait les informations clés d'un CV en une seule passe.
    """
    data = request.get_json()
    cv_text = data.get('cv_text', '')

    if not cv_text:
        return jsonify({"error": "cv_text est requis"}), 400

    profile = extract_full_profile(cv_text)
    return jsonify(profile)
