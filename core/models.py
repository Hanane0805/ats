"""
core/models.py
==============
Gestion du chargement "lazy" (à la demande) des modèles de Machine Learning.
Cela évite de charger BERT et SentenceTransformer au démarrage du serveur,
 ce qui réduit considérablement le temps de boot et la consommation mémoire.
"""

import os
from config import logger, MODEL_DIR

# -----------------------------------------------------------
# Variables globales pour le cache des modèles
# -----------------------------------------------------------
_ats_model = None          # Modèle BERT fine-tuné pour la classification ATS
_ats_tokenizer = None      # Tokenizer associé au modèle BERT
_embedding_model = None    # Modèle SentenceTransformer pour les embeddings


def load_ats_model():
    """
    Charge le modèle BERT de classification ATS (classe 0/1).
    Le modèle est mis en cache après le premier appel.
    """
    global _ats_model, _ats_tokenizer
    if _ats_model is not None:
        return _ats_model, _ats_tokenizer

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if os.path.exists(MODEL_DIR):
            _ats_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            _ats_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            _ats_model.eval()  # Mode évaluation (pas d'entraînement)
            logger.info(f"✅ Modèle ATS chargé depuis {MODEL_DIR}")
        else:
            logger.warning(f"⚠️ Modèle ATS non trouvé dans {MODEL_DIR}")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle ATS: {e}")

    return _ats_model, _ats_tokenizer


def load_embedding_model():
    """
    Charge le modèle SentenceTransformer pour générer les embeddings
    sémantiques (384 dimensions, multilingue).
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    try:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        logger.info("✅ Modèle embeddings chargé: paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle embeddings: {e}")

    return _embedding_model


def get_ats_model():
    """Retourne le modèle BERT (le charge si nécessaire)."""
    return load_ats_model()


def get_embedding_model():
    """Retourne le modèle d'embeddings (le charge si nécessaire)."""
    return load_embedding_model()
