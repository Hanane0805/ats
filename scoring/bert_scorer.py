"""
scoring/bert_scorer.py
======================
Wrapper autour du modèle BERT fine-tuné pour la classification ATS.
Simplifie l'appel au modèle et gère les cas où le modèle n'est pas disponible.
"""

import torch
from core.models import get_ats_model
from config import logger


def predict_ats_score(cv_text):
    """
    Classifie un CV avec le modèle BERT et retourne un score 0-100.

    Args:
        cv_text (str): Texte du CV

    Returns:
        tuple[float, str]: (score, méthode)
            score : probabilité classe 1 × 100
            méthode : "bert" ou "unavailable"
    """
    model, tokenizer = get_ats_model()

    if model is None or tokenizer is None:
        return None, "unavailable"

    try:
        inputs = tokenizer(cv_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            ats_prob = probs[0][1].item()  # Probabilité classe "ATS-friendly"

        score = round(ats_prob * 100, 1)
        return score, "bert"
    except Exception as e:
        logger.error(f"❌ Erreur prédiction BERT: {e}")
        return None, "error"
