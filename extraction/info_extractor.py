"""
extraction/info_extractor.py
=============================
Fonctions utilitaires d'extraction d'informations personnelles
et professionnelles depuis un texte de CV brut.
"""

import re
from scoring.compatibility import extract_skills, extract_experience_years


def extract_contact_info(cv_text):
    """
    Extrait les informations de contact d'un CV.

    Returns:
        dict: {email, phone, name}
    """
    # Email standard
    email = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', cv_text)

    # Téléphone (format international ou local)
    phone = re.search(r'\+?\d[\d\s\-]{8,}\d', cv_text)

    # Nom : début de ligne avec 2+ mots capitalisés
    name_match = re.search(
        r'^([A-ZÀÂÉÈÊËÏÎÔÙÛÜŸÇÆŒ][a-zàâéèêëïîôùûüÿçæœ]+'
        r'(?:\s+[A-ZÀÂÉÈÊËÏÎÔÙÛÜŸÇÆŒ][a-zàâéèêëïîôùûüÿçæœ]+)+)',
        cv_text, re.M
    )

    return {
        "email": email.group(0) if email else None,
        "phone": phone.group(0) if phone else None,
        "name": name_match.group(0).strip() if name_match else None,
    }


def extract_full_profile(cv_text):
    """
    Extraction complète du profil candidat.

    Returns:
        dict: {email, phone, name, skills, experience_years}
    """
    contact = extract_contact_info(cv_text)
    return {
        **contact,
        "skills": extract_skills(cv_text),
        "experience_years": extract_experience_years(cv_text),
    }
