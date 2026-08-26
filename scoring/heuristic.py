"""
scoring/heuristic.py
====================
Système de scoring heuristique basé sur 14 règles regex.
Chaque règle vérifie la présence d'un élément essentiel dans un CV
et attribue un poids. Le score final est un pourcentage sur 130 pts.
"""

import re

# -----------------------------------------------------------
# Dictionnaire des règles : regex + poids
# -----------------------------------------------------------
ATS_RULES = {
    "email":       {"regex": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', "weight": 10},
    "phone":       {"regex": r'\+?\d[\d\s\-]{8,}\d', "weight": 8},
    "experience":  {"regex": r'(experience|expérience|parcours|emploi|career|work|job|poste)', "weight": 15},
    "education":   {"regex": r'(education|formation|diplome|diploma|degree|university|université|bachelor|master|phd|baccalaureat|baccalauréat)', "weight": 15},
    "skills":      {"regex": r'(skill|competence|compétence|technologie|programming|langage|outil|logiciel)', "weight": 12},
    "length":      {"regex": None, "weight": 10},
    "dates":       {"regex": r'(\d{4}|janvier|fevrier|février|mars|avril|mai|juin|juillet|aout|août|septembre|octobre|novembre|decembre|décembre|january|february|march|april|may|june|july|august|september|october|november|december)', "weight": 8},
    "keywords":    {"regex": r'(java|python|sql|react|docker|git|api|linux|aws|azure|machine learning|spring|angular|typescript|javascript|html|css|nosql|mongodb|postgresql)', "weight": 10},
    "objective":   {"regex": r'(objective|profil|summary|about|objectif|résumé|resume)', "weight": 7},
    "clean":       {"regex": None, "weight": 5},
    "job_titles":  {"regex": r'(développeur|developpeur|developer|analyste|manager|consultant|ingénieur|ingenieur|engineer|chef de projet|product owner|scrum master|data scientist|devops|administrateur|administrator)', "weight": 8},
    "quantified":  {"regex": r'\b\d+%?\b.*(augment|réduit|créé|développé|géré|optimisé|amélioré|réalisé|diminué|boost|réduire|augmenter|développer|créer|gérer|optimiser|améliorer|réaliser|diminuer|généré|rapporté|économisé|réduits|augmentations)', "weight": 10},
    "acronyms":    {"regex": r'(?=.*\b(AWS|API|NLP|ML|AI|CI/CD|KPI|ERP|CRM|SaaS)\b)(?=.*\b(Amazon Web Services|Application Programming Interface|Natural Language Processing|Machine Learning|Artificial Intelligence|Continuous Integration|Key Performance Indicator|Enterprise Resource Planning|Customer Relationship Management|Software as a Service)\b)', "weight": 5},
    "reverse_chrono": {"regex": None, "weight": 7}
}

# Noms lisibles des règles pour l'affichage
RULE_NAMES = {
    "email": "Email de contact",
    "phone": "Téléphone de contact",
    "experience": "Section Expérience",
    "education": "Section Education/Formation",
    "skills": "Section Compétences",
    "length": "Longueur appropriée (200-5000 mots)",
    "dates": "Dates présentes",
    "keywords": "Mots-clés techniques",
    "objective": "Objectif/Profil/Résumé",
    "clean": "Format propre (pas de caractères spéciaux)",
    "job_titles": "Titres de postes mentionnés",
    "quantified": "Réalisations quantifiées (chiffres)",
    "acronyms": "Acronymes expliqués",
    "reverse_chrono": "Ordre chronologique inverse",
}


def ats_score(text):
    """
    Calcule le score heuristique ATS d'un CV.

    Pour chaque règle :
    - Si regex présente dans le texte → points complets
    - Si règle spéciale (length, clean, reverse_chrono) → logique dédiée
    - Si chronologie non respectée → demi-points

    Args:
        text (str): Texte brut du CV

    Returns:
        dict: Score, détails par règle, points forts et axes d'amélioration
    """
    total = sum(r["weight"] for r in ATS_RULES.values())
    earned = 0
    details = []

    for key, rule in ATS_RULES.items():
        matched = False
        points = 0

        # -----------------------------------------------------------
        # Règles sans regex (logique programmée)
        # -----------------------------------------------------------
        if rule["regex"] is None:
            if key == "length" and 200 <= len(text.split()) <= 5000:
                matched = True
                points = rule["weight"]
            elif key == "clean" and len(re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text)) == 0:
                matched = True
                points = rule["weight"]
            elif key == "reverse_chrono":
                years = re.findall(r'\b(19|20)\d{2}\b', text)
                if len(years) >= 2:
                    unique_years = []
                    for y in years:
                        if not unique_years or unique_years[-1] != y:
                            unique_years.append(y)
                    # Vérifie l'ordre décroissant
                    if all(int(unique_years[i]) >= int(unique_years[i+1])
                           for i in range(len(unique_years)-1)):
                        matched = True
                        points = rule["weight"]
                    else:
                        matched = False
                        points = rule["weight"] * 0.5  # Demi-points si désordonné
                else:
                    matched = False
                    points = rule["weight"] * 0.5
        # -----------------------------------------------------------
        # Règles avec regex
        # -----------------------------------------------------------
        else:
            if re.search(rule["regex"], text, re.I):
                matched = True
                points = rule["weight"]

        earned += points
        details.append({
            "rule": RULE_NAMES.get(key, key),
            "matched": matched,
            "points_earned": round(points, 1),
            "points_max": rule["weight"]
        })

    points_forts = [d["rule"] for d in details if d["matched"] and d["points_earned"] == d["points_max"]]
    a_ameliorer = [d["rule"] for d in details if not d["matched"]]

    return {
        "score": round((earned / total) * 100, 1) if total > 0 else 0,
        "details": details,
        "points_forts": points_forts,
        "a_ameliorer": a_ameliorer,
        "points_earned": round(earned, 1),
        "points_total": total
    }
