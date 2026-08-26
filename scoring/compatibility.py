"""
scoring/compatibility.py
========================
Calcul du score de compatibilité entre un CV et une offre d'emploi.
4 dimensions pondérées :
- Similarité sémantique (embeddings) : 40%
- Correspondance des compétences tech : 30%
- Adéquation de l'expérience (années) : 20%
- Correspondance des mots-clés (Jaccard) : 10%
"""

import re
import numpy as np

# -----------------------------------------------------------
# Référentiel de compétences techniques par catégorie
# -----------------------------------------------------------
TECH_KEYWORDS = {
    "langages": ["java", "python", "javascript", "typescript", "c#", "c++", "go", "rust", "php", "ruby", "swift", "kotlin"],
    "frameworks": ["spring", "react", "angular", "vue", "django", "flask", "express", "next.js", "node.js", ".net", "laravel"],
    "bases_donnees": ["sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle", "sqlite", "cassandra"],
    "devops": ["docker", "kubernetes", "jenkins", "git", "ci/cd", "terraform", "ansible", "aws", "azure", "gcp"],
    "data_ia": ["machine learning", "deep learning", "nlp", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "keras", "spark"],
    "outils": ["linux", "git", "maven", "gradle", "jira", "agile", "scrum", "rest", "api", "microservices"],
}

# Stop-words français et anglais à ignorer pour le Jaccard
STOP_WORDS = {
    'le', 'la', 'les', 'de', 'des', 'du', 'et', 'en', 'un', 'une', 'est', 'que', 'qui', 'dans',
    'pour', 'par', 'sur', 'avec', 'ce', 'cette', 'son', 'sa', 'ses', 'au', 'aux',
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'and', 'or', 'this', 'that'
}


def extract_skills(text):
    """
    Extrait les compétences techniques d'un texte.
    Combine deux approches :
    1. Recherche des mots-clés du référentiel TECH_KEYWORDS
    2. Parsing de la section "Compétences/Skills" du texte

    Returns:
        list[str]: Liste des compétences trouvées (unique, lowercase)
    """
    text_lower = text.lower()
    found = []

    # Approche 1 : mots-clés du référentiel
    for category, keywords in TECH_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(kw)

    # Approche 2 : parsing de la section compétences
    skill_section = re.split(r'(compétences?|skills?|technologies?|outils?)', text, flags=re.I)
    for i, part in enumerate(skill_section):
        if i > 0 and re.search(r'(compétences?|skills?|technologies?|outils?)', skill_section[i-1], re.I):
            items = re.split(r'[,;•\-\n]', part)
            for item in items:
                item = item.strip().lower()
                if 2 <= len(item) <= 50 and item not in found:
                    found.append(item)

    return list(set(found))


def extract_experience_years(text):
    """
    Extrait le nombre d'années d'expérience du CV.
    Stratégies :
    1. Recherche explicite : "5 ans d'expérience"
    2. Fallback : compte les postes mentionnés × 2 ans
    """
    match = re.search(r'(\d+)\+?\s*(ans|years|année|années)\s*(d\'|de\s)?(experience|expérience)', text, re.I)
    if match:
        return int(match.group(1))

    jobs = re.findall(r'(développeur|developer|analyste|manager|ingénieur|engineer|consultant|admin|chef|stagiare|intern|alternant)', text, re.I)
    return len(jobs) * 2


def extract_required_years(job_text):
    """
    Extrait le nombre d'années d'expérience requises dans l'offre.
    """
    match = re.search(r'(\d+)\+?\s*(ans|years|année|années)\s*(d\'|de\s)?(experience|expérience|minim|minimum)?', job_text, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)\+', job_text)
    if match:
        return int(match.group(1))
    return 0


def keyword_jaccard(text_cv, text_job):
    """
    Calcule le coefficient de Jaccard sur les mots-clés significatifs
    (hors stop-words) entre le CV et l'offre.

    Returns:
        float: Ratio intersection / union (0.0 à 1.0)
    """
    def get_keywords(text):
        words = re.findall(r'\b[a-zàâéèêëïîôùûüÿçæœ]{3,}\b', text.lower())
        return set(w for w in words if w not in STOP_WORDS)

    cv_kw = get_keywords(text_cv)
    job_kw = get_keywords(text_job)
    if not job_kw:
        return 0
    intersection = cv_kw & job_kw
    return len(intersection) / len(job_kw)


def _generate_recommendation(total_score, semantic, skill_match, exp_match, kw_match,
                               matched_skills, missing_skills, cv_exp, required_exp):
    """
    Génère un texte de recommandation humain à partir des scores.
    """
    lines = []
    if total_score >= 75:
        lines.append(f"Excellent profil ! Score de compatibilité de {total_score}%, très bonne adéquation avec le poste.")
    elif total_score >= 55:
        lines.append(f"Bon profil globalement compatible (score: {total_score}%). Quelques points à améliorer.")
    elif total_score >= 35:
        lines.append(f"Profil partiellement compatible (score: {total_score}%). Des compétences ou expérience manquantes.")
    else:
        lines.append(f"Profil peu compatible (score: {total_score}%). Écarts significatifs avec les exigences.")

    if matched_skills:
        skills_str = ", ".join(matched_skills[:8])
        if len(matched_skills) > 8:
            skills_str += f" et {len(matched_skills) - 8} autres"
        lines.append(f"Compétences correspondantes : {skills_str}.")
    else:
        lines.append("Aucune compétence technique requise retrouvée dans le CV.")

    if missing_skills:
        missing_str = ", ".join(missing_skills[:6])
        if len(missing_skills) > 6:
            missing_str += f" et {len(missing_skills) - 6} autres"
        lines.append(f"Compétences manquantes : {missing_str}.")

    if required_exp > 0:
        if cv_exp >= required_exp:
            lines.append(f"Expérience ({cv_exp} ans) conforme aux {required_exp} ans requis.")
        else:
            lines.append(f"Expérience ({cv_exp} ans) inférieure aux {required_exp} ans requis.")

    if semantic >= 60:
        lines.append(f"Contenu du CV en phase avec le poste (similarité: {semantic}%).")
    elif semantic >= 35:
        lines.append(f"Correspondance partielle avec le poste (similarité: {semantic}%).")
    else:
        lines.append(f"Vocabulaire éloigné du poste (similarité: {semantic}%). Adapter le CV.")

    if total_score >= 75:
        lines.append("Recommandation : Candidat à entrevoir. Profil correspond bien aux exigences.")
    elif total_score >= 55:
        lines.append("Recommandation : À envisager si les compétences manquantes sont acquises rapidement.")
    elif total_score >= 35:
        lines.append("Recommandation : À considérer si forte capacité d'apprentissage.")
    else:
        lines.append("Recommandation : Profil pas assez aligné. Privilégier d'autres candidats.")

    return " ".join(lines)


def compute_compatibility_score(cv_text, job_description, cv_embedding=None, job_embedding=None, model=None):
    """
    Calcule le score de compatibilité global entre un CV et une offre.

    Args:
        cv_text (str): Texte du CV
        job_description (str): Texte de l'offre
        cv_embedding (np.array, optional): Embedding pré-calculé du CV
        job_embedding (np.array, optional): Embedding pré-calculé de l'offre
        model (SentenceTransformer, optional): Modèle pour encoder à la volée

    Returns:
        dict: Scores détaillés + recommandation textuelle
    """
    W_SEMANTIC = 0.40
    W_SKILLS = 0.30
    W_EXPERIENCE = 0.20
    W_KEYWORDS = 0.10

    result = {}

    # -----------------------------------------------------------
    # 1. Similarité sémantique (embeddings)
    # -----------------------------------------------------------
    if cv_embedding is not None and job_embedding is not None:
        similarity = float(np.dot(cv_embedding, job_embedding) /
                          (np.linalg.norm(cv_embedding) * np.linalg.norm(job_embedding) + 1e-8))
        result["semantic_similarity"] = round(similarity * 100, 1)
    elif model is not None:
        embeddings = model.encode([cv_text, job_description])
        similarity = float(np.dot(embeddings[0], embeddings[1]) /
                          (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8))
        result["semantic_similarity"] = round(similarity * 100, 1)
    else:
        result["semantic_similarity"] = 0.0

    # -----------------------------------------------------------
    # 2. Match des compétences techniques
    # -----------------------------------------------------------
    cv_skills = extract_skills(cv_text)
    job_skills = extract_skills(job_description)
    if job_skills:
        matched = set(s.lower() for s in cv_skills) & set(s.lower() for s in job_skills)
        result["skill_match"] = round((len(matched) / len(job_skills)) * 100, 1)
        result["matched_skills"] = list(matched)
        result["missing_skills"] = list(set(s.lower() for s in job_skills) - set(s.lower() for s in cv_skills))
    else:
        result["skill_match"] = 50.0
        result["matched_skills"] = []
        result["missing_skills"] = []

    # -----------------------------------------------------------
    # 3. Match de l'expérience (années)
    # -----------------------------------------------------------
    cv_exp = extract_experience_years(cv_text)
    required_exp = extract_required_years(job_description)
    if required_exp > 0:
        if cv_exp >= required_exp:
            exp_ratio = min(cv_exp / required_exp, 1.5)
            result["experience_match"] = round(min(exp_ratio / 1.5 * 100, 100), 1)
        else:
            result["experience_match"] = round((cv_exp / required_exp) * 100, 1)
    else:
        result["experience_match"] = 50.0
    result["cv_experience_years"] = cv_exp
    result["required_experience_years"] = required_exp

    # -----------------------------------------------------------
    # 4. Correspondance des mots-clés (Jaccard)
    # -----------------------------------------------------------
    result["keyword_match"] = round(keyword_jaccard(cv_text, job_description) * 100, 1)

    # -----------------------------------------------------------
    # 5. Score final pondéré
    # -----------------------------------------------------------
    total = (W_SEMANTIC * result["semantic_similarity"] +
             W_SKILLS * result["skill_match"] +
             W_EXPERIENCE * result["experience_match"] +
             W_KEYWORDS * result["keyword_match"])

    result["total_score"] = round(total, 1)
    result["weights"] = {"semantic": W_SEMANTIC, "skills": W_SKILLS, "experience": W_EXPERIENCE, "keywords": W_KEYWORDS}
    result["recommandation"] = _generate_recommendation(
        result['total_score'], result['semantic_similarity'], result['skill_match'],
        result['experience_match'], result['keyword_match'], result['matched_skills'],
        result['missing_skills'], cv_exp, required_exp
    )
    result["explication"] = (
        f"Score de compatibilité: {result['total_score']}%. "
        f"Similarité sémantique: {result['semantic_similarity']}% (poids {int(W_SEMANTIC*100)}%), "
        f"Match compétences: {result['skill_match']}% (poids {int(W_SKILLS*100)}%), "
        f"Match expérience: {result['experience_match']}% (poids {int(W_EXPERIENCE*100)}%), "
        f"Mots-clés: {result['keyword_match']}% (poids {int(W_KEYWORDS*100)}%)."
    )
    return result
