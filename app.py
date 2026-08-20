import os
import re
import json
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# Configuration
# ============================================================
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration BDD pgvector
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": os.environ.get("DB_PORT", "5432"),
    "database": os.environ.get("DB_NAME", "ats_db"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "postgres"),
}

RAG_SERVICE_URL = os.environ.get("RAG_PYTHON_URL", "http://localhost:5000")

# ============================================================
# Chargement des modeles (lazy loading)
# ============================================================
ats_model = None
ats_tokenizer = None
embedding_model = None

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "ats_bert_model")

def load_ats_model():
    """Charge le modele BERT fine-tune pour classification ATS"""
    global ats_model, ats_tokenizer
    if ats_model is not None:
        return
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_path = MODEL_DIR
        if os.path.exists(model_path):
            ats_tokenizer = AutoTokenizer.from_pretrained(model_path)
            ats_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            ats_model.eval()
            logger.info(f"Modele ATS charge depuis {model_path}")
        else:
            logger.warning(f"Modele ATS non trouve dans {model_path} - mode heuristique")
    except Exception as e:
        logger.error(f"Erreur chargement modele ATS: {e}")

def load_embedding_model():
    """Charge le modele d'embeddings sentence-transformers"""
    global embedding_model
    if embedding_model is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        logger.info("Modele embeddings charge: paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:
        logger.error(f"Erreur chargement modele embeddings: {e}")

def get_db_connection():
    """Connexion a PostgreSQL avec pgvector"""
    import psycopg2
    return psycopg2.connect(**DB_CONFIG)


# ============================================================
# REGLES ATS HEURISTIQUES (VERSION AMELIOREE)
# ============================================================
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

# Noms en francais pour l'affichage
RULE_NAMES = {
    "email": "Email de contact",
    "phone": "Telephone de contact",
    "experience": "Section Experience",
    "education": "Section Education/Formation",
    "skills": "Section Competences",
    "length": "Longueur appropriee (200-5000 mots)",
    "dates": "Dates presentes",
    "keywords": "Mots-cles techniques",
    "objective": "Objectif/Profil/Resume",
    "clean": "Format propre (pas de caracteres speciaux)",
    "job_titles": "Titres de postes mentionnes",
    "quantified": "Realisations quantifiees (chiffres)",
    "acronyms": "Acronymes expliques",
    "reverse_chrono": "Ordre chronologique inverse",
}

def ats_score(text):
    """Calcule le score ATS heuristique (0-100) avec detail"""
    total = sum(r["weight"] for r in ATS_RULES.values())
    earned = 0
    details = []

    for key, rule in ATS_RULES.items():
        matched = False
        points = 0

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
                    if all(int(unique_years[i]) >= int(unique_years[i+1]) for i in range(len(unique_years)-1)):
                        matched = True
                        points = rule["weight"]
                    else:
                        matched = False
                        points = rule["weight"] * 0.5
                else:
                    matched = False
                    points = rule["weight"] * 0.5
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


# ============================================================
# SCORE DE COMPATIBILITE CV-JOB
# ============================================================

TECH_KEYWORDS = {
    "langages": ["java", "python", "javascript", "typescript", "c#", "c++", "go", "rust", "php", "ruby", "swift", "kotlin"],
    "frameworks": ["spring", "react", "angular", "vue", "django", "flask", "express", "next.js", "node.js", ".net", "laravel"],
    "bases_donnees": ["sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle", "sqlite", "cassandra"],
    "devops": ["docker", "kubernetes", "jenkins", "git", "ci/cd", "terraform", "ansible", "aws", "azure", "gcp"],
    "data_ia": ["machine learning", "deep learning", "nlp", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "keras", "spark"],
    "outils": ["linux", "git", "maven", "gradle", "jira", "agile", "scrum", "rest", "api", "microservices"],
}

def extract_skills(text):
    """Extrait les competences techniques d'un texte"""
    text_lower = text.lower()
    found = []
    for category, keywords in TECH_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(kw)
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
    """Extrait les annees d'experience mentionnees"""
    match = re.search(r'(\d+)\+?\s*(ans|years|année|années)\s*(d\'|de\s)?(experience|expérience)', text, re.I)
    if match:
        return int(match.group(1))
    jobs = re.findall(r'(développeur|developer|analyste|manager|ingénieur|engineer|consultant|admin|chef|stagiare|intern|alternant)', text, re.I)
    return len(jobs) * 2

def extract_required_years(job_text):
    """Extrait les annees d'experience requises dans l'offre"""
    match = re.search(r'(\d+)\+?\s*(ans|years|année|années)\s*(d\'|de\s)?(experience|expérience|minim|minimum)?', job_text, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)\+', job_text)
    if match:
        return int(match.group(1))
    return 0

def keyword_jaccard(text_cv, text_job):
    """Calcule le Jaccard index sur les mots-cles importants"""
    stop_words = {'le', 'la', 'les', 'de', 'des', 'du', 'et', 'en', 'un', 'une', 'est', 'que', 'qui', 'dans',
                  'pour', 'par', 'sur', 'avec', 'ce', 'cette', 'son', 'sa', 'ses', 'au', 'aux', 'the', 'a', 'an',
                  'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'and', 'or', 'this', 'that'}

    def get_keywords(text):
        words = re.findall(r'\b[a-zàâéèêëïîôùûüÿçæœ]{3,}\b', text.lower())
        return set(w for w in words if w not in stop_words)

    cv_kw = get_keywords(text_cv)
    job_kw = get_keywords(text_job)

    if not job_kw:
        return 0
    intersection = cv_kw & job_kw
    return len(intersection) / len(job_kw)

def compute_compatibility_score(cv_text, job_description, cv_embedding=None, job_embedding=None, model=None):
    """Calcule le score de compatibilite CV-Job (0-100)"""
    W_SEMANTIC = 0.40
    W_SKILLS = 0.30
    W_EXPERIENCE = 0.20
    W_KEYWORDS = 0.10

    result = {}

    # 1. Similarite semantique (40%)
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

    # 2. Match competences (30%)
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

    # 3. Match experience (20%)
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

    # 4. Mots-cles Jaccard (10%)
    result["keyword_match"] = round(keyword_jaccard(cv_text, job_description) * 100, 1)

    # ---- SCORE TOTAL ----
    total = (W_SEMANTIC * result["semantic_similarity"] +
             W_SKILLS * result["skill_match"] +
             W_EXPERIENCE * result["experience_match"] +
             W_KEYWORDS * result["keyword_match"])

    result["total_score"] = round(total, 1)
    result["weights"] = {
        "semantic": W_SEMANTIC,
        "skills": W_SKILLS,
        "experience": W_EXPERIENCE,
        "keywords": W_KEYWORDS
    }

    # Recommandation lisible en francais
    result["recommandation"] = _generate_recommendation(
        result['total_score'],
        result['semantic_similarity'],
        result['skill_match'],
        result['experience_match'],
        result['keyword_match'],
        result['matched_skills'],
        result['missing_skills'],
        cv_exp,
        required_exp
    )

    # Garder les metriques brutes pour debug (champ separe)
    result["explication"] = (
        f"Score de compatibilite: {result['total_score']}%. "
        f"Similarite semantique: {result['semantic_similarity']}% (poids {int(W_SEMANTIC*100)}%), "
        f"Match competences: {result['skill_match']}% (poids {int(W_SKILLS*100)}%, {len(result['matched_skills'])}/{len(job_skills)} competences trouvees), "
        f"Match experience: {result['experience_match']}% (poids {int(W_EXPERIENCE*100)}%, candidat: {cv_exp} ans, requis: {required_exp}), "
        f"Mots-cles: {result['keyword_match']}% (poids {int(W_KEYWORDS*100)}%)."
    )

    return result


def _generate_recommendation(total_score, semantic, skill_match, exp_match, kw_match,
                               matched_skills, missing_skills, cv_exp, required_exp):
    """Genere une recommandation humainement lisible en francais."""
    lines = []

    # Appreciation globale
    if total_score >= 75:
        lines.append(f"Excellent profil ! Le candidat obtient un score de compatibilite de {total_score}%, ce qui indique une tres bonne adequation avec le poste.")
    elif total_score >= 55:
        lines.append(f"Bon profil globalement compatible avec le poste, avec un score de {total_score}%. Quelques points d'amelioration sont possibles.")
    elif total_score >= 35:
        lines.append(f"Profil partiellement compatible (score: {total_score}%). Des competences ou de l'experience manquantes limitent l'adequation.")
    else:
        lines.append(f"Profil peu compatible avec le poste (score: {total_score}%). Des ecarts significatifs existent par rapport aux exigences.")

    # Competences trouvees
    if matched_skills:
        skills_str = ", ".join(matched_skills[:8])
        if len(matched_skills) > 8:
            skills_str += f" et {len(matched_skills) - 8} autres"
        lines.append(f"Competences correspondantes : {skills_str}.")
    else:
        lines.append("Aucune competence technique requise n'a ete retrouvee dans le CV.")

    # Competences manquantes
    if missing_skills:
        missing_str = ", ".join(missing_skills[:6])
        if len(missing_skills) > 6:
            missing_str += f" et {len(missing_skills) - 6} autres"
        lines.append(f"Competences manquantes a developper : {missing_str}.")

    # Experience
    if required_exp > 0:
        if cv_exp >= required_exp:
            lines.append(f"L'experience du candidat ({cv_exp} ans) est conforme ou superieure aux {required_exp} ans requis.")
        else:
            lines.append(f"L'experience du candidat ({cv_exp} ans) est inferieure aux {required_exp} ans requis pour ce poste.")

    # Semantique
    if semantic >= 60:
        lines.append(f"Le contenu du CV est globalement en phase avec la description du poste (similarite semantique : {semantic}%).")
    elif semantic >= 35:
        lines.append(f"Le contenu du CV presente une correspondance partielle avec le poste (similarite semantique : {semantic}%).")
    else:
        lines.append(f"Le vocabulaire et le contenu du CV sont eloignes de la description du poste (similarite semantique : {semantic}%). Le candidat devrait adapter son CV.")

    # Conseil final
    if total_score >= 75:
        lines.append("Recommandation : Ce candidat merit une entrevue approfondie. Son profil correspond bien aux exigences du poste.")
    elif total_score >= 55:
        lines.append("Recommandation : Candidat a envisager pour un entretien, en evaluant si les competences manquantes peuvent etre acquises rapidement en poste.")
    elif total_score >= 35:
        lines.append("Recommandation : A considerer si le candidat montre une forte capacite d'apprentissage ou si le poste offre une formation.")
    else:
        lines.append("Recommandation : Profil pas assez aligne avec le poste. D'autres candidats plus adaptes devraient etre privilgies.")

    return " ".join(lines)


# ============================================================
# ENDPOINTS
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "ats-rag-service"})


@app.route('/api/rag/analyze', methods=['POST'])
def analyze_cv():
    """
    Analyse ATS d'un CV avec detail complet.
    - Score ATS (BERT si dispo, sinon heuristique)
    - Points forts et points a ameliorer
    - Detail du calcul par regle
    - Explication du score
    - Similarite semantique si job_description fournie
    """
    data = request.get_json()
    cv_text = data.get('cv_text', '')
    job_description = data.get('job_description', None)

    if not cv_text:
        return jsonify({"error": "cv_text est requis"}), 400

    # --- Detail heuristique (toujours calcule) ---
    heuristic_result = ats_score(cv_text)

    # --- Score ATS (BERT si dispo, sinon heuristique) ---
    try:
        load_ats_model()
        if ats_model is not None:
            import torch
            inputs = ats_tokenizer(cv_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = ats_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                ats_prob = probs[0][1].item()
            ats_score_value = round(ats_prob * 100, 1)
            method = "bert"
        else:
            ats_score_value = heuristic_result["score"]
            method = "heuristic"
    except Exception as e:
        logger.error(f"Erreur analyse ATS: {e}")
        ats_score_value = heuristic_result["score"]
        method = "heuristic_fallback"

    result = {
        "ats_score": ats_score_value,
        "ats_method": method,
        "is_ats_friendly": ats_score_value >= 55,
        "heuristic_score": heuristic_result["score"],
        "points_forts": heuristic_result["points_forts"],
        "a_ameliorer": heuristic_result["a_ameliorer"],
        "score_details": heuristic_result["details"],
    }

    # --- Explication du calcul ---
    nb_matched = len(result["points_forts"])
    nb_total = len(heuristic_result["details"])
    points_earned = heuristic_result["points_earned"]
    points_total = heuristic_result["points_total"]

    if method == "bert":
        result["explication"] = (
            f"Score BERT: {ats_score_value}% (classification par modele fine-tune). "
            f"Score heuristique: {heuristic_result['score']}%. "
            f"{nb_matched}/{nb_total} criteres valides ({points_earned}/{points_total} points). "
            f"Points manquants: {', '.join(result['a_ameliorer']) if result['a_ameliorer'] else 'aucun'}."
        )
    else:
        result["explication"] = (
            f"Score heuristique: {ats_score_value}%. "
            f"{nb_matched}/{nb_total} criteres valides ({points_earned}/{points_total} points). "
            f"Points manquants: {', '.join(result['a_ameliorer']) if result['a_ameliorer'] else 'aucun'}."
        )

    # --- Similarite cosine si job_description fournie ---
    if job_description:
        try:
            load_embedding_model()
            if embedding_model is not None:
                emb_cv = embedding_model.encode(cv_text)
                emb_job = embedding_model.encode(job_description)
                similarity = float(np.dot(emb_cv, emb_job) /
                                  (np.linalg.norm(emb_cv) * np.linalg.norm(emb_job) + 1e-8))
                result["semantic_similarity"] = round(similarity * 100, 1)
        except Exception as e:
            logger.error(f"Erreur similarite: {e}")

    return jsonify(result)


@app.route('/api/rag/embed', methods=['POST'])
def embed_cv():
    """Genere l'embedding d'un CV et le stocke dans pgvector"""
    data = request.get_json()
    cv_text = data.get('cv_text', '')
    candidat_id = data.get('candidat_id')

    if not cv_text or not candidat_id:
        return jsonify({"error": "cv_text et candidat_id sont requis"}), 400

    try:
        load_embedding_model()
        if embedding_model is None:
            return jsonify({"error": "Modele embeddings non disponible"}), 503

        embedding = embedding_model.encode(cv_text).tolist()

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO cv_embeddings (candidat_id, embedding, cv_text)
            VALUES (%s, %s, %s)
            ON CONFLICT (candidat_id)
            DO UPDATE SET embedding = EXCLUDED.embedding, cv_text = EXCLUDED.cv_text
        """, (candidat_id, str(embedding), cv_text))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"message": "Embedding stocke avec succes", "candidat_id": candidat_id})

    except Exception as e:
        logger.error(f"Erreur embed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/rag/search', methods=['POST'])
def semantic_search():
    """Recherche semantique de CV par rapport a une requete"""
    data = request.get_json()
    query = data.get('query', '')
    top_k = data.get('top_k', 10)

    if not query:
        return jsonify({"error": "query est requise"}), 400

    try:
        load_embedding_model()
        if embedding_model is None:
            return jsonify({"error": "Modele embeddings non disponible"}), 503

        query_embedding = embedding_model.encode(query).tolist()

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT candidat_id, cv_text,
                   1 - (embedding <#> %s::vector) as similarity
            FROM cv_embeddings
            ORDER BY embedding <#> %s::vector
            LIMIT %s
        """, (str(query_embedding), str(query_embedding), top_k))

        results = []
        for row in cur.fetchall():
            results.append({
                "candidat_id": row[0],
                "cv_text": row[1],
                "similarity": round(float(row[2]), 4)
            })

        cur.close()
        conn.close()

        return jsonify({"results": results, "query": query, "total": len(results)})

    except Exception as e:
        logger.error(f"Erreur recherche: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/rag/extract', methods=['POST'])
def extract_info():
    """Extrait les informations cles d'un CV (regex)"""
    data = request.get_json()
    cv_text = data.get('cv_text', '')

    if not cv_text:
        return jsonify({"error": "cv_text est requis"}), 400

    email = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', cv_text)
    phone = re.search(r'\+?\d[\d\s\-]{8,}\d', cv_text)
    name_match = re.search(r'^([A-ZÀÂÉÈÊËÏÎÔÙÛÜŸÇÆŒ][a-zàâéèêëïîôùûüÿçæœ]+(?:\s+[A-ZÀÂÉÈÊËÏÎÔÙÛÜŸÇÆŒ][a-zàâéèêëïîôùûüÿçæœ]+)+)', cv_text, re.M)

    skills = extract_skills(cv_text)
    exp_years = extract_experience_years(cv_text)

    info = {
        "email": email.group(0) if email else None,
        "phone": phone.group(0) if phone else None,
        "name": name_match.group(0).strip() if name_match else None,
        "skills": skills,
        "experience_years": exp_years,
    }

    return jsonify(info)


@app.route('/api/rag/compatibility', methods=['POST'])
def compatibility_score():
    """
    Calcule le score de compatibilite entre un CV et une offre d'emploi.
    Score combine: 40% semantique + 30% competences + 20% experience + 10% mots-cles
    """
    data = request.get_json()
    cv_text = data.get('cv_text', '')
    job_description = data.get('job_description', '')

    if not cv_text or not job_description:
        return jsonify({"error": "cv_text et job_description sont requis"}), 400

    try:
        load_embedding_model()
        cv_emb = None
        job_emb = None
        if embedding_model is not None:
            cv_emb = embedding_model.encode(cv_text)
            job_emb = embedding_model.encode(job_description)

        result = compute_compatibility_score(
            cv_text=cv_text,
            job_description=job_description,
            cv_embedding=cv_emb,
            job_embedding=job_emb,
            model=embedding_model
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Erreur compatibilite: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    logger.info("Demarrage du service RAG ATS...")
    app.run(host='0.0.0.0', port=5000, debug=True)
