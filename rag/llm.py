"""
rag/llm.py
==========
Communication avec le modèle de langage local (llama3.2:3b)
via l'API Ollama. Génère l'analyse finale augmentée par les
chunks récupérés (Retrieval-Augmented Generation).
"""

import re
import json
import requests
from config import OLLAMA_URL, logger
from config import OLLAMA_URL, OLLAMA_MODEL, logger


def check_ollama_available():
    """
    Vérifie si le serveur Ollama est accessible.

    Returns:
        bool: True si Ollama répond, False sinon
    """
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def generate_llm_analysis(relevant_chunks, job_description, cv_text,
                          ats_score, heuristic_score):
    """
    Appelle tinyllama:1.1b via Ollama pour générer une analyse structurée
    du CV par rapport à l'offre d'emploi.
    """
    # Construction du texte des chunks (tronqué pour performance)
    chunks_text = "\n---\n".join(
        f"Chunk {i+1} (similarite: {c['similarity']}):\n{c['chunk_text'][:300]}"
        for i, c in enumerate(relevant_chunks)
    )[:1200]

    # Prompt texte simple (mieux adapte a tinyllama 1.1B)
    prompt = f"""Analyse ce CV pour le poste.

POSTE: {job_description[:800]}

CV: {cv_text[:800]}

EXTRAITS PERTINENTS:
{chunks_text[:1000]}

Scores: ATS={ats_score}, Heuristique={heuristic_score}

Reponds UNIQUEMENT en JSON valide avec ces champs:
- "points_forts": ["point1", "point2", "point3"]
- "faiblesses": ["faible1", "faible2"]
- "compatibilite": 75
- "analyse_detaillee": "Paragraphe d'analyse en francais, 3-5 phrases"
- "recommandations": ["reco1", "reco2"]

JSON:"""

    try:
        logger.info(f"🤖 Appel Ollama: model=tinyllama:1.1b, url={OLLAMA_URL}/api/generate")

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 512
                }
            },
            timeout=300
        )

        if response.status_code == 200:
            result = response.json()
            llm_text = result.get("response", "")
            logger.info(f"✅ Ollama réponse reçue ({len(llm_text)} chars)")

            # Extraire le JSON de la réponse (chercher le dernier bloc JSON)
            try:
                # Chercher tous les blocs JSON et prendre le dernier
                json_matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_text))
                if json_matches:
                    for match in reversed(json_matches):
                        try:
                            parsed = json.loads(match.group())
                            # Vérifier que c'est notre JSON (pas le prompt)
                            if "analyse_detaillee" in parsed or "compatibilite" in parsed:
                                return parsed
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass

            # Fallback : retourne le texte brut
            return {
                "analyse_detaillee": llm_text[:500] if llm_text else "Analyse non disponible",
                "points_forts": [],
                "faiblesses": [],
                "recommandations": [],
                "compatibilite": None,
                "raw": True
            }
        else:
            logger.error(f"❌ Ollama erreur HTTP: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.Timeout:
        logger.error("❌ Ollama timeout (300s)")
        return None
    except Exception as e:
        logger.error(f"❌ Erreur LLM: {e}")
        return None