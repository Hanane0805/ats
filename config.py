"""
config.py
=========
Configuration centralisée du service ATS-RAG.
Tous les paramètres sensibles (DB, URLs) sont lus depuis les variables
 d'environnement avec des valeurs par défaut pour le développement local.
"""

import os
import logging

# -----------------------------------------------------------
# Configuration du logging
# -----------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# Configuration PostgreSQL + pgvector
# -----------------------------------------------------------
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": os.environ.get("DB_PORT", "5433"),      # Port PostgreSQL (ici 5433)
    "database": os.environ.get("DB_NAME", "ats_db"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "postgres"),
}

# -----------------------------------------------------------
# Configuration Ollama (LLM local)
# -----------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
# -----------------------------------------------------------
# Chemins des modèles locaux
# -----------------------------------------------------------
MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "models", "ats_bert_model"
)
