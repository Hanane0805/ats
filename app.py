"""
app.py
======
Point d'entrée du AI Service (Flask - Port 5000).
Appelé par le backend Spring Boot (port 8080) via :
    flask.ai.service.url=http://localhost:5000

Architecture modulaire :
- core/       : Modèles ML + Base de données
- rag/        : Pipeline RAG (chunking, storage, retrieval, LLM)
- scoring/    : Algorithmes de scoring (heuristique, BERT, compatibilité)
- extraction/ : Extraction d'informations depuis les CVs
- api/        : Endpoints Flask (Blueprints)
"""

from flask import Flask
from flask_cors import CORS
from config import logger
from api.routes import register_blueprints
from core.database import init_pgvector_tables

app = Flask(__name__)
CORS(app)
register_blueprints(app)

if __name__ == '__main__':
    logger.info("🚀 Démarrage du service ATS RAG (Port 5000)...")
    try:
        init_pgvector_tables()
    except Exception as e:
        logger.warning(f"⚠️ Tables pgvector non initialisées: {e}")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
