"""
api/routes.py
=============
Regroupe tous les blueprints Flask et les enregistre sur l'app principale.
C'est le seul endroit où les routes sont connectées à l'application.
"""

from api.health import health_bp
from api.analyze import analyze_bp
from api.embed import embed_bp
from api.search import search_bp
from api.extract import extract_bp
from api.compatibility import compatibility_bp


def register_blueprints(app):
    """
    Enregistre tous les blueprints sur l'application Flask.

    Args:
        app (Flask): Instance de l'application Flask
    """
    app.register_blueprint(health_bp)
    app.register_blueprint(analyze_bp)
    app.register_blueprint(embed_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(extract_bp)
    app.register_blueprint(compatibility_bp)
