# app/__init__.py
from flask import Flask


def create_app():
    app = Flask(__name__)

    from .dashboard import bp as dashboard_bp
    app.register_blueprint(dashboard_bp)

    return app
