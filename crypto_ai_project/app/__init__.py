# app/__init__.py
def create_app():
    # A projektben a dashboard a `dashboard2.py` modulban van egyben (Flask app instance-szel).
    # Itt csak visszaadjuk azt, hogy `python -m app.run_dashboard` működjön.
    from .dashboard2 import app as dashboard_app

    return dashboard_app
