# run_dashboard.py
from app import create_app

app = create_app()

if __name__ == "__main__":
    # fejlesztéshez:
    app.run(debug=True)
    # élesben inkább: app.run(host="0.0.0.0", port=8000)
