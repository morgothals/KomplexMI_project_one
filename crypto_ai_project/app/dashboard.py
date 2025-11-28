# app/dashboard.py
from flask import Blueprint, render_template, jsonify
import pandas as pd

from modules.config import MARKET_DATA_CSV, SENTIMENT_DATA_CSV
from modules.advisor import generate_advice

bp = Blueprint("dashboard", __name__)


@bp.route("/")
def index():
    # Csak a HTML-t adja, az adatokat AJAX-szal töltjük
    return render_template("dashboard.html")


@bp.route("/api/ohlcv")
def api_ohlcv():
    """
    Utolsó N gyertya OHLCV adatai a gyertya grafikonnak.
    """
    N = 200  # mennyi gyertyát mutassunk
    df = pd.read_csv(MARKET_DATA_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df = df.sort_index().tail(N)

    data = {
        "timestamp": df.index.astype(str).tolist(),
        "open": df["open"].tolist(),
        "high": df["high"].tolist(),
        "low": df["low"].tolist(),
        "close": df["close"].tolist(),
        "volume": df["volume"].tolist(),
    }
    return jsonify(data)


@bp.route("/api/fear_greed")
def api_fear_greed():
    """
    Fear & Greed + hírsentiment az utolsó ~60 napra.
    """
    try:
        df = pd.read_csv(SENTIMENT_DATA_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    except FileNotFoundError:
        return jsonify({"timestamp": [], "fear_greed": [], "news_sentiment": []})

    df = df.sort_index().last("60D")

    data = {
        "timestamp": df.index.astype(str).tolist(),
        "fear_greed": df["fear_greed"].fillna(0).astype(float).tolist()
        if "fear_greed" in df.columns else [],
        "news_sentiment": df["news_sentiment"].fillna(0).astype(float).tolist()
        if "news_sentiment" in df.columns else [],
    }
    return jsonify(data)


@bp.route("/api/advice")
def api_advice():
    """
    Modell előrejelzés + jelzés.
    """
    try:
        advice = generate_advice()
    except Exception as e:
        # Ha valami gond van (pl. nincs modell még)
        return jsonify({"error": str(e)}), 500

    return jsonify(advice)
