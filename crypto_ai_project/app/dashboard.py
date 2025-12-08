# app/dashboard.py

from pathlib import Path
from flask import Flask, render_template, jsonify
from flask_cors import CORS  # <--- Hozzáadva: UI kommunikációhoz
import pandas as pd
import numpy as np

from modules.config import (
    MARKET_DATA_CSV,
    MARKET_INTRADAY_1M_CSV,
    SENTIMENT_DATA_CSV,
    BASE_DIR,
)
from modules.advisor import generate_advice

app = Flask(__name__)
CORS(app)  # <--- Hozzáadva: Engedélyezi a böngészőnek az adatlekérést a React portról

# ---------- Segédfüggvények adatokhoz ----------

def load_ohlcv_1h(limit: int = 200) -> list[dict]:
    """
    Utolsó 'limit' darab 1H OHLCV gyertya a MARKET_DATA_CSV-ből.
    Vissza: list[ dict(time, open, high, low, close, volume) ]
    """
    path = Path(MARKET_DATA_CSV)
    if not path.exists():
        return []

    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df.empty:
        return []

    df = df.sort_values("timestamp").tail(limit)

    candles = []
    for _, row in df.iterrows():
        candles.append(
            {
                # Javítás: A React 'timestamp'-et vár a dátumhoz
                "timestamp": row["timestamp"].isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
            }
        )
    return candles


def load_longterm_curve() -> dict:
    """
    Log-regressziós hosszútávú BTC görbe betöltése a predictions/btc_log_curve_prediction.csv-ből.
    Vissza:
      {
        "labels": [...],             # Évek stringként
        "pred_price": [...],
        "pred_price_low": [...],
        "pred_price_high": [...]
      }
    """
    path = Path(BASE_DIR) / "predictions" / "btc_log_curve_prediction.csv"
    if not path.exists():
        return {
            "labels": [],
            "pred_price": [],
            "pred_price_low": [],
            "pred_price_high": [],
        }

    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df.empty:
        return {
            "labels": [],
            "pred_price": [],
            "pred_price_low": [],
            "pred_price_high": [],
        }

    df = df.sort_values("timestamp")

    # X tengelyen csak az ÉV-et akarjuk kiírni
    labels = [str(ts.year) for ts in df["timestamp"]]

    return {
        "labels": labels,
        "pred_price": [float(x) for x in df["pred_price"]],
        "pred_price_low": [float(x) for x in df["pred_price_low"]],
        "pred_price_high": [float(x) for x in df["pred_price_high"]],
    }


def load_intraday_1m(limit: int = 300) -> list[dict]:
    """
    Aznapi 1 perces OHLCV (ha létezik MARKET_INTRADAY_1M_CSV).
    Jó egy kis zoom-os intraday chartra.
    """
    path = Path(MARKET_INTRADAY_1M_CSV)
    if not path.exists():
        return []

    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df.empty:
        return []

    df = df.sort_values("timestamp").tail(limit)

    points = []
    for _, row in df.iterrows():
        points.append(
            {
                # Javítás: Reacthoz egységes kulcsnevek
                "timestamp": row["timestamp"].isoformat(),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
            }
        )
    return points


def load_sentiment_series(days: int = 60) -> dict:
    """
    Sentiment rövid idősor a SENTIMENT_DATA_CSV-ből.
    Visszaad:
      {
        "timestamps": [...],
        "news_sentiment": [...],
        "fear_greed": [...],
        "latest": {"news_sentiment": float|None, "fear_greed": int|None}
      }
    """
    path = Path(SENTIMENT_DATA_CSV)
    if not path.exists():
        return {
            "timestamps": [],
            "news_sentiment": [],
            "fear_greed": [],
            "latest": {"news_sentiment": 0, "fear_greed": 50},
        }

    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df.empty:
        return {
            "timestamps": [],
            "news_sentiment": [],
            "fear_greed": [],
            "latest": {"news_sentiment": 0, "fear_greed": 50},
        }

    df = df.sort_values("timestamp")
    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    df = df[df["timestamp"] >= cutoff]

    timestamps = [ts.isoformat() for ts in df["timestamp"]]

    # Javítás: NaN értékek kezelése (JSON nem szereti a NaN-t)
    news_sent = df["news_sentiment"].fillna(0).tolist() if "news_sentiment" in df.columns else []
    fear_greed = df["fear_greed"].fillna(50).tolist() if "fear_greed" in df.columns else []

    latest_news = news_sent[-1] if news_sent else 0
    latest_fg = int(fear_greed[-1]) if fear_greed else 50

    return {
        "timestamps": timestamps,
        "news_sentiment": news_sent,
        "fear_greed": fear_greed,
        "latest": {
            "news_sentiment": latest_news,
            "fear_greed": latest_fg,
        },
    }


# ---------- Flask route-ok ----------

@app.route("/")
def index():
    """
    HTML dashboard (Chart.js grafikonokkal).
    Az adatok JS-ben az /api/state végpontról jönnek.
    """
    # A React frontend külön fut, ez csak fallback üzenet
    return "Crypto AI Backend Running. Please use React Frontend on port 5173."


@app.route("/api/state")
def api_state():
    """
    Frontend által használt JSON:
      - 1H OHLCV gyertyák
      - intraday 1m ár
      - sentiment idősor + aktuális értékek
      - modell előrejelzés + BUY / HOLD / SELL
    """
    # 1) OHLCV
    candles_1h = load_ohlcv_1h(limit=200)
    intraday_1m = load_intraday_1m(limit=300)

    # 2) Sentiment
    sentiment = load_sentiment_series(days=60)

    # 3) Hosszútávú log-görbe
    long_curve = load_longterm_curve()

    # 4) Modell előrejelzés és tanács
    try:
        raw_advice = generate_advice()
        # Átalakítás a frontend által várt struktúrára
        advice = {
            "signal": raw_advice.get("signal", "HOLD"),
            "predicted_price": raw_advice.get("next_price_pred"),
            "direction": "Bullish" if raw_advice.get("rel_change_pred", 0) > 0 else "Bearish",
            "sentiment_score": sentiment["latest"]["news_sentiment"]
        }
    except Exception as e:
        print(f"Advisor Error: {e}")
        advice = {
            "signal": "ERROR",
            "error": str(e),
            "predicted_price": None,
            "direction": "Neutral",
            "sentiment_score": 0,
        }

    payload = {
        "candles_1h": candles_1h,
        "intraday_1m": intraday_1m,
        "sentiment": sentiment,
        "long_curve": long_curve,
        "advice": advice,
    }

    return jsonify(payload)

def create_app():
    """
    Ha később gunicorn/uwsgi vagy más WSGI server alá akarnád rakni:
    from app.dashboard import create_app
    app = create_app()
    """
    return app


if __name__ == "__main__":
    # Lokális futtatáshoz:
    #   (venv) python -m app.dashboard
    app.run(host="0.0.0.0", port=5000, debug=True)