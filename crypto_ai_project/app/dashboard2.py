# app/dashboard.py

from pathlib import Path
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np

from modules.config import (
    MARKET_DATA_CSV,
    MARKET_INTRADAY_1M_CSV,
    SENTIMENT_DATA_CSV,
    NEWS_DATA_CSV,
    BASE_DIR,
)
from LLM.news_adjuster import build_adjusted_forecast
from LLM.chatbot import crypto_chat

app = Flask(__name__)

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
                "time": row["timestamp"].isoformat(),
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
                "time": row["timestamp"].isoformat(),
                "price": float(row["close"]),
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
            "latest": {"news_sentiment": None, "fear_greed": None},
        }

    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df.empty:
        return {
            "timestamps": [],
            "news_sentiment": [],
            "fear_greed": [],
            "latest": {"news_sentiment": None, "fear_greed": None},
        }

    df = df.sort_values("timestamp")
    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    df = df[df["timestamp"] >= cutoff]

    timestamps = [ts.isoformat() for ts in df["timestamp"]]

    news_sent = df.get("news_sentiment")
    fear_greed = df.get("fear_greed")

    news_sent_list = (
        [float(x) if pd.notna(x) else None for x in news_sent]
        if news_sent is not None
        else []
    )
    fear_greed_list = (
        [int(x) if pd.notna(x) else None for x in fear_greed]
        if fear_greed is not None
        else []
    )

    latest_news = news_sent_list[-1] if news_sent_list else None
    latest_fg = fear_greed_list[-1] if fear_greed_list else None

    return {
        "timestamps": timestamps,
        "news_sentiment": news_sent_list,
        "fear_greed": fear_greed_list,
        "latest": {
            "news_sentiment": latest_news,
            "fear_greed": latest_fg,
        },
    }


def load_daily_news(limit: int = 25) -> list[dict]:
    """Utolsó ~30 napos hírek a NEWS_DATA_CSV-ből (runtime).

    Vissza: [{timestamp, source, title, summary, url}]
    """
    path = Path(NEWS_DATA_CSV)
    if not path.exists():
        return []

    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    if df.empty:
        return []

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=False)
    else:
        df = df.tail(limit)

    out = []
    for _, row in df.head(limit).iterrows():
        ts = row.get("timestamp")
        out.append(
            {
                "timestamp": ts.isoformat() if pd.notna(ts) else None,
                "source": str(row.get("source") or ""),
                "title": str(row.get("title") or ""),
                "summary": str(row.get("summary") or ""),
                "url": str(row.get("url") or ""),
            }
        )
    return out


# ---------- Flask route-ok ----------

@app.route("/")
def index():
    """
    HTML dashboard (Chart.js grafikonokkal).
    Az adatok JS-ben az /api/state végpontról jönnek.
    """
    return render_template("dashboard.html")


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

    # 2b) Napi hírek
    news = load_daily_news(limit=25)

    # 3) Hosszútávú log-görbe
    long_curve = load_longterm_curve()

    # 4) Modell előrejelzés és tanács
    try:
        from modules.advisor import generate_advice

        advice = generate_advice()
    except Exception as e:
        advice = {
            "signal": "ERROR",
            "error": str(e),
            "last_close": None,
            "next_price_pred": None,
            "rel_change_pred": None,
            "fear_greed": sentiment["latest"].get("fear_greed"),
            "news_sentiment": sentiment["latest"].get("news_sentiment"),
        }

    payload = {
        "candles_1h": candles_1h,
        "intraday_1m": intraday_1m,
        "sentiment": sentiment,
        "news": news,
        "long_curve": long_curve,  # <--- ÚJ
        "advice": advice,
    }

    return jsonify(payload)


@app.route("/api/llm/adjusted_forecast", methods=["GET"])
def api_adjusted_forecast():
    """
    LLM-alapú hír-összefoglalóval és sentimenttel módosított előrejelzés.
    Base: generate_advice(), plusz heuristika vagy LLM ha van kulcs.
    """
    try:
        from modules.advisor import generate_advice

        base = generate_advice()
        sentiment = load_sentiment_series(days=90)
        long_curve = load_longterm_curve()
        force_llm = (request.args.get("force_llm") or "").strip() in ("1", "true", "True")
        adjusted = build_adjusted_forecast(base, sentiment, long_curve, force_llm=force_llm)
        return jsonify({"base": base, "adjusted": adjusted})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.route("/api/llm/chat", methods=["POST"])
def api_llm_chat():
    """
    "CryptoGPT" jellegű asszisztens endpoint. Kérdés a body-ban: {"question": "..."}
    """
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "").strip()
    allow_llm = bool(payload.get("allow_llm") is True or str(payload.get("allow_llm") or "").strip() in ("1", "true", "True"))
    if not question:
        return jsonify({"error": "question is required"}), 400
    try:
        answer = crypto_chat(question, allow_llm=allow_llm)
        return jsonify(answer)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

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
