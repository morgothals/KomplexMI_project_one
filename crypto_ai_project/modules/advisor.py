# modules/advisor.py
import numpy as np

from .forecast_model import predict_next_close


def generate_advice() -> dict:
    """
    - Betölti a training_features_1h.csv legfrissebb sorát
    - Meghívja a predict_next_close()-t
    - Egyszerű szabályalapú BUY / HOLD / SELL jelzést ad
    """
    next_price, last_close, last_row = predict_next_close()

    rel_change = (next_price - last_close) / last_close if last_close != 0 else 0.0

    fear_greed = float(last_row.get("fear_greed", np.nan)) if "fear_greed" in last_row.index else np.nan

    # sentiment: napi átlag, amit a training_sentiment_features-ből jön
    if "news_sentiment_mean" in last_row.index:
        news_sent = float(last_row["news_sentiment_mean"])
    else:
        news_sent = np.nan

    # EGY NAGYON egyszerű szabályrendszer – később finomíthatjuk:
    if rel_change > 0.02 and (np.isnan(fear_greed) or fear_greed < 70) and (np.isnan(news_sent) or news_sent >= 0):
        signal = "BUY"
    elif rel_change < -0.02 and (np.isnan(fear_greed) or fear_greed > 30) and (np.isnan(news_sent) or news_sent <= 0):
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "last_close": float(last_close),
        "next_price_pred": float(next_price),
        "rel_change_pred": float(rel_change),
        "fear_greed": None if np.isnan(fear_greed) else float(fear_greed),
        "news_sentiment": None if np.isnan(news_sent) else float(news_sent),
    }
