# modules/advisor.py
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import TRAINING_FEATURES_CSV, TRAINING_SENTIMENT_FEATURES_CSV
from .forecast_model import predict_next_close


def _to_float_or_none(value):
    try:
        if value is None:
            return None
        f = float(value)
        if np.isnan(f):
            return None
        return f
    except Exception:
        return None


def _to_int_or_none(value):
    try:
        if value is None:
            return None
        i = int(float(value))
        return i
    except Exception:
        return None


def _get_last_valid_from_training_sentiment(col_name: str):
    try:
        df = pd.read_csv(TRAINING_SENTIMENT_FEATURES_CSV, parse_dates=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        if col_name not in df.columns:
            return None
        s = df[col_name].dropna()
        if s.empty:
            return None
        return s.iloc[-1]
    except Exception:
        return None


def generate_advice() -> dict:
    """Tanács generálás a modell predikció + utolsó feature sor alapján.

    Visszatér egy olyan dict-tel, ami:
    - tartalmazza a jelzést (BUY/HOLD/SELL)
    - a modell predikcióját (ár + várható változás)
    - döntést segítő kontextust (trend/momentum/volatilítás/sentiment/makró)
    - rövid, emberi indoklást
    """
    next_price, last_close, last_row = predict_next_close()

    rel_change = (next_price - last_close) / last_close if last_close else 0.0
    pred_log_return = float(np.log(next_price / last_close)) if last_close else 0.0

    # timestamp az utolsó sorból
    last_ts = getattr(last_row, "name", None)
    if isinstance(last_ts, pd.Timestamp):
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        last_ts_iso = last_ts.isoformat()
        data_age_hours = (datetime.now(timezone.utc) - last_ts.to_pydatetime()).total_seconds() / 3600
    else:
        last_ts_iso = None
        data_age_hours = None

    # sentiment oszlopok a training_features-ben: news_sentiment, fear_greed, bullish_ratio, bearish_ratio, news_sentiment_std
    fear_greed = _to_int_or_none(last_row.get("fear_greed"))
    news_sent = _to_float_or_none(last_row.get("news_sentiment"))
    news_sent_std = _to_float_or_none(last_row.get("news_sentiment_std"))
    bullish_ratio = _to_float_or_none(last_row.get("bullish_ratio"))
    bearish_ratio = _to_float_or_none(last_row.get("bearish_ratio"))

    # fallback: ha valamiért NaN maradt a training_features-ben
    if news_sent is None:
        news_sent = _to_float_or_none(_get_last_valid_from_training_sentiment("news_sentiment"))
    if fear_greed is None:
        fear_greed = _to_int_or_none(_get_last_valid_from_training_sentiment("fear_greed"))

    # market/indikátorok (ha vannak)
    rsi_14 = _to_float_or_none(last_row.get("rsi_14"))
    atr_14 = _to_float_or_none(last_row.get("atr_14"))
    ret_std_30 = _to_float_or_none(last_row.get("ret_std_30"))
    ma_21 = _to_float_or_none(last_row.get("ma_21"))
    ma_50 = _to_float_or_none(last_row.get("ma_50"))
    vwap = _to_float_or_none(last_row.get("vwap"))
    vol_change = _to_float_or_none(last_row.get("vol_change"))

    trend_notes = []
    if ma_21 is not None:
        trend_notes.append("above_ma21" if last_close > ma_21 else "below_ma21")
    if ma_50 is not None:
        trend_notes.append("above_ma50" if last_close > ma_50 else "below_ma50")

    # rövid távú hozamok a training_features-ből (tail read)
    recent_returns = {}
    try:
        df_tail = pd.read_csv(TRAINING_FEATURES_CSV, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        close = df_tail["close"].astype(float)
        def pct(n):
            if len(close) > n:
                return _to_float_or_none((close.iloc[-1] / close.iloc[-(n + 1)] - 1.0) * 100.0)
            return None

        recent_returns = {
            "ret_1h_pct": pct(1),
            "ret_24h_pct": pct(24),
            "ret_7d_pct": pct(24 * 7),
        }
    except Exception:
        recent_returns = {"ret_1h_pct": None, "ret_24h_pct": None, "ret_7d_pct": None}

    # makró (ha van)
    macro = {
        "sp500_close": _to_float_or_none(last_row.get("sp500_close")),
        "dxy_close": _to_float_or_none(last_row.get("dxy_close")),
    }

    # on-chain (név variációk miatt mindkét formát próbáljuk)
    onchain = {}
    for key, candidates in {
        "tx_count": ["tx_count", "n-transactions"],
        "active_addresses": ["active_addresses", "n-unique-addresses"],
        "hash_rate": ["hash_rate", "hash-rate"],
        "avg_block_size": ["avg_block_size", "avg-block-size"],
        "miners_revenue": ["miners_revenue", "miners-revenue"],
    }.items():
        val = None
        for c in candidates:
            val = _to_float_or_none(last_row.get(c))
            if val is not None:
                break
        onchain[key] = val

    # Jelzés + indoklás
    rationale = []
    pred_change_pct = rel_change * 100.0

    # nagyon egyszerű szabályrendszer (továbbra is determinisztikus, de több kontextust adunk hozzá)
    buy_ok = pred_change_pct > 2.0
    sell_ok = pred_change_pct < -2.0

    # sentiment guardrails (ha van adat)
    if fear_greed is not None:
        rationale.append(f"Fear&Greed={fear_greed}")
    if news_sent is not None:
        rationale.append(f"NewsSent={news_sent:.3f}" + (f", std={news_sent_std:.3f}" if news_sent_std is not None else ""))

    if rsi_14 is not None:
        rationale.append(f"RSI14={rsi_14:.1f}")
    if trend_notes:
        rationale.append("Trend=" + ",".join(trend_notes))

    if buy_ok and (fear_greed is None or fear_greed < 70) and (news_sent is None or news_sent >= 0):
        signal = "BUY"
    elif sell_ok and (fear_greed is None or fear_greed > 30) and (news_sent is None or news_sent <= 0):
        signal = "SELL"
    else:
        signal = "HOLD"

    # további, döntést segítő “notes”
    notes = []
    if data_age_hours is not None and data_age_hours > 2:
        notes.append(f"Data is {data_age_hours:.1f}h old")
    if rsi_14 is not None:
        if rsi_14 >= 70:
            notes.append("RSI suggests overbought")
        elif rsi_14 <= 30:
            notes.append("RSI suggests oversold")
    if atr_14 is not None and last_close:
        notes.append(f"ATR14~{(atr_14/last_close*100.0):.2f}% of price")

    return {
        "signal": signal,
        "timestamp": last_ts_iso,
        "data_age_hours": _to_float_or_none(data_age_hours),
        "horizon": "1h",
        "last_close": _to_float_or_none(last_close),
        "next_price_pred": _to_float_or_none(next_price),
        "pred_log_return": _to_float_or_none(pred_log_return),
        "pred_change_pct": _to_float_or_none(pred_change_pct),
        # Backward-compatible fields (korábbi dashboard/LLM modulokhoz)
        "rel_change_pred": _to_float_or_none(rel_change),
        "fear_greed": fear_greed,
        "news_sentiment": news_sent,
        "recent_returns": recent_returns,
        "market": {
            "rsi_14": rsi_14,
            "atr_14": atr_14,
            "ret_std_30": ret_std_30,
            "ma_21": ma_21,
            "ma_50": ma_50,
            "vwap": vwap,
            "vol_change": vol_change,
        },
        "sentiment": {
            "fear_greed": fear_greed,
            "news_sentiment": news_sent,
            "news_sentiment_std": news_sent_std,
            "bullish_ratio": bullish_ratio,
            "bearish_ratio": bearish_ratio,
        },
        "macro": macro,
        "onchain": onchain,
        "rationale": rationale,
        "notes": notes,
    }
