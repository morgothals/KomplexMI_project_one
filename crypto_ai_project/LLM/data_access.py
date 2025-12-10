"""Helpers to pull structured context from existing CSV artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from modules import config


def _read_csv(path: Path, parse_dates: List[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)


def load_recent_news(max_items: int = 12) -> List[Dict[str, str]]:
    df = _read_csv(Path(config.NEWS_DATA_CSV), parse_dates=["timestamp", "published", "date"])
    if df.empty:
        return []

    time_col = None
    for candidate in ["timestamp", "published", "date"]:
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col:
        df = df.sort_values(time_col, ascending=False)
    else:
        df = df.tail(max_items)

    rows = []
    for _, row in df.head(max_items).iterrows():
        rows.append(
            {
                "headline": str(row.get("title") or row.get("headline") or row.get("news") or ""),
                "summary": str(row.get("summary") or row.get("description") or ""),
                "source": str(row.get("source") or ""),
                "published": str(row.get(time_col) or ""),
                "sentiment": row.get("compound") or row.get("sentiment"),
                "url": str(row.get("url") or ""),
            }
        )
    return rows


def load_sentiment_snapshot(days: int = 60) -> Dict[str, float | None]:
    df = _read_csv(Path(config.SENTIMENT_DATA_CSV), parse_dates=["timestamp"])
    if df.empty:
        return {"latest": None, "avg": None, "fear_greed": None}

    df = df.sort_values("timestamp")
    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    df_recent = df[df["timestamp"] >= cutoff]

    latest = df_recent.tail(1)
    return {
        "latest": float(latest["news_sentiment"].iloc[0]) if not latest.empty and pd.notna(latest["news_sentiment"].iloc[0]) else None,
        "avg": float(df_recent["news_sentiment"].mean()) if "news_sentiment" in df_recent else None,
        "fear_greed": float(latest["fear_greed"].iloc[0]) if not latest.empty and "fear_greed" in latest and pd.notna(latest["fear_greed"].iloc[0]) else None,
    }


def load_market_context(days: int = 180) -> Dict[str, float | None]:
    # prefer full history, fallback to operational
    for path in [config.MARKET_DATA_FULL_CSV, config.MARKET_DATA_CSV]:
        df = _read_csv(Path(path), parse_dates=["timestamp"])
        if not df.empty:
            break
    if df.empty:
        return {"last_close": None, "return_%": None, "volatility_%": None}

    df = df.sort_values("timestamp")
    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    df_recent = df[df["timestamp"] >= cutoff]

    if df_recent.empty:
        return {"last_close": None, "return_%": None, "volatility_%": None}

    closes = df_recent["close"].astype(float)
    last_close = float(closes.iloc[-1])
    first_close = float(closes.iloc[0])
    ret_pct = (last_close / first_close - 1.0) * 100 if first_close else None

    log_returns = np.log(closes).diff().dropna()
    vol = float(log_returns.std() * np.sqrt(365)) * 100 if not log_returns.empty else None

    return {"last_close": last_close, "return_%": ret_pct, "volatility_%": vol}


def load_long_curve(years_ahead: int = 5) -> Tuple[List[str], List[float]]:
    path = Path(config.BASE_DIR) / "predictions" / "btc_log_curve_prediction.csv"
    df = _read_csv(path, parse_dates=["timestamp"])
    if df.empty:
        return [], []

    df = df.sort_values("timestamp")
    labels = [str(ts.year) for ts in df["timestamp"]]
    prices = [float(x) for x in df["pred_price"]]

    if years_ahead and len(prices) > years_ahead:
        labels = labels[:years_ahead]
        prices = prices[:years_ahead]

    return labels, prices
