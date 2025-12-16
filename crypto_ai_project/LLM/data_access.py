"""Helpers to pull structured context from existing CSV artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

from modules import config


def _read_csv(path: Path, parse_dates: List[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)


def _df_to_records(df: pd.DataFrame, *, max_rows: int) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    out: List[Dict[str, Any]] = []
    df2 = df.head(max_rows)
    for _, row in df2.iterrows():
        item: Dict[str, Any] = {}
        for col in df2.columns:
            val = row.get(col)
            if isinstance(val, (pd.Timestamp,)):
                item[col] = val.isoformat()
            elif pd.isna(val):
                item[col] = None
            else:
                try:
                    item[col] = float(val) if isinstance(val, (np.floating, np.integer)) else val
                except Exception:
                    item[col] = str(val)
        out.append(item)
    return out


def _infer_time_col(df: pd.DataFrame) -> str | None:
    for c in ["timestamp", "date", "time", "datetime"]:
        if c in df.columns:
            return c
    return None


def load_training_features_last_row() -> Dict[str, Any]:
    """Utolsó sor a training_features_1h.csv-ből oszlopnevekkel.

    Vissza: {"columns": [...], "row": {...}}.
    """
    path = Path(config.TRAINING_FEATURES_CSV)
    df = _read_csv(path)
    if df.empty:
        return {"columns": [], "row": {}}

    time_col = _infer_time_col(df)
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.dropna(subset=[time_col]).sort_values(time_col)

    last = df.tail(1)
    if last.empty:
        return {"columns": list(df.columns), "row": {}}

    row_dict = _df_to_records(last, max_rows=1)[0]
    return {"columns": list(df.columns), "row": row_dict}


def load_longterm_features_last_year() -> Dict[str, Any]:
    """Utolsó 1 év a longterm_features_15d.csv-ből oszlopnevekkel.

    15 napos lépték miatt tipikusan ~25 sor.
    Vissza: {"columns": [...], "rows": [...], "time_range": {"start":...,"end":...}}
    """
    path = Path(config.LONGTERM_FEATURES_15D_CSV)
    df = _read_csv(path)
    if df.empty:
        return {"columns": [], "rows": [], "time_range": None}

    time_col = _infer_time_col(df)
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.dropna(subset=[time_col]).sort_values(time_col)
        end = df[time_col].max()
        start = end - pd.Timedelta(days=365)
        df = df[df[time_col] >= start]
        time_range = {"start": start.isoformat(), "end": end.isoformat()}
    else:
        # ha nincs időoszlop, akkor csak az utolsó ~25 sort adjuk
        df = df.tail(25)
        time_range = None

    return {
        "columns": list(df.columns),
        "rows": _df_to_records(df, max_rows=60),
        "time_range": time_range,
    }


def load_last_day_bundle() -> Dict[str, Any]:
    """Utolsó 24 óra snapshot 'mindenből' (kompakt formában).

    - Market 1h: utolsó 24 gyertya (timestamp/open/high/low/close/volume ha van)
    - Intraday 1m: napi összegzés + utolsó 120 pont
    - Sentiment: utolsó nap (és rövid trend)
    - Macro + Onchain: legfrissebb sor
    - News: utolsó ~24 óra (limit)
    """
    out: Dict[str, Any] = {}

    # --- market 1h last 24 rows ---
    df_mkt = _read_csv(Path(config.MARKET_DATA_CSV))
    if not df_mkt.empty and "timestamp" in df_mkt.columns:
        df_mkt["timestamp"] = pd.to_datetime(df_mkt["timestamp"], errors="coerce", utc=True)
        df_mkt = df_mkt.dropna(subset=["timestamp"]).sort_values("timestamp")
        cutoff = df_mkt["timestamp"].max() - pd.Timedelta(hours=24)
        df_mkt_24h = df_mkt[df_mkt["timestamp"] >= cutoff]
        keep_cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df_mkt_24h.columns]
        out["market_1h_24h"] = {
            "columns": keep_cols,
            "rows": _df_to_records(df_mkt_24h[keep_cols].tail(30), max_rows=30),
        }
    else:
        out["market_1h_24h"] = {"columns": [], "rows": []}

    # --- intraday 1m summary + tail ---
    df_intra = _read_csv(Path(config.MARKET_INTRADAY_1M_CSV))
    if not df_intra.empty and "timestamp" in df_intra.columns and "close" in df_intra.columns:
        df_intra["timestamp"] = pd.to_datetime(df_intra["timestamp"], errors="coerce", utc=True)
        df_intra = df_intra.dropna(subset=["timestamp"]).sort_values("timestamp")
        # napi ablak: max - 24h
        cutoff = df_intra["timestamp"].max() - pd.Timedelta(hours=24)
        day = df_intra[df_intra["timestamp"] >= cutoff]
        closes = pd.to_numeric(day["close"], errors="coerce").dropna()
        summary = {
            "points": int(len(day)),
            "last_close": float(closes.iloc[-1]) if not closes.empty else None,
            "high": float(closes.max()) if not closes.empty else None,
            "low": float(closes.min()) if not closes.empty else None,
        }
        keep_cols = [c for c in ["timestamp", "close", "volume"] if c in day.columns]
        out["intraday_1m"] = {
            "summary": summary,
            "columns": keep_cols,
            "tail_rows": _df_to_records(day[keep_cols].tail(120), max_rows=120),
        }
    else:
        out["intraday_1m"] = {"summary": None, "columns": [], "tail_rows": []}

    # --- sentiment last 7d (daily) ---
    df_sent = _read_csv(Path(config.SENTIMENT_DATA_CSV), parse_dates=["timestamp"])
    if not df_sent.empty and "timestamp" in df_sent.columns:
        df_sent = df_sent.sort_values("timestamp")
        cutoff = df_sent["timestamp"].max() - pd.Timedelta(days=7)
        df_sent = df_sent[df_sent["timestamp"] >= cutoff]
        keep_cols = [c for c in ["timestamp", "news_sentiment", "fear_greed"] if c in df_sent.columns]
        out["sentiment_7d"] = {"columns": keep_cols, "rows": _df_to_records(df_sent[keep_cols], max_rows=60)}
    else:
        out["sentiment_7d"] = {"columns": [], "rows": []}

    # --- macro latest ---
    df_macro = _read_csv(Path(config.MACRO_DATA_CSV))
    if not df_macro.empty:
        tcol = _infer_time_col(df_macro)
        if tcol:
            df_macro[tcol] = pd.to_datetime(df_macro[tcol], errors="coerce", utc=True)
            df_macro = df_macro.dropna(subset=[tcol]).sort_values(tcol)
        latest = df_macro.tail(1)
        out["macro_latest"] = {"columns": list(df_macro.columns), "row": _df_to_records(latest, max_rows=1)[0] if not latest.empty else {}}
    else:
        out["macro_latest"] = {"columns": [], "row": {}}

    # --- onchain latest ---
    df_on = _read_csv(Path(config.ONCHAIN_DATA_CSV))
    if not df_on.empty:
        tcol = _infer_time_col(df_on)
        if tcol:
            df_on[tcol] = pd.to_datetime(df_on[tcol], errors="coerce", utc=True)
            df_on = df_on.dropna(subset=[tcol]).sort_values(tcol)
        latest = df_on.tail(1)
        out["onchain_latest"] = {"columns": list(df_on.columns), "row": _df_to_records(latest, max_rows=1)[0] if not latest.empty else {}}
    else:
        out["onchain_latest"] = {"columns": [], "row": {}}

    # --- news last 24h ---
    news = load_recent_news(max_items=25)
    out["news_recent"] = news

    return out


def load_recent_news(max_items: int = 12) -> List[Dict[str, str]]:
    # Ne használjunk parse_dates listát előre, mert ha egy oszlop hiányzik,
    # pandas: "Missing column provided to 'parse_dates'" hibát dob.
    df = _read_csv(Path(config.NEWS_DATA_CSV))
    if df.empty:
        return []

    # időoszlopok óvatos normalizálása
    for candidate in ["timestamp", "published", "date"]:
        if candidate in df.columns:
            df[candidate] = pd.to_datetime(df[candidate], errors="coerce", utc=True)

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
