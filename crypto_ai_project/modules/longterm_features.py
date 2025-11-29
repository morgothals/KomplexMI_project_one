# modules/longterm_features.py

import math
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import (

    MARKET_DATA_FULL_CSV,
    ONCHAIN_DATA_CSV,
    MACRO_DATA_CSV,
    TRAINING_SENTIMENT_FEATURES_CSV,
    LONGTERM_FEATURES_15D_CSV,
)


def _load_market_data() -> pd.DataFrame:
    df = pd.read_csv(MARKET_DATA_FULL_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Napi OHLC aggregálás
    df_daily = df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })
    df_daily = df_daily.dropna(subset=["close"])
    return df_daily

def _load_onchain_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(ONCHAIN_DATA_CSV)
    except FileNotFoundError:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def _load_macro_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(MACRO_DATA_CSV)
    except FileNotFoundError:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def _load_sentiment_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(TRAINING_SENTIMENT_FEATURES_CSV, parse_dates=["timestamp"])
    except FileNotFoundError:
        return pd.DataFrame()
    df = df.set_index("timestamp").sort_index()
    # várjuk, hogy legyen: news_sentiment, fear_greed
    return df


def build_longterm_btc_features() -> pd.DataFrame:
    """
    Hosszútávú, 15 napos bontású feature dataset BTC árfolyam előrejelzéséhez.

    Források:
      - market_data.csv  (Binance OHLCV)
      - onchain_data.csv (tx_count, active_addresses, stb.)
      - macro_data.csv   (sp500, dxy, stb.)
      - training_sentiment_features.csv (news_sentiment, fear_greed)

    Kimenet:
      - LONGTERM_FEATURES_15D_CSV
      - index: timestamp (UTC, a 15 napos ablak vége)
    """

    # --- 1) Market data: napi ár + loghozamok + rolling mutatók ---

    df_mkt = _load_market_data()
    if df_mkt.empty:
        raise RuntimeError("MARKET_DATA_CSV üres vagy hiányzik.")

    # napi loghozam
    df_mkt["close"] = df_mkt["close"].astype(float)
    df_mkt["log_return_1d"] = (df_mkt["close"] / df_mkt["close"].shift(1)).apply(
        lambda x: math.log(x) if x > 0 else float("nan")
    )

    # rolling mutatók napi szinten
    df_mkt["sma_30d"] = df_mkt["close"].rolling(window=30).mean()
    df_mkt["sma_90d"] = df_mkt["close"].rolling(window=90).mean()
    df_mkt["sma_180d"] = df_mkt["close"].rolling(window=180).mean()

    df_mkt["vol_30d"] = df_mkt["log_return_1d"].rolling(window=30).std()
    df_mkt["vol_90d"] = df_mkt["log_return_1d"].rolling(window=90).std()

    # drawdown 180 napos maxhoz képest
    rolling_max_180 = df_mkt["close"].rolling(window=180).max()
    df_mkt["drawdown_180d"] = df_mkt["close"] / rolling_max_180 - 1.0

    # --- 2) 15 napos rácsra aggregálás (áras feature-ök) ---

    # 15 napos záróár (ablak vége)
    df_price_15d = df_mkt[["close"]].resample("15D", label="right", closed="right").last()
    df_price_15d = df_price_15d.rename(columns={"close": "price_close"})

    # 15 / 30 / 90 napos loghozam a napi close-ból
    # (itt a 15 napos rácson számoljuk, shiftelt close alapján)
    df_price_15d["log_return_15d"] = (
        df_price_15d["price_close"] / df_price_15d["price_close"].shift(1)
    ).apply(lambda x: math.log(x) if x > 0 else float("nan"))

    # 30 nap ~ 2 * 15 nap, 90 nap ~ 6 * 15 nap
    df_price_15d["log_return_30d"] = (
        df_price_15d["price_close"] / df_price_15d["price_close"].shift(2)
    ).apply(lambda x: math.log(x) if x > 0 else float("nan"))

    df_price_15d["log_return_90d"] = (
        df_price_15d["price_close"] / df_price_15d["price_close"].shift(6)
    ).apply(lambda x: math.log(x) if x > 0 else float("nan"))

    # napi rolling mutatók 15 napos rácsra átszedve (utolsó érték az ablak végén)
    df_roll_15d = df_mkt[["sma_30d", "sma_90d", "sma_180d", "vol_30d", "vol_90d", "drawdown_180d"]] \
        .resample("15D", label="right", closed="right").last()

    df_price_15d = df_price_15d.join(df_roll_15d, how="left")

    # --- 3) On-chain: 15 napos átlagok ---

    df_on = _load_onchain_data()
    if not df_on.empty:
        # neveket igazítsd a saját onchain_data.csv-oszlopaidhoz
        cols_on = [c for c in df_on.columns if c not in ("timestamp",)]
        df_on_15d = df_on[cols_on].resample("15D", label="right", closed="right").mean()
    else:
        df_on_15d = pd.DataFrame(index=df_price_15d.index)

    # --- 4) Makró: 15 napos hozamok / átlagok ---

    df_macro = _load_macro_data()
    if not df_macro.empty:
        # feltételezzük, hogy van pl. 'sp500_close', 'dxy_close'
        macro_cols = [c for c in df_macro.columns if c not in ("timestamp",)]
        df_macro_15d = df_macro[macro_cols].resample("15D", label="right", closed="right").last()

        if "sp500_close" in df_macro_15d.columns:
            df_macro_15d["sp500_15d_return"] = (
                df_macro_15d["sp500_close"] / df_macro_15d["sp500_close"].shift(1)
            ).apply(lambda x: math.log(x) if x > 0 else float("nan"))

        if "dxy_close" in df_macro_15d.columns:
            df_macro_15d["dxy_15d_return"] = (
                df_macro_15d["dxy_close"] / df_macro_15d["dxy_close"].shift(1)
            ).apply(lambda x: math.log(x) if x > 0 else float("nan"))
    else:
        df_macro_15d = pd.DataFrame(index=df_price_15d.index)

    # --- 5) Sentiment: 15 napos átlagok ---

    df_sent = _load_sentiment_data()
    if not df_sent.empty:
        # elvárt: 'news_sentiment', 'fear_greed'
        df_sent_15d = df_sent[["news_sentiment", "fear_greed"]] \
            .resample("15D", label="right", closed="right").mean()

        # egyszerű sentiment trend jel: utolsó 15 nap átlag - előző 15 nap átlag
        df_sent_15d["news_sentiment_15d_mean"] = df_sent_15d["news_sentiment"]
        df_sent_15d["fear_greed_15d_mean"] = df_sent_15d["fear_greed"]
        df_sent_15d["news_sentiment_15d_trend"] = (
            df_sent_15d["news_sentiment_15d_mean"] -
            df_sent_15d["news_sentiment_15d_mean"].shift(1)
        )
    else:
        df_sent_15d = pd.DataFrame(index=df_price_15d.index)

    # --- 6) Minden feature összefésülése 15 napos rácson ---

    df_15d = df_price_15d.join(df_on_15d, how="left")
    df_15d = df_15d.join(df_macro_15d, how="left")
    df_15d = df_15d.join(df_sent_15d, how="left")

    # --- 7) Célváltozók: jövőbeli hozam + volatilitás ---

    # napi szintű price_close a célhoz
    daily_close = df_mkt["close"]

    # 1 éves (365 napos) jövőbeli loghozam és volatilitás
    horizon_1y = 365
    future_close_1y = daily_close.shift(-horizon_1y)
    future_logret_1y_daily = (future_close_1y / daily_close).apply(
        lambda x: math.log(x) if x > 0 else float("nan")
    )

    # 1 éves jövőbeli napi loghozamok szórása (rolling "előrefelé")
    # egyszerű hack: visszafelé görgetés, aztán visszafordítás
    rev_logret = df_mkt["log_return_1d"][::-1]
    future_vol_1y_rev = rev_logret.rolling(window=horizon_1y).std()
    future_vol_1y = future_vol_1y_rev[::-1]

    # ezekből 15 napos rácsra mintavételezés (utolsó elérhető nap)
    df_target_1y = pd.DataFrame(index=df_mkt.index)
    df_target_1y["target_log_return_1y"] = future_logret_1y_daily
    df_target_1y["target_vol_1y"] = future_vol_1y

    df_target_1y_15d = df_target_1y.resample("15D", label="right", closed="right").last()

    df_15d = df_15d.join(df_target_1y_15d, how="left")

    # opcionálisan: 5 éves target (365*5), csak ahol van elég adat
    horizon_5y = 365 * 5
    if len(daily_close) > horizon_5y + 10:  # csak ha van értelme
        future_close_5y = daily_close.shift(-horizon_5y)
        future_logret_5y_daily = (future_close_5y / daily_close).apply(
            lambda x: math.log(x) if x > 0 else float("nan")
        )
        rev_logret_5y = df_mkt["log_return_1d"][::-1]
        future_vol_5y_rev = rev_logret_5y.rolling(window=horizon_5y).std()
        future_vol_5y = future_vol_5y_rev[::-1]

        df_target_5y = pd.DataFrame(index=df_mkt.index)
        df_target_5y["target_log_return_5y"] = future_logret_5y_daily
        df_target_5y["target_vol_5y"] = future_vol_5y

        df_target_5y_15d = df_target_5y.resample("15D", label="right", closed="right").last()
        df_15d = df_15d.join(df_target_5y_15d, how="left")

    # --- 8) Index és mentés ---

    df_15d = df_15d.sort_index()
    # index timestamp marad (UTC), csak nevezzük el
    df_15d.index.name = "timestamp"

    LONGTERM_FEATURES_15D_CSV.parent.mkdir(exist_ok=True, parents=True)
    df_15d.to_csv(LONGTERM_FEATURES_15D_CSV, index=True)
    print(f"Hosszútávú 15 napos feature dataset mentve: {LONGTERM_FEATURES_15D_CSV}, shape={df_15d.shape}")

    return df_15d
