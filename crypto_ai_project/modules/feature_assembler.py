# modules/feature_assembler.py
import pandas as pd
from datetime import timezone

from .config import (
    MARKET_FEATURES_CSV,
    ONCHAIN_DATA_CSV,
    MACRO_DATA_CSV,
    SENTIMENT_DATA_CSV,
    ALL_FEATURES_CSV,
)


def _load_df_or_empty(path, index_col="timestamp"):
    try:
        df = pd.read_csv(path, parse_dates=[index_col])
        # Ha nincs timezone, tegyük UTC-re
        if df[index_col].dt.tz is None:
            df[index_col] = df[index_col].dt.tz_localize("UTC")
        df = df.set_index(index_col)
        return df.sort_index()
    except FileNotFoundError:
        return pd.DataFrame()


def build_all_features(resample_rule: str = "1H") -> pd.DataFrame:
    """
    Összefűzi a market feature-öket, on-chain, makró és sentiment adatokat
    közös 1 órás időskálára.
    Eredményt ALL_FEATURES_CSV-be menti.
    """

    # 1) Market technikai feature-ök – EZ a baseline (1H)
    df_mkt = _load_df_or_empty(MARKET_FEATURES_CSV)
    if df_mkt.empty:
        raise RuntimeError("MARKET_FEATURES_CSV üres vagy nincs. Futtasd: build_features először.")

    # biztos ami biztos, resample 1H
    df_mkt = df_mkt.resample(resample_rule).last().dropna()

    # 2) On-chain
    df_onchain = _load_df_or_empty(ONCHAIN_DATA_CSV)
    if not df_onchain.empty:
        df_onchain = df_onchain.resample(resample_rule).ffill()

    # 3) Makró (Yahoo Finance)
    df_macro = _load_df_or_empty(MACRO_DATA_CSV)
    if not df_macro.empty:
        df_macro = df_macro.resample(resample_rule).ffill()

    # 4) Sentiment (news_sentiment + fear_greed – napi idősor)
    df_sent = _load_df_or_empty(SENTIMENT_DATA_CSV)
    if not df_sent.empty:
        # napi → órás, forward fill
        df_sent = df_sent.resample(resample_rule).ffill()

    # 5) Join – market az alap, a többire left join + ffill
    df_all = df_mkt.copy()

    for extra_df in [df_onchain, df_macro, df_sent]:
        if extra_df is not None and not extra_df.empty:
            df_all = df_all.join(extra_df, how="left")

    # Előre kitöltjük a hiányzó makró/sentiment/on-chain értékeket,
    # hogy az LSTM ne kapjon NaN-t.
    df_all = df_all.ffill().dropna()

    # Mentés
    df_all.to_csv(ALL_FEATURES_CSV, index_label="timestamp")
    return df_all
