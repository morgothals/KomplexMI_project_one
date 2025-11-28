# build_training_features.py
"""
Training feature store összeállítása:

- market_data_full.csv (1H OHLCV) + technikai indikátorok
- on-chain mutatók (ONCHAIN_DATA_CSV)
- makró mutatók (MACRO_DATA_CSV)
- hosszú távú sentiment (TRAINING_SENTIMENT_FEATURES_CSV)
- esemény-jellegű feature-ök (halving, China ban, COVID, ETF, stb.)

Kimenet: data/processed/training_features_1h.csv
"""

import numpy as np
import pandas as pd

from modules.config import (
    MARKET_DATA_FULL_CSV,
    ONCHAIN_DATA_CSV,
    MACRO_DATA_CSV,
    TRAINING_SENTIMENT_FEATURES_CSV,
    PROCESSED_DIR,
)
from modules.feature_engineering import add_all_features
from modules.event_features import build_event_features


TRAINING_FEATURES_CSV = PROCESSED_DIR / "training_features_1h.csv"


def _load_df_or_empty(path, index_col="timestamp"):
    """
    CSV beolvasása biztonságosan.
    - ha a fájl nem létezik -> üres DataFrame
    - ha nincs index_col -> üres DataFrame
    - ha van, akkor kényszerítve datetime (utc), NaN timestamp sorok kidobása
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"{path} nem létezik, üres DataFrame-et adunk vissza.")
        return pd.DataFrame()

    if index_col not in df.columns:
        print(f"{path} nem tartalmaz '{index_col}' oszlopot, üres DataFrame-et adunk vissza.")
        return pd.DataFrame()

    df[index_col] = pd.to_datetime(df[index_col], errors="coerce", utc=True)
    df = df.dropna(subset=[index_col])

    if df.empty:
        print(f"{path} csak érvénytelen/üres timestamp sorokat tartalmaz, üres DataFrame.")
        return pd.DataFrame()

    df = df.set_index(index_col).sort_index()
    return df


def build_training_features():
    # 1) Market full history (1H OHLCV)
    df_mkt = _load_df_or_empty(MARKET_DATA_FULL_CSV)
    if df_mkt.empty:
        raise RuntimeError(
            "MARKET_DATA_FULL_CSV üres vagy hiányzik. Futtasd először: bootstrap_market_data.py"
        )

    if not set(["open", "high", "low", "close", "volume"]).issubset(df_mkt.columns):
        raise RuntimeError("MARKET_DATA_FULL_CSV nem tartalmazza az OHLCV oszlopokat.")

    print("Market full shape:", df_mkt.shape)

    # biztosítsuk az 1h felbontást
    df_mkt_1h = df_mkt.resample("1h").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna(subset=["open", "close"])

    print("Market 1h resampled shape:", df_mkt_1h.shape)

    # 2) Technikai indikátorok
    df_feat = add_all_features(df_mkt_1h)
    print("Market with features shape:", df_feat.shape)

    # 3) On-chain
    df_onchain = _load_df_or_empty(ONCHAIN_DATA_CSV)
    if not df_onchain.empty:
        print("On-chain raw shape:", df_onchain.shape)
        df_onchain_1h = df_onchain.resample("1h").ffill()
    else:
        df_onchain_1h = pd.DataFrame(index=df_feat.index)
    # 4) Makró
    df_macro = _load_df_or_empty(MACRO_DATA_CSV)
    if not df_macro.empty:
        print("Macro raw shape:", df_macro.shape)
        df_macro_1h = df_macro.resample("1h").ffill()
    else:
        df_macro_1h = pd.DataFrame(index=df_feat.index)

    # 5) Sentiment (napi) -> 1h
    df_sent_long = _load_df_or_empty(TRAINING_SENTIMENT_FEATURES_CSV)
    if not df_sent_long.empty:
        print("Sentiment long shape:", df_sent_long.shape)
        df_sent_1h = df_sent_long.resample("1h").ffill()
    else:
        df_sent_1h = pd.DataFrame(index=df_feat.index)

    # 6) Esemény feature-ök
    df_events = build_event_features(df_feat.index)
    print("Events shape:", df_events.shape)

    # 7) Join mindenre – market feature-ök a bázis
    df_all = df_feat.join(df_onchain_1h, how="left")
    df_all = df_all.join(df_macro_1h, how="left")
    df_all = df_all.join(df_sent_1h, how="left")
    df_all = df_all.join(df_events, how="left")

    print("Joined (raw) shape:", df_all.shape)


    # NaN / inf kezelése

    # 1) Inf-ekből NaN
    df_all = df_all.replace([np.inf, -np.inf], np.nan)

    # 2) Először időben valamennyire simítsunk: ffill/bfill
    df_all = df_all.ffill()
    df_all = df_all.bfill()

    # 3) Oszloponként döntés:
    #    - ha egy oszlopban gyakorlatilag soha nincs érték -> dobjuk
    #    - különben a maradék NaN-eket töltsük oszlop-átlaggal (tipikus érték)
    cols_to_drop = []
    n_rows = len(df_all)

    for col in df_all.columns:
        non_nan = df_all[col].notna().sum()
        ratio = non_nan / n_rows if n_rows > 0 else 0

        if ratio == 0:
            # soha nincs értelmes adat ebben az oszlopban -> felesleges feature
            print(f"Oszlop eldobjuk (nincs adat): {col}")
            cols_to_drop.append(col)
            continue

        # maradék NaN-ek kitöltése a historikus átlaggal
        mean_val = df_all[col].mean(skipna=True)
        df_all[col] = df_all[col].fillna(mean_val)

    if cols_to_drop:
        df_all = df_all.drop(columns=cols_to_drop)

    print("After cleaning shape:", df_all.shape)

    TRAINING_FEATURES_CSV.parent.mkdir(exist_ok=True, parents=True)
    df_all.to_csv(TRAINING_FEATURES_CSV, index_label="timestamp")

    print("Training features shape:", df_all.shape)
    print(f"Mentve: {TRAINING_FEATURES_CSV}")



if __name__ == "__main__":
    build_training_features()
