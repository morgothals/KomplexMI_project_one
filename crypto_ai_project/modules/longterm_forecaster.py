# modules/longterm_forecaster.py

import math
from datetime import datetime, timedelta, timezone

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from .config import LONGTERM_FEATURES_15D_CSV, BASE_DIR


# ---------- Adatbetöltés ----------

def load_longterm_features() -> pd.DataFrame:
    """
    Beolvassa a 15 napos longterm feature setet.
    Elvárás: timestamp index, oszlopok közt:
      - price_close
      - target_log_return_5y
      - target_vol_5y
      - (illetve az összes egyéb feature, amit a longterm builder generált)
    """
    df = pd.read_csv(LONGTERM_FEATURES_15D_CSV, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


# ---------- Modell tanítás egy adott targetre ----------

def train_rf_model(df: pd.DataFrame, target_col: str):
    """
    Egyszerű RandomForestRegressor az adott targetre.
    Csak azokat a sorokat használjuk, ahol a target nem NaN.
    """

    df_train = df.dropna(subset=[target_col]).copy()
    if df_train.empty:
        raise RuntimeError(f"Nincs elég adat a(z) {target_col} tanításához (minden NaN).")

    # Feature oszlopok – a target és más target oszlopok nélkül
    drop_targets = [
        "target_log_return_5y",
        "target_vol_5y",
        "target_log_return_1y",
        "target_vol_1y",
    ]
    feature_cols = [c for c in df_train.columns if c not in drop_targets]

    X = df_train[feature_cols].copy()
    y = df_train[target_col].copy()

    # Egyszerű idősoros split (idő szerint rendezett, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"RandomForest score ({target_col}): {score:.4f}")

    return model, feature_cols


def predict_on_dataframe(model, df_features: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """
    Modell alkalmazása egy tetszőleges DataFrame-en (ugyanazzal a feature set-tel).
    """
    X = df_features[feature_cols].copy()
    preds = model.predict(X)
    return preds


# ---------- Éves rács + előrejelzés ----------

def build_annual_feature_grid(df: pd.DataFrame,
                              start_year: int | None = None,
                              end_year: int | None = None) -> pd.DataFrame:
    """
    15 napos longterm feature setből ÉVES rács:
      - minden év végén (dec 31 körül) egy sor
      - NEM nyúlunk túl a valós adatokon: end_year legfeljebb az utolsó elérhető év.
    """

    # Éves resample – minden év végére egy sor
    df_annual = df.resample("1Y").last()

    # Elérhető valós év-tartomány
    first_year_real = df_annual.index.min().year
    last_year_real = df_annual.index.max().year

    # Paraméterek normalizálása
    if start_year is None or start_year < first_year_real:
        start_year = first_year_real
    if end_year is None or end_year > last_year_real:
        end_year = last_year_real

    # Éves dátumok: start_year..end_year, év vége (UTC)
    annual_dates = pd.date_range(
        start=datetime(start_year, 12, 31, tzinfo=timezone.utc),
        end=datetime(end_year, 12, 31, tzinfo=timezone.utc),
        freq="12M",
    )

    # Ráhúzzuk az éves resample-re és legközelebbi értéket vesszük
    df_annual = df_annual.reindex(annual_dates, method="nearest")
    df_annual.index.name = "timestamp"
    return df_annual


def build_long_horizon_curve(start_year: int = 2012,
                             end_year: int = 2030,
                             sigma_multiplier: float = 1.0) -> pd.DataFrame:
    """
    Hosszú távú (5 éves) előrejelzési görbe készítése éves bontásban.

    - Minden sor: egy év vége (t)
    - Modellek:
        * target_log_return_5y  (várható loghozam 5 év múlva)
        * target_vol_5y         (5 éves volatilitás – közelítés)
    - Kimenet oszlopok:
        * current_timestamp = t
        * horizon_timestamp  = t + 5 év
        * current_price
        * pred_log_return_5y
        * pred_vol_5y
        * pred_price_5y
        * pred_price_5y_low_1sigma
        * pred_price_5y_high_1sigma
    """

    df = load_longterm_features()

    # 1) Modellek tanítása a teljes history-n
    model_ret, feature_cols = train_rf_model(df, "target_log_return_5y")
    model_vol, _ = train_rf_model(df, "target_vol_5y")

    # 2) Éves rács építése 2012..2030 (vagy amit paraméterben kérsz)
    df_annual = build_annual_feature_grid(
        df,
        start_year=start_year,
        end_year=end_year,
    )

    if "price_close" not in df_annual.columns:
        raise RuntimeError("A longterm feature set nem tartalmaz 'price_close' oszlopot.")

    # 3) Előrejelzések az éves gridre
    preds_ret = predict_on_dataframe(model_ret, df_annual, feature_cols)
    preds_vol = predict_on_dataframe(model_vol, df_annual, feature_cols)

    current_prices = df_annual["price_close"].astype(float).values

    # 4) Jövőbeli 5 éves árak számítása
    # price_5y = price_now * exp(pred_log_return_5y)
    pred_price_5y = current_prices * np.exp(preds_ret)

    # egyszerű sáv: ± sigma_multiplier * pred_vol_5y
    # Itt pred_vol_5y-t szórásként kezeljük a loghozamra.
    pred_price_low = current_prices * np.exp(preds_ret - sigma_multiplier * preds_vol)
    pred_price_high = current_prices * np.exp(preds_ret + sigma_multiplier * preds_vol)

    # 5) Eredmény DataFrame
    result = pd.DataFrame(
        {
            "current_timestamp": df_annual.index,
            "horizon_timestamp": df_annual.index + pd.to_timedelta(5 * 365, unit="D"),
            "current_price": current_prices,
            "pred_log_return_5y": preds_ret,
            "pred_vol_5y": preds_vol,
            "pred_price_5y": pred_price_5y,
            "pred_price_5y_low_1sigma": pred_price_low,
            "pred_price_5y_high_1sigma": pred_price_high,
        }
    )

    result = result.sort_values("current_timestamp").reset_index(drop=True)
    return result


# ---------- Mentés a predictions mappába ----------

def save_long_horizon_curve(df_curve: pd.DataFrame, filename: str = "btc_5y_curve_annual.csv"):
    pred_dir = Path(BASE_DIR) / "predictions"
    pred_dir.mkdir(exist_ok=True, parents=True)
    out_path = pred_dir / filename
    df_curve.to_csv(out_path, index=False)
    print(f"Hosszú távú 5 éves görbe mentve: {out_path}, shape={df_curve.shape}")


def run_build_long_horizon_curve(
    start_year: int = 2012,
    end_year: int = 2030,
    sigma_multiplier: float = 1.0,
):
    """
    Fő belépési pont: 5 éves görbe + ±1σ sáv generálása éves bontásban,
    és mentése a predictions mappába.
    """
    print(f">>> Hosszú távú 5Y görbe építése {start_year}–{end_year} intervallumra...")
    curve = build_long_horizon_curve(
        start_year=start_year,
        end_year=end_year,
        sigma_multiplier=sigma_multiplier,
    )
    save_long_horizon_curve(curve)
    return curve
