# modules/log_curve_forecaster.py

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression

from .config import MARKET_DATA_FULL_CSV, BASE_DIR


def load_btc_history():
    """ Beolvassuk a többéves napi BTC close adatot. """
    df = pd.read_csv(MARKET_DATA_FULL_CSV, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    df = df[["close"]].dropna()
    df["close"] = df["close"].astype(float)
    return df


def build_log_regression_curve(end_year=2030, sigma_mult=1.0):
    """
    Logár–idő alapú görbe:
      - súlyozott lineáris regresszió log(price) ~ t
      - a meredekséget a historyból tanuljuk,
      - az egyenest ÚJRAINÁLLÍTJUK, hogy ÁTMENJEN az utolsó biztos ponton.
    """

    df = load_btc_history()

    # -------- 1) Idő index mint szám --------
    start = df.index.min()
    df["t"] = (df.index - start).days.astype(float)

    # -------- 2) Log ár --------
    df["log_price"] = np.log(df["close"])

    # -------- 3) Súlyozott lineáris regresszió --------
    X = df[["t"]].values
    y = df["log_price"].values

    # Súlyok: régi adatok kisebb, frissek nagyobb súly (0.3 -> 1.0)
    n = len(df)
    sample_weights = np.linspace(0.3, 1.0, n)

    model = LinearRegression()
    model.fit(X, y, sample_weight=sample_weights)

    # Eredeti (a, b)
    b = model.coef_[0]
    a = model.intercept_

    # -------- 4) Re-anchoring az utolsó biztos pontra --------
    t_last = df["t"].iloc[-1]
    log_last = df["log_price"].iloc[-1]

    # Új intercept: a_adj úgy, hogy: log_last = a_adj + b * t_last
    a_adj = log_last - b * t_last

    # Predikció függvény az új, anchorolt egyenesre
    def predict_log(t_array):
        return a_adj + b * t_array

    # Residualok az anchorolt modellre (a trend körüli szórás)
    pred_hist = predict_log(df["t"].values)
    residuals = y - pred_hist
    std = residuals.std()

    # -------- 5) Jövőbeli pontok generálása (éves) --------
    years = np.arange(df.index.min().year, end_year + 1)
    future_dates = [datetime(y, 12, 31, tzinfo=timezone.utc) for y in years]
    future_t = np.array([(fd - start).days for fd in future_dates], dtype=float)

    future_log = predict_log(future_t)

    # Ársávok
    future_price = np.exp(future_log)
    future_price_low = np.exp(future_log - sigma_mult * std)
    future_price_high = np.exp(future_log + sigma_mult * std)

    df_pred = pd.DataFrame({
        "timestamp": future_dates,
        "pred_log_price": future_log,
        "pred_price": future_price,
        "pred_price_low": future_price_low,
        "pred_price_high": future_price_high,
    })

    return df_pred, (a_adj, b, std)


def save_log_curve(df_pred):
    pred_dir = Path(BASE_DIR) / "predictions"
    pred_dir.mkdir(exist_ok=True, parents=True)

    out_path = pred_dir / "btc_log_curve_prediction.csv"
    df_pred.to_csv(out_path, index=False)
    print(f"BTC log-regression curve prediction saved to {out_path}, shape={df_pred.shape}")


def run_log_regression_curve(end_year=2030, sigma_mult=1.0):
    df_pred, params = build_log_regression_curve(end_year=end_year, sigma_mult=sigma_mult)
    a_adj, b, std = params
    print(f"Log-regression params: a={a_adj:.4f}, b={b:.8f}, std={std:.4f}")
    save_log_curve(df_pred)
    return df_pred
