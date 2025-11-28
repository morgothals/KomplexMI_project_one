# modules/feature_engineering.py
import numpy as np
import pandas as pd


def add_basic_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hl_range"] = df["high"] - df["low"]
    df["oc_diff"] = df["close"] - df["open"]
    df["ret"] = df["close"].pct_change()
    return df


def ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True
    )


def hma(series: pd.Series, period: int) -> pd.Series:
    """
    Hull Moving Average:
    HMA(n) = WMA( WMA(price, n/2)*2 - WMA(price, n), sqrt(n) )
    """
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    wma_half = wma(series, half)
    wma_full = wma(series, period)
    hma_input = 2 * wma_half - wma_full
    return wma(hma_input, sqrt_n)


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    df["ma_7"] = ma(close, 7)
    df["ma_21"] = ma(close, 21)
    df["ma_50"] = ma(close, 50)

    df["ema_12"] = ema(close, 12)
    df["ema_26"] = ema(close, 26)

    df["hma_21"] = hma(close, 21)
    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)

    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_down = pd.Series(down, index=series.index).rolling(period).mean()

    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    return series.pct_change(periods=period)


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi_14"] = rsi(df["close"], 14)
    df["roc_10"] = roc(df["close"], 10)
    return df


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_std_7"] = df["ret"].rolling(7).std()
    df["ret_std_30"] = df["ret"].rolling(30).std()

    # ATR: Average True Range
    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift()).abs()
    low_close_prev = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    return df


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # OBV
    direction = np.sign(df["close"].diff().fillna(0))
    df["obv"] = (direction * df["volume"]).cumsum()

    # VWAP (egyszerű intraday-mentes verzió, folyamatos kumulált)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vp = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    df["vwap"] = cum_vp / cum_vol

    df["vol_change"] = df["volume"].pct_change()
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A teljes feature pipeline:
    - basic price features
    - trend
    - momentum
    - volatility
    - volume
    """
    df_fe = add_basic_price_features(df)
    df_fe = add_trend_indicators(df_fe)
    df_fe = add_momentum_indicators(df_fe)
    df_fe = add_volatility_indicators(df_fe)
    df_fe = add_volume_indicators(df_fe)

    # nullák/inf-ek kiszűrése
    df_fe = df_fe.replace([np.inf, -np.inf], np.nan)
    df_fe = df_fe.dropna()
    return df_fe
