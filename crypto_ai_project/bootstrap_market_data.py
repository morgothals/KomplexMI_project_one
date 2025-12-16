# bootstrap_market_data.py
"""
Egyszeri + inkrementális bootstrap script:

- Kaggle bitcoin CSV betöltése (lokális, statikus)
- 1 órás OHLCV-re resample-ölése
- Binance 1H OHLCV történelmi adat INKREMENTÁLIS frissítése:
    - ha már van binance_market_1h.csv, onnan folytatja
    - ha még nincs, 2017-től indul
- Kaggle + Binance 1H egyesítése -> market_data_full.csv (deduplikálva)
"""

from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
from requests.exceptions import ReadTimeout, ConnectionError

from modules.config import (
    KAGGLE_MARKET_CSV,
    MARKET_DATA_FULL_CSV,
    BINANCE_MARKET_FULL_CSV,
    BINANCE_BASE_URL,
    SYMBOL,
)


# ---------- Kaggle betöltés & 1h resample ----------

def load_kaggle_bitcoin_1h() -> pd.DataFrame:
    """
    Kaggle bitcoin historikus adat betöltése és 1 órás OHLCV-re resample-ölése.
    Feltételezzük, hogy:
        data/raw/bitcoin_kaggle.csv
    létezik, és tartalmaz 'Timestamp' (epoch sec) vagy 'Date' oszlopot.
    """
    print(f"Kaggle adat betöltése innen: {KAGGLE_MARKET_CSV}")
    df = pd.read_csv(KAGGLE_MARKET_CSV)

    # időoszlop felderítése
    if "Timestamp" in df.columns:  # tipikus Kaggle: UNIX epoch seconds
        ts = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
    elif "Date" in df.columns:
        ts = pd.to_datetime(df["Date"], utc=True)
    else:
        raise ValueError("Nem található 'Timestamp' vagy 'Date' oszlop a Kaggle fájlban.")

    df["timestamp"] = ts

    # OHLCV oszlopok azonosítása
    col_map = {}
    for col in df.columns:
        c = col.lower()
        if c == "open":
            col_map["open"] = col
        elif c == "high":
            col_map["high"] = col
        elif c == "low":
            col_map["low"] = col
        elif c == "close":
            col_map["close"] = col

    vol_col = None
    for col in df.columns:
        if "volume" in col.lower():
            vol_col = col
            break

    if len(col_map) < 4 or vol_col is None:
        raise ValueError("Nem sikerült egyértelműen azonosítani az OHLCV oszlopokat a Kaggle fájlban.")

    df = df[["timestamp", col_map["open"], col_map["high"], col_map["low"], col_map["close"], vol_col]]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

    df = df.set_index("timestamp").sort_index()

    # 1h resample (kis 'h', hogy ne sírjon a pandas warninggal)
    df_1h = df.resample("1h").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # dobjuk azokat a gyertyákat, ahol nincs open/close
    df_1h = df_1h.dropna(subset=["open", "close"])

    print("Kaggle 1h shape:", df_1h.shape)
    return df_1h


# ---------- Binance kline batch + retry ----------

def fetch_binance_klines_batch(symbol, interval, start_ms=None, end_ms=None,
                               limit=1000, timeout=20, max_retries=3):
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (ReadTimeout, ConnectionError) as e:
            print(f"Binance kline timeout/hálózati hiba (próbálkozás {attempt}/{max_retries}): {e}")
            last_err = e
        except requests.HTTPError as e:
            print(f"Binance HTTP hiba: {e}")
            last_err = e
            break

    print("Többszöri próbálkozás után sem sikerült lekérni a batch-et, kilépünk ebből a szakaszból.")
    return []  # üres -> a hívó fél kilép a ciklusból


# ---------- Binance history INKREMENTÁLIS frissítése ----------

def update_binance_history_1h(symbol=SYMBOL, interval="1h") -> pd.DataFrame:
    """
    Inkrementálisan frissíti a binance_market_1h.csv-t:

    - ha a fájl nem létezik:
        * 2017-01-01 UTC-től indul
    - ha létezik:
        * betölti, megnézi a legutolsó timestampet,
        * onnan felfelé húz további gyertyákat egészen 'most'-ig (vagy amíg engedi az API)
    """

    try:
        df_existing = pd.read_csv(BINANCE_MARKET_FULL_CSV, parse_dates=["timestamp"])
        df_existing = df_existing.set_index("timestamp").sort_index()
        print(f"Binance meglévő history: {df_existing.shape}")
        # induljunk a legutolsó idősorról
        last_ts = df_existing.index.max()
        # Binance klines startTime ms-ben
        start_ts = int(last_ts.timestamp() * 1000) + 1
    except FileNotFoundError:
        print("Nincs még binance_market_1h.csv, teljes history-t húzunk 2017-től.")
        df_existing = pd.DataFrame()
        start_ts = int(datetime(2017, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_new_rows = []

    # Ha már nagyon közel vagyunk a jelenhez (pl. < 1h gap), akár ki is léphetünk
    if df_existing is not None and not df_existing.empty and start_ts >= now_ms - 60 * 60 * 1000:
        print("Binance history már naprakésznek tűnik, nem húzunk új adatot.")
    else:
        print("Binance 1h history frissítése... (lehet, hogy pár perc)")
        while True:
            batch = fetch_binance_klines_batch(symbol, interval, start_ms=start_ts)
            if not batch:
                break

            all_new_rows.extend(batch)
            last_open_time = batch[-1][0]

            # ha elértük a jelent, vagy már nem jön 1000 elem, lépjünk ki
            if last_open_time >= now_ms or len(batch) < 1000:
                break

            start_ts = last_open_time + 1

    # Ha nincs semmi új, és van már meglévő adat, térjünk vissza vele
    if (not all_new_rows) and (df_existing is not None) and (not df_existing.empty):
        print("Nem érkezett új Binance gyertya, marad a meglévő adat.")
        df_all = df_existing
    else:
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "num_trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ]

        df_new = pd.DataFrame(all_new_rows, columns=cols) if all_new_rows else pd.DataFrame(columns=cols)
        if not df_new.empty:
            df_new["open_time"] = pd.to_datetime(df_new["open_time"], unit="ms", utc=True)
            df_new = df_new.set_index("open_time").sort_index()
            df_new[["open", "high", "low", "close", "volume"]] = df_new[
                ["open", "high", "low", "close", "volume"]
            ].astype(float)
            df_new = df_new[["open", "high", "low", "close", "volume"]]

        if df_existing is not None and not df_existing.empty:
            df_all = pd.concat([df_existing, df_new])
        else:
            df_all = df_new

        df_all = df_all.sort_index()
        df_all = df_all[~df_all.index.duplicated(keep="last")]

    df_all.to_csv(BINANCE_MARKET_FULL_CSV, index_label="timestamp")
    print(f"Binance 1h history mentve: {BINANCE_MARKET_FULL_CSV}, shape: {df_all.shape}")
    return df_all


# ---------- Összefésülés: Kaggle + Binance -> market_data_full ----------

def build_market_data_full() -> pd.DataFrame:
    """
    Kaggle + Binance 1H adatból előállítja a MARKET_DATA_FULL_CSV-t.

    Robusztus viselkedés:
    - Ha a Kaggle fájl hiányzik/hibás, akkor is frissíti a Binance history-t,
      és market_data_full.csv-t Binance-only alapon állít elő.
    """

    try:
        kaggle_1h = load_kaggle_bitcoin_1h()
    except Exception as e:
        print(f"Kaggle betöltés hiba ({e}) – folytatjuk Binance-only módban.")
        kaggle_1h = pd.DataFrame()

    binance_1h = update_binance_history_1h()

    if kaggle_1h is not None and not kaggle_1h.empty:
        combined = pd.concat([kaggle_1h, binance_1h])
    else:
        combined = binance_1h

    combined = combined.sort_index()
    # duplák eltávolítása – Binance adat legyen előnyben az átfedésben
    combined = combined[~combined.index.duplicated(keep="last")]

    print("Összesített market 1h shape:", combined.shape)
    MARKET_DATA_FULL_CSV.parent.mkdir(exist_ok=True, parents=True)
    combined.to_csv(MARKET_DATA_FULL_CSV, index_label="timestamp")
    print(f"Mentve: {MARKET_DATA_FULL_CSV}")
    return combined


if __name__ == "__main__":
    build_market_data_full()
