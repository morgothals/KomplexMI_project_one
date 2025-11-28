# modules/data_collector.py
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

import yfinance as yf

from .config import (
    BINANCE_BASE_URL,
    SYMBOL,
    INTERVAL,
    MARKET_DATA_CSV,
    FEAR_GREED_API_URL,
    BLOCKCHAIR_STATS_URL,
    BLOCKCHAIN_CHARTS_BASE,
    ONCHAIN_DATA_CSV,
    MACRO_DATA_CSV,
    DATA_DIR,
    YF_TICKERS,
    MARKET_INTRADAY_1M_CSV,
)

DATA_DIR.mkdir(exist_ok=True, parents=True)


# ------------ BINANCE OHLCV ------------

def fetch_binance_klines(symbol=SYMBOL, interval=INTERVAL, limit=1000,
                         start_time=None, end_time=None) -> pd.DataFrame:
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_time is not None:
        params["startTime"] = int(start_time.timestamp() * 1000)
    if end_time is not None:
        params["endTime"] = int(end_time.timestamp() * 1000)

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_blockchain_chart(chart_name: str, timespan: str = "all", rolling_average: str | None = None) -> pd.DataFrame:
    """
    Blockchain.com charts API:
      pl. https://api.blockchain.info/charts/n-transactions?timespan=all&format=json&sampled=false
    Visszaad egy DataFrame-et: index = timestamp (UTC), value oszlop.
    """
    url = f"https://api.blockchain.info/charts/{chart_name}"
    params = {
        "timespan": timespan,
        "format": "json",
        "sampled": "false",
    }
    if rolling_average:
        params["rollingAverage"] = rolling_average

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    values = data.get("values", [])
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    # x = unix timestamp, y = value
    df["timestamp"] = pd.to_datetime(df["x"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.rename(columns={"y": "value"})
    return df[["value"]]


def update_market_data_csv(symbol=SYMBOL, interval=INTERVAL) -> pd.DataFrame:
    """
    market_data.csv frissítése: ha létezik, az utolsó időponttól felfelé tölt.
    Ha nem létezik, vagy hibás a formátuma, lehúz egy nagyobb, mondjuk 1000-es blokkot.
    """
    existing = None
    start_time = None

    try:
        # Megpróbáljuk beolvasni a meglévő fájlt
        df = pd.read_csv(MARKET_DATA_CSV, parse_dates=["timestamp"])
        # Ha nincs timestamp oszlop, ValueError ment volna, de
        # ha valamiért mégis átjött, ellenőrizzük:
        if "timestamp" not in df.columns:
            raise ValueError("timestamp column missing in MARKET_DATA_CSV")

        # timezone: ha nincs tz, tegyük UTC-re
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

        existing = df.set_index("timestamp").sort_index()
        last_ts = existing.index.max()
        start_time = last_ts + pd.Timedelta(milliseconds=1)

    except FileNotFoundError:
        # első futás: nincs fájl → újra lehúzzuk
        print("MARKET_DATA_CSV nem létezik, új fájl lesz létrehozva.")
        existing = None
        start_time = None
    except ValueError as e:
        # régi/bontott/hibás fájl → inkább újrakezdjük
        print(f"MARKET_DATA_CSV formátum hiba ({e}), újraépítjük az adatfájlt.")
        existing = None
        start_time = None
    except Exception as e:
        # bármilyen egyéb hiba esetén is inkább nulláról kezdünk
        print(f"MARKET_DATA_CSV beolvasási hiba ({e}), újraépítjük az adatfájlt.")
        existing = None
        start_time = None

    # Új adatok lehúzása Binance-ről
    df_new = fetch_binance_klines(symbol=symbol, interval=interval,
                                  start_time=start_time)

    if existing is not None and not df_new.empty:
        combined = pd.concat([existing, df_new])
        combined = combined[~combined.index.duplicated(keep="last")]
    elif existing is None:
        combined = df_new
    else:
        combined = existing

    combined = combined.sort_index()
    combined.to_csv(MARKET_DATA_CSV, index_label="timestamp")
    return combined


# ------------ FEAR & GREED ------------

def fetch_fear_and_greed_history(limit=0) -> pd.DataFrame:
    """
    Fear & Greed index Alternative.me API-n keresztül (limit: hány rekord, 0 = all).
    """
    params = {"limit": limit, "format": "json"}
    r = requests.get(FEAR_GREED_API_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()["data"]

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df = df.set_index("timestamp")
    df["fear_greed"] = df["value"].astype(int)
    return df[["fear_greed"]]


# ------------ ON-CHAIN: BLOCKCHAIR + BLOCKCHAIN.COM ------------

def fetch_blockchair_stats() -> pd.DataFrame:
    """
    Egyszerű snapshot a Blockchair /bitcoin/stats API-ról. :contentReference[oaicite:4]{index=4}
    Ez inkább "mai" on-chain állapot, nem hosszú idősor.
    """
    r = requests.get(BLOCKCHAIR_STATS_URL, timeout=10)
    r.raise_for_status()
    data = r.json()["data"]

    now = datetime.now(timezone.utc)
    df = pd.DataFrame(
        {
            "timestamp": [now],
            "bc_tx_24h": [data.get("transactions_24h")],
            "bc_hashrate_24h": [data.get("hashrate_24h")],
            "bc_addresses_active_24h": [data.get("addresses_active_24h")],
        }
    ).set_index("timestamp")
    return df


def fetch_blockchain_chart(chart_name: str, timespan: str = "1month") -> pd.DataFrame:
    """
    Blockchain.com Charts API – pl. n-transactions, hash-rate, n-unique-addresses. :contentReference[oaicite:5]{index=5}
    """
    url = f"{BLOCKCHAIN_CHARTS_BASE}/{chart_name}"
    params = {"timespan": timespan, "format": "json"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    values = data.get("values", [])
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    df["timestamp"] = pd.to_datetime(df["x"], unit="s", utc=True)
    df = df.set_index("timestamp")
    df[chart_name] = df["y"].astype(float)
    return df[[chart_name]]


def update_onchain_data() -> pd.DataFrame:
    """
    On-chain mutatók frissítése a teljes bitcoin historyra (genesis óta).

    Blockchain.com charts API:
      - tx_count          -> n-transactions
      - active_addresses  -> n-unique-addresses
      - hash_rate         -> hash-rate
      - avg_block_size    -> avg-block-size
      - miners_revenue    -> miners-revenue

    Kimenet: napi (1d) idősor, timestamp (UTC) index, oszlopok:
      [tx_count, active_addresses, hash_rate, avg_block_size, miners_revenue]
    """
    print(">>> On-chain (Blockchain.com, teljes history) letöltés...")

    mapping = {
        "tx_count": "n-transactions",
        "active_addresses": "n-unique-addresses",
        "hash_rate": "hash-rate",
        "avg_block_size": "avg-block-size",
        "miners_revenue": "miners-revenue",
    }

    dfs = []
    for col_name, chart_name in mapping.items():
        print(f"  - {col_name} ({chart_name})...")
        df_chart = fetch_blockchain_chart(chart_name, timespan="all")
        if df_chart.empty:
            print(f"    Figyelem: {chart_name} üres adatot adott vissza.")
            continue
        df_chart = df_chart.rename(columns={"value": col_name})
        dfs.append(df_chart)

    if not dfs:
        print("Nem sikerült on-chain adatot lekérni, üres DataFrame-et adunk vissza.")
        return pd.DataFrame()

    # összejoinoljuk az összes chartot timestamp szerint
    df_onchain = dfs[0]
    for df in dfs[1:]:
        df_onchain = df_onchain.join(df, how="outer")

    df_onchain = df_onchain.sort_index()

    # mentsük CSV-be
    ONCHAIN_DATA_CSV.parent.mkdir(exist_ok=True, parents=True)
    df_onchain.to_csv(ONCHAIN_DATA_CSV, index_label="timestamp")
    print(f"On-chain shape (full history): {df_onchain.shape}")
    print(f"On-chain mentve ide: {ONCHAIN_DATA_CSV}")

    return df_onchain


# ------------ Yahoo Finance (makró) ------------

def update_macro_data() -> pd.DataFrame:
    """
    Makró mutatók teljes history-val (amennyit a Yahoo Finance ad):

      - S&P 500 index: ^GSPC
      - DXY (US Dollar Index): DX-Y.NYB  (ha ez nem működne, alternatíva: UUP ETF)

    Napi (1d) idősor, oszlopok:
      [sp500_close, dxy_close]
    """
    print(">>> Makró adatok (Yahoo Finance, teljes history) letöltése...")

    tickers = {
        "sp500_close": "^GSPC",
        "dxy_close": "DX-Y.NYB",
    }

    dfs = []
    for col_name, ticker in tickers.items():
        print(f"  - {col_name} ({ticker})...")
        data = yf.download(
            ticker,
            period="max",
            interval="1d",
            auto_adjust=False,    # explicit, hogy ne változzon viselkedés
            progress=False,
        )
        if data is None or data.empty:
            print(f"    Figyelem: {ticker} üres adatot adott vissza.")
            continue

        # Ha data egy Series lenne (ritka), csináljunk belőle DataFrame-et
        if isinstance(data, pd.Series):
            series = data.copy()
        else:
            # tipikus eset: DataFrame 'Adj Close' vagy 'Close' oszloppal
            if "Adj Close" in data.columns:
                series = data["Adj Close"].copy()
            elif "Close" in data.columns:
                series = data["Close"].copy()
            else:
                print(f"    Figyelem: {ticker} nem tartalmaz 'Adj Close' vagy 'Close' oszlopot.")
                continue

        # Most lehet, hogy még mindig DataFrame vagy Series, ezt kezeljük
        if isinstance(series, pd.DataFrame):
            df_t = series.copy()
            # ha több oszlop lenne, csak az elsőt használjuk
            if df_t.shape[1] > 1:
                first_col = df_t.columns[0]
                df_t = df_t[[first_col]]
            df_t.columns = [col_name]
        else:
            # Series -> 1 oszlopos DataFrame
            df_t = series.to_frame(name=col_name)

        # index: dátum (naive), tegyük UTC-re
        df_t.index = pd.to_datetime(df_t.index).tz_localize("UTC")
        dfs.append(df_t)

    if not dfs:
        print("Nem sikerült makró adatot lekérni, üres DataFrame-et adunk vissza.")
        return pd.DataFrame()

    df_macro = dfs[0]
    for df in dfs[1:]:
        df_macro = df_macro.join(df, how="outer")

    df_macro = df_macro.sort_index()

    MACRO_DATA_CSV.parent.mkdir(exist_ok=True, parents=True)
    df_macro.to_csv(MACRO_DATA_CSV, index_label="timestamp")
    print(f"Makró shape (full history): {df_macro.shape}")
    print(f"Makró mentve ide: {MACRO_DATA_CSV}")

    return df_macro


def _fetch_binance_1m_today(symbol=SYMBOL):
    """
    BTCUSDT 1 perces gyertyák lekérése a MAI napra (UTC-ben).
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"

    # Mai nap 00:00:00 UTC
    today_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_ms = int(today_utc.timestamp() * 1000)

    all_rows = []
    limit = 1000
    while True:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "limit": limit,
            "startTime": start_ms,
        }

        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break

        all_rows.extend(batch)
        last_open_time = batch[-1][0]

        # ha átléptük a következő nap kezdetét, vagy kevesebb mint 1000 jött, kilépünk
        tomorrow_utc = today_utc + timedelta(days=1)
        tomorrow_ms = int(tomorrow_utc.timestamp() * 1000)

        if last_open_time >= tomorrow_ms or len(batch) < limit:
            break

        start_ms = last_open_time + 1

    if not all_rows:
        return pd.DataFrame()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]

    df = pd.DataFrame(all_rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)

    df = df[["open", "high", "low", "close", "volume"]]
    return df


def update_intraday_minute_data(symbol=SYMBOL):
    """
    Aznapi (UTC) 1 perces BTCUSDT OHLCV frissítése.
    Minden futáskor LECSERÉLJÜK az aznapi 1m file-t (nem inkrementális, hanem 'full rewrite').
    """
    print(">>> Intraday (1m) Binance adatok frissítése a mai napra...")
    df_1m = _fetch_binance_1m_today(symbol=symbol)
    if df_1m.empty:
        print("Nem érkezett intraday 1m adat.")
    else:
        MARKET_INTRADAY_1M_CSV.parent.mkdir(exist_ok=True, parents=True)
        df_1m.to_csv(MARKET_INTRADAY_1M_CSV, index_label="timestamp")
        print(f"Intraday 1m shape: {df_1m.shape}, mentve ide: {MARKET_INTRADAY_1M_CSV}")
    return df_1m
