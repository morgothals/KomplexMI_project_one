# modules/sentiment_analyzer.py
import math
from datetime import datetime, timedelta, timezone
from pandas.errors import EmptyDataError
import requests
import pandas as pd
import feedparser
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .config import (
    COINDESK_RSS_URL,
    REDDIT_CRYPTO_RSS_URL,
    COINTELEGRAPH_TAG_URLS,
    FEAR_GREED_API_URL,
    NEWS_DATA_CSV,
    SENTIMENT_DATA_CSV,
    TRAINING_SENTIMENT_FEATURES_CSV,
    DATA_DIR,
)

DATA_DIR.mkdir(exist_ok=True, parents=True)

analyzer = SentimentIntensityAnalyzer()


# ---------- Segédfüggvények ----------

def _now_utc():
    return datetime.now(timezone.utc)


def _one_month_ago():
    return _now_utc() - timedelta(days=30)


# ---------- RSS alapú hírek ----------

def fetch_coindesk_rss(limit=100) -> pd.DataFrame:
    """
    CoinDesk összes hír RSS-ből. :contentReference[oaicite:7]{index=7}
    """
    feed = feedparser.parse(COINDESK_RSS_URL)
    rows = []
    for entry in feed.entries[:limit]:
        # published_parsed lehet None, ezért fallback
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        else:
            ts = _now_utc()
        rows.append(
            {
                "timestamp": ts,
                "source": "coindesk",
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "url": entry.get("link", ""),
            }
        )
    return pd.DataFrame(rows)


def fetch_reddit_crypto_rss(limit=100) -> pd.DataFrame:
    """
    Reddit r/CryptoCurrency RSS. Friss posztok. 
    """
    feed = feedparser.parse(REDDIT_CRYPTO_RSS_URL)
    rows = []
    for entry in feed.entries[:limit]:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        else:
            ts = _now_utc()
        rows.append(
            {
                "timestamp": ts,
                "source": "reddit_CryptoCurrency",
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "url": entry.get("link", ""),
            }
        )
    return pd.DataFrame(rows)


# ---------- Cointelegraph HTML parsolás ----------

def _parse_cointelegraph_relative_date(text: str) -> datetime:
    """
    Cointelegraph list oldalon:
      - '3 hours ago'
      - 'Nov 27, 2025'
    Mindkettőt megpróbáljuk kezelni.
    """
    text = text.strip()
    now = _now_utc()

    #  '3 hours ago', '15 minutes ago'
    parts = text.split()
    if len(parts) >= 3 and parts[-1].lower() == "ago":
        try:
            value = int(parts[0])
            unit = parts[1].lower()
            if "hour" in unit:
                return now - timedelta(hours=value)
            if "minute" in unit:
                return now - timedelta(minutes=value)
            if "day" in unit:
                return now - timedelta(days=value)
        except ValueError:
            pass

    # Dátum formátum: 'Nov 27, 2025'
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            dt = datetime.strptime(text, fmt)
            # Cointelegraph nem ír időt, tekintsük UTC fél napnak
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    # fallback: most
    return now


def fetch_cointelegraph_tag_page(tag_url: str) -> pd.DataFrame:
    """
    Egyszerű scraper Cointelegraph tag oldalakról (markets, bitcoin). :contentReference[oaicite:8]{index=8}

    Az oldal HTML-je változhat a jövőben, ez egy best-effort parser:
    - Keressük azokat a részeket, ahol link + időpont (pl. '3 hours ago' / 'Nov 27, 2025') egymás közelében van.
    """
    r = requests.get(tag_url, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    rows = []

    # Observáció: a hírek címei <a> tag-ekben vannak, a környezetükben szerepel
    # a relatív/dátum string (pl. '3 hours ago').
    # Itt egyszerűen azokat az <a>-kat nézzük, amik a fő contentben vannak, és
    # utána keresünk egy "idő" jellegű szöveget a szomszédos elemekben.
    main = soup  # ha később szűkíteni akarod, megkereshetsz egy konténert

    for link in main.find_all("a", href=True):
        title = link.get_text(strip=True)
        href = link["href"]
        if not title or not href:
            continue
        if not href.startswith("http"):
            href = "https://cointelegraph.com" + href

        # kis heuristika: csak olyan link, ami "news" vagy "analysis" stb.
        if "/news/" not in href and "/markets/" not in href and "/bitcoin/" not in href:
            continue

        # keressünk idő szöveget a következő testvér/elem környékén
        date_text = None
        # nézzük a szülő környékét
        parent = link.parent
        for sibling in parent.next_siblings:
            if hasattr(sibling, "get_text"):
                txt = sibling.get_text(strip=True)
            else:
                txt = str(sibling).strip()
            if not txt:
                continue
            # heuristika: 'ago' vagy ',' (dátumszerű)
            if "ago" in txt.lower() or "," in txt:
                date_text = txt
                break

        if not date_text:
            # fallback: próbáljuk a parent korábbi tag-jeit
            for sibling in parent.previous_siblings:
                if hasattr(sibling, "get_text"):
                    txt = sibling.get_text(strip=True)
                else:
                    txt = str(sibling).strip()
                if not txt:
                    continue
                if "ago" in txt.lower() or "," in txt:
                    date_text = txt
                    break

        ts = _parse_cointelegraph_relative_date(date_text or "")

        rows.append(
            {
                "timestamp": ts,
                "source": "cointelegraph",
                "title": title,
                "summary": "",
                "url": href,
            }
        )

    df = pd.DataFrame(rows)
    # Szűrés: csak 1 hónapon belüli hírek
    if not df.empty:
        df = df[df["timestamp"] >= _one_month_ago()]
    return df


def fetch_cointelegraph_all_tags() -> pd.DataFrame:
    dfs = []
    for url in COINTELEGRAPH_TAG_URLS:
        try:
            df = fetch_cointelegraph_tag_page(url)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Cointelegraph hiba ({url}):", e)
    if not dfs:
        return pd.DataFrame(columns=["timestamp", "source", "title", "summary", "url"])
    df_all = pd.concat(dfs, axis=0)
    # duplikált cikkek kiszűrése (URL alapján)
    df_all = df_all.drop_duplicates(subset=["url"])
    return df_all


# ---------- Hírek egyesítése + tárolása 1 hónapig ----------

def update_news_store() -> pd.DataFrame:
    """
    Hírek összegyűjtése:
      - CoinDesk RSS
      - Reddit r/CryptoCurrency RSS
      - Cointelegraph tags (markets, bitcoin)
    + Deduplikálás URL szerint
    + Csak az utolsó 30 nap marad meg
    + Mentés: NEWS_DATA_CSV
    """
    # --- RÉGI CSV BIZTONSÁGOS BEOLVASÁSA ---
    try:
        df_old = pd.read_csv(NEWS_DATA_CSV)
    except FileNotFoundError:
        print("NEWS_DATA_CSV nem létezik, új hír-adatbázist hozunk létre.")
        df_old = pd.DataFrame(columns=["timestamp", "source", "title", "summary", "url"])
    except EmptyDataError:
        print("NEWS_DATA_CSV üres, újraépítjük a hír-adatbázist.")
        df_old = pd.DataFrame(columns=["timestamp", "source", "title", "summary", "url"])
    except Exception as e:
        print(f"NEWS_DATA_CSV beolvasási hiba ({e}), újraépítjük a hír-adatbázist.")
        df_old = pd.DataFrame(columns=["timestamp", "source", "title", "summary", "url"])

    # timestamp oszlop normalizálása
    if not df_old.empty:
        if "timestamp" in df_old.columns:
            df_old["timestamp"] = pd.to_datetime(df_old["timestamp"], errors="coerce", utc=True)
            df_old = df_old.dropna(subset=["timestamp"])
        else:
            print("NEWS_DATA_CSV nem tartalmaz 'timestamp' oszlopot, eldobjuk a régi adatot.")
            df_old = pd.DataFrame(columns=["timestamp", "source", "title", "summary", "url"])

    dfs_new = []

    try:
        df_cd = fetch_coindesk_rss(limit=100)
        dfs_new.append(df_cd)
    except Exception as e:
        print("CoinDesk RSS hiba:", e)

    try:
        df_rd = fetch_reddit_crypto_rss(limit=100)
        dfs_new.append(df_rd)
    except Exception as e:
        print("Reddit RSS hiba:", e)

    try:
        df_ct = fetch_cointelegraph_all_tags()
        dfs_new.append(df_ct)
    except Exception as e:
        print("Cointelegraph hiba:", e)

    if dfs_new:
        df_new = pd.concat(dfs_new, axis=0)
    else:
        df_new = pd.DataFrame(columns=["timestamp", "source", "title", "summary", "url"])

    # Összefűzés régi + új
    df_all = pd.concat([df_old, df_new], axis=0, ignore_index=True)

    # időzónák egységesítése
    if not df_all.empty:
        if df_all["timestamp"].dtype == "datetime64[ns]":
            df_all["timestamp"] = df_all["timestamp"].dt.tz_localize("UTC")

    # duplikált URL-ek törlése
    if "url" in df_all.columns:
        df_all = df_all.drop_duplicates(subset=["url"], keep="last")

    # csak az utolsó 30 nap
    cutoff = _one_month_ago()
    df_all = df_all[df_all["timestamp"] >= cutoff]

    # rendezés idő szerint
    df_all = df_all.sort_values("timestamp")

    df_all.to_csv(NEWS_DATA_CSV, index=False)
    return df_all


# ---------- Fear&Greed + hírsentiment idősor ----------

def fetch_latest_fear_and_greed(limit=60) -> pd.DataFrame:
    """
    Fear & Greed index (utolsó 'limit' nap).
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


def analyze_news_sentiment(df_news: pd.DataFrame) -> pd.DataFrame:
    if df_news.empty:
        return df_news

    def score_row(row):
        text = (row.get("title") or "") + " " + (row.get("summary") or "")
        vs = analyzer.polarity_scores(text)
        return vs["compound"]

    df = df_news.copy()
    df["sentiment"] = df.apply(score_row, axis=1)
    return df


def build_sentiment_timeseries() -> pd.DataFrame:
    """
    - Frissíti a hírtárat (NEWS_DATA_CSV) -> az marad max 30 nap (raw hírek)
    - Hírekre VADER-rel sentimentet számol (article szinten)
    - Napi szinten aggregál:
        news_sentiment      = átlagos compound score
        news_sentiment_std  = szórás
        bullish_ratio       = arány, ahol sentiment > 0
        bearish_ratio       = arány, ahol sentiment < 0
    - Fear & Greed indexet hozzácsatolja (daily)
    - Rövid (~60 napos) idősor:
        -> SENTIMENT_DATA_CSV (timestamp, news_sentiment, fear_greed)
    - Hosszú távú training store:
        -> TRAINING_SENTIMENT_FEATURES_CSV
           (timestamp, news_sentiment, news_sentiment_std,
            bullish_ratio, bearish_ratio, fear_greed)
    """
    # 1) Rövid távú news store frissítése (max 30 nap)
    df_news = update_news_store()
    if df_news.empty:
        print("Nincs új hír, sentiment timeseries üres marad.")
        return pd.DataFrame()

    # timestamp biztos konvertálása
    if "timestamp" not in df_news.columns:
        raise RuntimeError("df_news nem tartalmaz 'timestamp' oszlopot, nem tudunk idősoros sentimentet építeni.")

    df_news["timestamp"] = pd.to_datetime(df_news["timestamp"], errors="coerce", utc=True)
    df_news = df_news.dropna(subset=["timestamp"])
    if df_news.empty:
        print("Minden hír timestamp-je érvénytelen lett, üres a df_news.")
        return pd.DataFrame()

    # 2) Hírsentiment kiszámítása VADER-rel cikkenként
    df_scored = analyze_news_sentiment(df_news)
    if "sentiment" not in df_scored.columns:
        # biztonsági fallback, de normálisan ide nem szabadna eljutni
        print("Figyelem: analyze_news_sentiment nem hozott létre 'sentiment' oszlopot, semlegesre állítjuk (0.0).")
        df_scored["sentiment"] = 0.0

    # 3) Napi aggregáció
    df_scored["date"] = df_scored["timestamp"].dt.floor("D")
    grouped = df_scored.groupby("date")

    df_daily = pd.DataFrame(index=grouped.size().index)
    df_daily["news_sentiment"] = grouped["sentiment"].mean()
    df_daily["news_sentiment_std"] = grouped["sentiment"].std().fillna(0.0)

    bullish = grouped["sentiment"].agg(lambda s: (s > 0).mean())
    bearish = grouped["sentiment"].agg(lambda s: (s < 0).mean())
    df_daily["bullish_ratio"] = bullish
    df_daily["bearish_ratio"] = bearish

    # 4) Fear & Greed index (utolsó ~90 nap), napi összehangolás
    try:
        df_fng = fetch_latest_fear_and_greed(limit=90)
        if not df_fng.empty:
            # napi resample: ha egy napon több pont lenne, az utolsót vesszük
            df_fng_daily = df_fng.resample("1D").last()
            df_fng_daily.index = df_fng_daily.index.floor("D")
            df_fng_daily.index.name = "date"
            # join date-indexen
            df_daily = df_daily.join(df_fng_daily, how="left")
        else:
            df_daily["fear_greed"] = None
    except Exception as e:
        print(f"Fear&Greed lekérése nem sikerült: {e}")
        df_daily["fear_greed"] = None

    # index átnevezése timestamp-re, hogy egységes legyen
    df_daily.index.name = "timestamp"
    df_daily = df_daily.sort_index()

    # 4) Hosszú távú training sentiment store frissítése (append + dedup)
    try:
        df_old = pd.read_csv(TRAINING_SENTIMENT_FEATURES_CSV, parse_dates=["timestamp"])
        df_old = df_old.set_index("timestamp").sort_index()
        print(f"Régi training_sentiment store shape: {df_old.shape}")
    except FileNotFoundError:
        df_old = pd.DataFrame()
        print("Nincs korábbi training_sentiment store, új fájl lesz létrehozva.")
    except EmptyDataError:
        df_old = pd.DataFrame()
        print("Korábbi training_sentiment store üres, újraépítjük.")

    if not df_old.empty:
        df_long = pd.concat([df_old, df_daily])
        # deduplikáljunk index alapján (ha ugyanarra a napra újraszámolunk)
        df_long = df_long[~df_long.index.duplicated(keep="last")]
    else:
        df_long = df_daily

    df_long = df_long.sort_index()
    TRAINING_SENTIMENT_FEATURES_CSV.parent.mkdir(exist_ok=True, parents=True)
    df_long.to_csv(TRAINING_SENTIMENT_FEATURES_CSV, index_label="timestamp")
    print(f"Training_sentiment_features mentve: {TRAINING_SENTIMENT_FEATURES_CSV}, shape: {df_long.shape}")

    # 5) Rövid távú (runtime) idősor – az EGÉSZ store-ból (ne csak az aktuális futás napi adataiból)
    cutoff = datetime.now(timezone.utc) - timedelta(days=60)
    df_short = df_long[df_long.index >= cutoff].copy()

    # -> sentiment_data.csv: timestamp, news_sentiment, fear_greed
    df_short_export = df_short[["news_sentiment", "fear_greed"]].reset_index()
    df_short_export.to_csv(SENTIMENT_DATA_CSV, index=False)
    print(f"Rövid távú sentiment mentve: {SENTIMENT_DATA_CSV}, shape: {df_short_export.shape}")

    return df_short