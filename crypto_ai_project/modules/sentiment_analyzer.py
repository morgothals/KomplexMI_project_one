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
    NEWS_ALLTIME_CSV,
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


def fetch_fear_and_greed_history(days: int = 0) -> pd.DataFrame:
    """
    Fear & Greed index az utolsó 'days' napra az API-ból.
    Visszatérés: index = date (tz-naiv), oszlop: fear_greed
    """
    params = {"format": "json", "limit": days}
    r = requests.get(FEAR_GREED_API_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()["data"]

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()

    df["fear_greed"] = df["value"].astype(int)

    # napi resample, ha naponta több adat lenne
    df_daily = df[["fear_greed"]].resample("1D").last()
    df_daily.index = df_daily.index.tz_convert(None).floor("D")
    df_daily.index.name = "date"

    # csak előre töltünk (ugyanazon intervallumon belül), NEM visszafelé 2012-ig
    df_daily["fear_greed"] = df_daily["fear_greed"].ffill()
    return df_daily

def analyze_news_sentiment(df_news: pd.DataFrame) -> pd.DataFrame:
    if df_news.empty:
        return df_news

    def score_row(row):
        title = row.get("title")
        summary = row.get("summary")

        # NaN / None → "" és mindenből stringet csinálunk
        if pd.isna(title):
            title = ""
        else:
            title = str(title)

        if pd.isna(summary):
            summary = ""
        else:
            summary = str(summary)

        text = title + " " + summary
        vs = analyzer.polarity_scores(text)
        return vs["compound"]

    df = df_news.copy()
    df["sentiment"] = df.apply(score_row, axis=1)
    return df

def build_sentiment_timeseries() -> pd.DataFrame:
    """
    - update_news_store() továbbra is frissíti a NEWS_DATA_CSV-t.
    - Biztos sentiment pontok:
        * Hosszú múlt: news_alltime.csv (compute_alltime_sentiment_points) → news_sentiment
        * Friss napok: update_news_store() → build_recent_news_sentiment_from_store → news_sentiment + std + ratio
    - news_sentiment: a két forrás pontjait összefűzzük, majd minden napra (min..ma) lineárisan interpolálunk.
    - bullish_ratio / bearish_ratio:
        * CSAK a friss napokra (ahol több cikk van) maradnak meg.
        * Az alltime + interpolált napokon NaN, nem 0.
    - Fear & Greed: utolsó ~60 nap, FNG API-ból.
    - TRAINING_SENTIMENT_FEATURES_CSV:
        * régi adatokat betöltjük, ratio-kat eldobjuk,
        * új idősorral összefűzzük,
        * index szerint deduplikálunk (új értékek felülírják a régieket),
        * így a korábbi „mindenhol 0” ratio-k is eltűnnek.
    - SENTIMENT_DATA_CSV:
        * az utolsó 60 nap: timestamp, news_sentiment, fear_greed
    """
    # 0) Friss hírek store (RSS)
    df_news = update_news_store()

    # 1) Alltime backbone pontok
    df_alltime_points = compute_alltime_sentiment_points()  # index=date, col=news_sentiment

    # 2) Friss napok statisztikái (itt vannak ratio-k)
    df_recent_stats = build_recent_news_sentiment_from_store(df_news)  # index=date

    # 3) Összes biztos sentiment pont összefésülése (csak news_sentiment!)
    point_dfs = []
    if not df_alltime_points.empty:
        point_dfs.append(df_alltime_points[["news_sentiment"]])
    if not df_recent_stats.empty:
        point_dfs.append(df_recent_stats[["news_sentiment"]])

    if not point_dfs:
        print("Nincs egyetlen biztos sentiment pont sem (se alltime, se recent).")
        return pd.DataFrame()

    df_points = pd.concat(point_dfs, axis=0)
    # ha ugyanarra a napra több pont van (alltime + recent), a FRISS írja felül
    df_points = df_points[~df_points.index.duplicated(keep="last")]
    df_points = df_points.sort_index()

    # 4) Teljes napi idősor: min(pont) .. ma
    today = datetime.now().date()
    full_index = pd.date_range(df_points.index.min(), today, freq="1D")

    df_daily = df_points.reindex(full_index)
    df_daily.index.name = "date"

    # 5) Lineáris interpoláció: két legközelebbi biztos pont között
    df_daily["news_sentiment"] = df_daily["news_sentiment"].interpolate(
        method="time", limit_direction="both"
    )

    # 6) Friss napokra visszahozzuk a std + ratio értékeket
    if not df_recent_stats.empty:
        df_daily = df_daily.join(
            df_recent_stats[["news_sentiment_std", "bullish_ratio", "bearish_ratio"]],
            how="left"
        )
    else:
        df_daily["news_sentiment_std"] = 0.0
        df_daily["bullish_ratio"] = pd.NA
        df_daily["bearish_ratio"] = pd.NA

    # ahol nincs friss napi std, legyen 0.0
    df_daily["news_sentiment_std"] = df_daily["news_sentiment_std"].fillna(0.0)

    # 🔴 FIGYELEM: ratio-kat NEM töltjük ki, maradjanak NaN a nem-friss napokon!
    # df_daily["bullish_ratio"] és df_daily["bearish_ratio"] → csak recent indexen nem NaN

    # 7) Fear & Greed (~60 nap)
    try:
        df_fng_daily = fetch_fear_and_greed_history(days=60)  # index=date
    except Exception as e:
        print(f"Fear&Greed history lekérése nem sikerült: {e}")
        df_fng_daily = pd.DataFrame(columns=["fear_greed"])

    if not df_fng_daily.empty:
        df_daily = df_daily.join(df_fng_daily, how="left")
    else:
        df_daily["fear_greed"] = None

    # 8) index → timestamp (UTC)
    df_daily = df_daily.sort_index()
    df_daily.index = df_daily.index.tz_localize("UTC")
    df_daily.index.name = "timestamp"

    # 9) Régi training store betöltése, ratio-k kidobása
    try:
        df_old = pd.read_csv(TRAINING_SENTIMENT_FEATURES_CSV, parse_dates=["timestamp"])
        df_old = df_old.set_index("timestamp").sort_index()
        # csak azokat az oszlopokat tartjuk, amik nem ratio-k
        keep_cols = [c for c in df_old.columns if c not in ("bullish_ratio", "bearish_ratio")]
        df_old = df_old[keep_cols]
        print(f"Régi training_sentiment store shape (ratio nélkül): {df_old.shape}")
    except FileNotFoundError:
        df_old = pd.DataFrame()
        print("Nincs korábbi training_sentiment store, új fájl lesz.")
    except EmptyDataError:
        df_old = pd.DataFrame()
        print("Korábbi training_sentiment store üres, újraépítjük.")

    # 10) Új hosszú idősor: régi + új, index alapján dedup (új érték felülír)
    if not df_old.empty:
        df_long = pd.concat([df_old, df_daily], axis=0)
        df_long = df_long[~df_long.index.duplicated(keep="last")]
    else:
        df_long = df_daily

    df_long = df_long.sort_index()

    TRAINING_SENTIMENT_FEATURES_CSV.parent.mkdir(exist_ok=True, parents=True)
    df_long.to_csv(TRAINING_SENTIMENT_FEATURES_CSV, index_label="timestamp")
    print(
        f"Training_sentiment_features mentve: {TRAINING_SENTIMENT_FEATURES_CSV}, "
        f"shape: {df_long.shape}"
    )

    # 11) Rövid távú idősor a dashboardnak – utolsó 60 nap
    cutoff = datetime.now(timezone.utc) - timedelta(days=60)
    df_short = df_long[df_long.index >= cutoff].copy()

    df_short_export = df_short[["news_sentiment", "fear_greed"]].reset_index()
    df_short_export.to_csv(SENTIMENT_DATA_CSV, index=False)
    print(
        f"Rövid távú sentiment mentve (60 nap): {SENTIMENT_DATA_CSV}, "
        f"shape: {df_short_export.shape}"
    )

    return df_short


def build_news_sentiment_from_alltime_csv() -> pd.DataFrame:
    """
    Az új data/raw/news_alltime.csv alapján épít egy NAPI news_sentiment idősor history-t.

    CSV elvárt formátuma:
        date, news

    - lehet több sor UGYANAZZAL a dátummal -> ezeket napi szinten átlagoljuk.
    - hiányzó napokra: lineáris interpoláció a két legközelebbi biztos pont között.
    """
    try:
        df = pd.read_csv(NEWS_ALLTIME_CSV)
    except FileNotFoundError:
        raise RuntimeError(f"NEWS_ALLTIME_CSV nem található: {NEWS_ALLTIME_CSV}")
    except EmptyDataError:
        raise RuntimeError(f"NEWS_ALLTIME_CSV üres: {NEWS_ALLTIME_CSV}")

    if "date" not in df.columns or "news" not in df.columns:
        raise RuntimeError("news_alltime.csv-nek legalább 'date' és 'news' oszlopot kell tartalmaznia.")

    # dátum parse + napra kerekítés
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        raise RuntimeError("news_alltime.csv-ben minden 'date' érvénytelennek bizonyult.")

    df["date"] = df["date"].dt.floor("D")

    # VADER sentiment minden sorra
    def score_text(txt: str) -> float:
        if pd.isna(txt):
            txt = ""
        else:
            txt = str(txt)
        vs = analyzer.polarity_scores(txt)
        return vs["compound"]

    df["news_sentiment"] = df["news"].apply(score_text)

    # 🔹 NAPI aggregálás: ha ugyanarra a napra több sor van, átlagoljuk
    grouped = df.groupby("date")

    df_daily = pd.DataFrame(index=grouped.size().index)
    df_daily["news_sentiment"] = grouped["news_sentiment"].mean()
    df_daily["news_sentiment_std"] = grouped["news_sentiment"].std().fillna(0.0)

    df_daily["bullish_ratio"] = grouped["news_sentiment"].agg(lambda s: (s > 0).mean())
    df_daily["bearish_ratio"] = grouped["news_sentiment"].agg(lambda s: (s < 0).mean())

    df_daily.index.name = "date"
    df_daily = df_daily.sort_index()

    # teljes napi intervallum (min..max)
    full_index = pd.date_range(
        df_daily.index.min(),
        df_daily.index.max(),
        freq="1D"
    )

    # itt már NINCS duplikált index, nyugodtan reindexelhetünk
    df_daily = df_daily.reindex(full_index)
    df_daily.index.name = "date"

    # hiányzó napok: lineáris interpoláció idő szerint a news_sentimentre
    df_daily["news_sentiment"] = df_daily["news_sentiment"].interpolate(
        method="time", limit_direction="both"
    )

    # ahol std / bullish / bearish hiányzott az interpoláció miatt, nullázzuk / számoljuk ésszerűen
    df_daily["news_sentiment_std"] = df_daily["news_sentiment_std"].fillna(0.0)

    # bullish/bearish – itt már naponta 1 értékünk van → arány = 0 vagy 1
    df_daily["bullish_ratio"] = (df_daily["news_sentiment"] > 0).astype(float)
    df_daily["bearish_ratio"] = (df_daily["news_sentiment"] < 0).astype(float)

    return df_daily[["news_sentiment", "news_sentiment_std", "bullish_ratio", "bearish_ratio"]]


def build_recent_news_sentiment_from_store(df_news: pd.DataFrame) -> pd.DataFrame:
    """
    update_news_store() által adott df_news-ből (utolsó ~30 nap) napi aggregált sentiment.
    Vissza: index = date (tz-naiv), oszlopok:
      - news_sentiment
      - news_sentiment_std
      - bullish_ratio
      - bearish_ratio
    (itt van értelme a ratio-knak, mert több cikk/napi sor van)
    """
    if df_news.empty:
        return pd.DataFrame(
            columns=["news_sentiment", "news_sentiment_std", "bullish_ratio", "bearish_ratio"]
        )

    if "timestamp" not in df_news.columns:
        raise RuntimeError("df_news nem tartalmaz 'timestamp' oszlopot.")

    df_news = df_news.copy()
    df_news["timestamp"] = pd.to_datetime(df_news["timestamp"], errors="coerce", utc=True)
    df_news = df_news.dropna(subset=["timestamp"])
    if df_news.empty:
        return pd.DataFrame(
            columns=["news_sentiment", "news_sentiment_std", "bullish_ratio", "bearish_ratio"]
        )

    df_scored = analyze_news_sentiment(df_news)

    # tz-aware → tz-naiv, napra kerekítve
    df_scored["date"] = df_scored["timestamp"].dt.tz_convert(None).dt.floor("D")

    grouped = df_scored.groupby("date")

    df_daily = pd.DataFrame(index=grouped.size().index)
    df_daily["news_sentiment"] = grouped["sentiment"].mean()
    df_daily["news_sentiment_std"] = grouped["sentiment"].std().fillna(0.0)
    df_daily["bullish_ratio"] = grouped["sentiment"].agg(lambda s: (s > 0).mean())
    df_daily["bearish_ratio"] = grouped["sentiment"].agg(lambda s: (s < 0).mean())

    df_daily.index.name = "date"
    df_daily = df_daily.sort_index()
    return df_daily


def compute_alltime_sentiment_points() -> pd.DataFrame:
    """
    news_alltime.csv -> biztos pontok (ritkább, pl. havi) sentiment értékkel.
    Vissza: index = date (tz-naiv), oszlop: news_sentiment
    """
    try:
        df = pd.read_csv(NEWS_ALLTIME_CSV)
    except FileNotFoundError:
        print(f"NEWS_ALLTIME_CSV nem található: {NEWS_ALLTIME_CSV}")
        return pd.DataFrame(columns=["news_sentiment"])
    except EmptyDataError:
        print(f"NEWS_ALLTIME_CSV üres: {NEWS_ALLTIME_CSV}")
        return pd.DataFrame(columns=["news_sentiment"])

    if "date" not in df.columns or "news" not in df.columns:
        raise RuntimeError("news_alltime.csv-nek legalább 'date' és 'news' oszlopot kell tartalmaznia.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        print("news_alltime.csv-ben minden 'date' érvénytelen.")
        return pd.DataFrame(columns=["news_sentiment"])

    df["date"] = df["date"].dt.floor("D")

    def score_text(txt: str) -> float:
        if pd.isna(txt):
            txt = ""
        else:
            txt = str(txt)
        vs = analyzer.polarity_scores(txt)
        return vs["compound"]

    df["news_sentiment"] = df["news"].apply(score_text)

    # ha egy napra több sor van, átlagoljuk
    df_daily = (
        df.groupby("date")["news_sentiment"]
        .mean()
        .to_frame("news_sentiment")
        .sort_index()
    )
    df_daily.index.name = "date"
    return df_daily