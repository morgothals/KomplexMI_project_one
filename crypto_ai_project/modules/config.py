# modules/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RUNTIME_DIR = DATA_DIR / "runtime"

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, RUNTIME_DIR]:
    d.mkdir(exist_ok=True, parents=True)

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# ---------- Fájlok ----------

# Régebbi, inkrementális OHLCV (pl. dashboardhoz)
MARKET_DATA_CSV = DATA_DIR / "market_data.csv"

# Teljes, többéves történelmi OHLCV (Kaggle + Binance bootstrap)
MARKET_DATA_FULL_CSV = PROCESSED_DIR / "market_data_full.csv"

# Kaggle bitcoin CSV (ezt neked kell letölteni és ide tenni)
# pl. data/raw/bitcoin_kaggle.csv
KAGGLE_MARKET_CSV = RAW_DIR / "bitcoin_kaggle.csv"

# Technikai indikátorokkal bővített market features (1H)
MARKET_FEATURES_CSV = PROCESSED_DIR / "market_data_features.csv"

# Rövid távú összevont feature-k (market+onchain+macro+sentiment) – ha használod
ALL_FEATURES_CSV = PROCESSED_DIR / "all_features.csv"



# Hosszú távú training sentiment store (napi aggregált)
TRAINING_SENTIMENT_FEATURES_CSV = PROCESSED_DIR / "training_sentiment_features.csv"

TRAINING_FEATURES_CSV = PROCESSED_DIR / "training_features_1h.csv"
BINANCE_MARKET_FULL_CSV = PROCESSED_DIR / "binance_market_1h.csv"

MARKET_INTRADAY_1M_CSV = RUNTIME_DIR / "market_intraday_1m.csv"

# On-chain, makró, rövid távú sentiment (dashboardnak)
ONCHAIN_DATA_CSV = PROCESSED_DIR / "onchain_data.csv"
MACRO_DATA_CSV = PROCESSED_DIR / "macro_data.csv"
SENTIMENT_DATA_CSV = PROCESSED_DIR / "sentiment_data.csv"

# Rövid távú raw hírek: csak 30 nap, LLM + dashboard
NEWS_DATA_CSV = RUNTIME_DIR / "news_data_30d.csv"

# Modellfájlok

FORECAST_MODEL_PATH = BASE_DIR / "models" / "forecast_model.keras"
FORECAST_SCALER_PATH = MODELS_DIR / "forecast_scaler.pkl"

# ---------- Crypto beállítások ----------

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LOOKBACK = 60  # LSTM ablak

BINANCE_BASE_URL = "https://api.binance.com"
FEAR_GREED_API_URL = "https://api.alternative.me/fng/"

# On-chain alternatívák
BLOCKCHAIR_STATS_URL = "https://api.blockchair.com/bitcoin/stats"
BLOCKCHAIN_CHARTS_BASE = "https://api.blockchain.info/charts"

# Crypto hírek
COINDESK_RSS_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"
REDDIT_CRYPTO_RSS_URL = "https://www.reddit.com/r/CryptoCurrency/.rss"
COINTELEGRAPH_TAG_URLS = [
    "https://cointelegraph.com/tags/markets",
    "https://cointelegraph.com/tags/bitcoin",
]

# Makró tickerek (pl. S&P 500, DXY)
YF_TICKERS = [
    "^GSPC",      # S&P 500
    "DX-Y.NYB",   # Dollar Index (ellenőrizd nálad, mi a pontos ticker)
]

CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")  # ha később használod
