import csv
import random
from datetime import datetime, timedelta

# ----------------------------------------
# FULL COIN LIST
# ----------------------------------------

COINS_RAW = [
    "Aave", "BinanceCoin", "Bitcoin", "Cardano", "ChainLink",
    "Cosmos", "CryptocomCoin", "Dogecoin", "EOS", "Ethereum",
    "lota", "Litecoin", "Monero", "NEM", "Polkadot",
    "Solana", "Stellar", "Tether", "Tron", "Uniswap",
    "USDCoin", "WrappedBitcoin", "XRP"
]

# Convert names → SYMBOL-USD
COINS = [f"{c.upper()}-USD" for c in COINS_RAW]

NEWS_PER_COIN = 80
OUTPUT_FILE = "generated_news.csv"


# ----------------------------------------
# SOURCES
# ----------------------------------------

SOURCES = [
    "CoinDesk", "CoinTelegraph", "Yahoo Finance", "CryptoNews", "Decrypt",
    "The Block", "Bloomberg Crypto", "Forbes Crypto"
]

BASE_URL = "https://news.example.com/{coin}/{id}"


# ----------------------------------------
# TEMPLATES (domináns pozitív + negatív)
# ----------------------------------------

TITLE_TEMPLATES = [
    # Positive
    ("{coin} surges amid renewed market optimism", "positive"),
    ("{coin} adoption rises sharply as institutions enter", "positive"),
    ("Whales accumulate large amounts of {coin}", "positive"),
    ("{coin} ecosystem expands with new partnerships", "positive"),
    ("{coin} experiences strong bullish momentum", "positive"),
    ("Analysts predict upward movement for {coin}", "positive"),
    ("{coin} sees major spike in transaction activity", "positive"),
    ("Investors show strong confidence in {coin}", "positive"),
    ("{coin} shows a strong recovery after recent dip", "positive"),
    ("Liquidity for {coin} increases significantly", "positive"),
    ("{coin} gains traction in global markets", "positive"),
    ("{coin} receives strong community support", "positive"),
    ("New upgrade boosts performance of {coin}", "positive"),

    # Negative
    ("{coin} drops after heavy selling pressure", "negative"),
    ("Investors pull liquidity from {coin} markets", "negative"),
    ("{coin} faces short-term bearish correction", "negative"),
    ("Regulatory pressure negatively impacts {coin}", "negative"),
    ("Whales reduce exposure to {coin}", "negative"),
    ("{coin} experiences unusual volatility spike", "negative"),
    ("Market uncertainty pushes {coin} downward", "negative"),
    ("Technical indicators show weakness for {coin}", "negative"),
    ("{coin} underperforms compared to competitors", "negative"),
    ("{coin} sees declining trading activity", "negative"),
    ("Negative sentiment grows around {coin}", "negative"),

    # Neutral (kevés)
    ("{coin} trades sideways with minimal volatility", "neutral"),
    ("Market activity around {coin} remains stable", "neutral"),
    ("{coin} shows no major price movement today", "neutral"),
    ("{coin} experiences a calm trading session", "neutral"),
]


SUMMARY_TEMPLATES = {
    "positive": [
        "{coin} benefited from increased market confidence.",
        "Analysts highlight rising adoption of {coin}.",
        "Technical indicators show strong performance for {coin}.",
        "Investors remain optimistic about {coin}'s long-term potential.",
        "{coin} saw improved trading activity across major exchanges.",
        "Market sentiment surrounding {coin} is highly positive.",
    ],
    "negative": [
        "{coin} faced strong selling pressure today.",
        "Market uncertainty negatively affected {coin}.",
        "{coin} experienced reduced liquidity during the session.",
        "Traders expressed concern over {coin}'s performance.",
        "Bearish indicators appeared for {coin}.",
        "{coin} continues to face downward pressure.",
    ],
    "neutral": [
        "{coin} showed steady performance with few major events.",
        "Traders reported balanced activity for {coin}.",
        "{coin} remained stable throughout the trading period.",
    ]
}


# ----------------------------------------
# GENERATOR
# ----------------------------------------

def generate_news():
    rows = []

    for symbol in COINS:
        coin = symbol.replace("-USD", "")

        for i in range(NEWS_PER_COIN):

            title, sentiment = random.choices(
                TITLE_TEMPLATES,
                weights=[5 if s == "positive" else 5 if s == "negative" else 1
                         for (_, s) in TITLE_TEMPLATES],
                k=1
            )[0]

            title = title.format(coin=coin)
            summary = random.choice(SUMMARY_TEMPLATES[sentiment]).format(coin=coin)
            source = random.choice(SOURCES)

            dt = (datetime.utcnow() - timedelta(minutes=random.randint(0, 200000))).isoformat() + "Z"

            url = BASE_URL.format(coin=coin.lower(), id=random.randint(10000, 99999))

            rows.append([dt, coin, source, title, summary, url, sentiment])

    return rows


def save_csv(rows):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime", "symbol", "source", "title", "summary", "url", "sentiment"])
        writer.writerows(rows)

    print(f"Generated {len(rows)} news rows → saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    data = generate_news()
    save_csv(data)
