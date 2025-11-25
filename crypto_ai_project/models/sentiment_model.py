import os
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# Szövegtisztítás
URL_RE = re.compile(r'https?://\S+|www\.\S+')
HTML_RE = re.compile(r'<.*?>')
NON_PRINT_RE = re.compile(r'[^\x20-\x7E\n]')

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    t = URL_RE.sub(' ', text)
    t = HTML_RE.sub(' ', t)
    t = NON_PRINT_RE.sub(' ', t)
    t = t.replace('\n', ' ').replace('\r', ' ')
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def score_news(df_news: pd.DataFrame) -> pd.DataFrame:
    """
    Csak a sentiment adatokat adja vissza:
    - datetime
    - symbol
    - title
    - sentiment_score
    - sentiment_label
    """
    df_news = df_news.copy()
    required_cols = {'datetime', 'symbol', 'title'}
    if not required_cols.issubset(df_news.columns):
        raise ValueError(f'df_news must contain columns: {required_cols}')

    # Hiányzó értékek kezelése
    df_news['title'] = df_news['title'].fillna('').astype(str)

    analyzer = SentimentIntensityAnalyzer()
    scores = []
    labels = []

    for _, row in tqdm(df_news.iterrows(), total=len(df_news), desc="Scoring news"):
        text = clean_text(row['title'])
        score = analyzer.polarity_scores(text)['compound']
        if score >= 0.05:
            label = 'positive'
        elif score <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        scores.append(score)
        labels.append(label)

    df_news['sentiment_score'] = scores
    df_news['sentiment_label'] = labels

    return df_news[['datetime', 'symbol', 'title', 'sentiment_score', 'sentiment_label']]

def process_market_and_news(news_csv: str, market_csv: str, out_csv: str) -> None:
    """
    Beolvassa a híreket és a piaci adatokat.
    Csak a sentiment adatokat menti ki.
    """
    # Hírek beolvasása
    news_df = pd.read_csv(news_csv)

    # Piaci adatok beolvasása (csak felhasználásra)
    market_df = pd.read_csv(market_csv)
    # (itt lehetne pl. szűrni, összevonni, de most csak beolvassuk)

    # Sentiment kiértékelés
    sentiment_df = score_news(news_df)

    # Mentés
    sentiment_df.to_csv(out_csv, index=False)
    print(f"Sentiment CSV saved to {out_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score news CSV and output sentiment data")
    parser.add_argument("--news", required=True, help="Input CSV file with news")
    parser.add_argument("--market", required=True, help="Input CSV file with market data (used but not in output)")
    parser.add_argument("--out", required=True, help="Output CSV file for sentiment results")
    args = parser.parse_args()

    process_market_and_news(args.news, args.market, args.out)
