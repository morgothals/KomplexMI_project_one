Tökéletes — ez **nagyon jó, valóságos** és **tanárbarát** téma egy komplex MI iskolai projektre.
Az alapötlet kerek, és tényleg lehet belőle *narrow AI* rendszert csinálni, ahol több részmodul együtt dolgozik.
Mivel **4 ember** dolgozik rajta, az ideális felosztás a **rendszer architektúrája szerint** történik (nem lineárisan, hanem párhuzamos modulokban).

---

## 🧩 Összefoglaló architektúra

```
[Adatgyűjtés modul] → [Szövegelemző modell] → [Idősor-előrejelző modell] → [Tanácsadó (döntéstámogató) modul]
```

Minden modul önállóan fejleszthető, és később egy `main.py` vagy webes felület integrálja őket.

---

## 👥 Szereposztás és feladatok

### 🧑‍💻 **1. személy – Adatgyűjtés és API integráció (Data Engineer)**

**Cél:**
Automatikusan gyűjti a piaci adatokat és híreket.

**Feladatai:**

* Árfolyamadatok letöltése (Yahoo Finance, CoinGecko, Binance API, stb.)
* Kriptovaluta hírek, tweetek, vagy RSS feedek lekérése
* Időbélyegzett (timestampelt) adatok mentése CSV-be / SQLite-ba
* Adattisztítás (hiányzó értékek, duplikátumok kezelése)
* Adatelőkészítés az LSTM modellhez

**Kimenet:**

* `data/market_data.csv` (árfolyamok)
* `data/news_data.csv` (hírek szövege, forrás, dátum)

**Tech stack:**

* Python + `requests`, `pandas`, `yfinance`, `BeautifulSoup4`, `tweepy`

---

### 🤖 **2. személy – Szövegelemzés és hangulatelemzés (NLP specialist)**

**Cél:**
A hírek, tweetek és cikkek szövegének automatikus értelmezése.

**Feladatai:**

* Szöveg tisztítása (URL-ek, szimbólumok, tokenizálás)
* Sentiment-analízis (pozitív / negatív / semleges)
* Kulcsszavak kinyerése (pl. "ETF", "halving", "regulation")
* Modell választása:

  * Egyszerű: `TextBlob`, `VADER`
  * Haladó: `BERT`, `FinBERT`, `HuggingFace transformers`
* Az eredményt numerikus formában (pl. +1 / 0 / -1) menti el az árfolyam-adatok mellé

**Kimenet:**

* `data/sentiment_data.csv` (datetime, sentiment, source)

**Tech stack:**

* Python + `transformers`, `nltk`, `textblob`, `pandas`

---

### 📈 **3. személy – Idősor-előrejelzés (ML engineer / Data Scientist)**

**Cél:**
A piaci és hangulatadatok alapján előrejelzést adni az árfolyam irányára.

**Feladatai:**

* Az előkészített adatokból idősor (time series) létrehozása
* Feature engineering (pl. mozgóátlag, RSI, hangulat-index)
* Modell kiválasztása:

  * Alap: LSTM, GRU
  * Alternatíva: RandomForestRegressor vagy XGBoost
* Tanítás és tesztelés
* Modell mentése (`.h5` vagy `.pkl`)

**Kimenet:**

* `models/crypto_forecast_model.h5`
* `predictions/next_6h_forecast.csv`

**Tech stack:**

* Python + `tensorflow` / `keras` vagy `scikit-learn`, `matplotlib`

---

### 🧠 **4. személy – Tanácsadó és front-end integráció (AI logic & UI)**

**Cél:**
A rendszer eredményeit emberi nyelven értelmezhető módon tálalni.

**Feladatai:**

* Integrálni a három előző modult
* Betölti a legfrissebb árfolyamot, hírek hangulatát és a modell előrejelzését
* Összegzi az eredményt:

  * “A piaci hangulat pozitív → vétel ajánlott.”
  * “Negatív trend + rossz hírek → eladás ajánlott.”
* Egyszerű GUI vagy webes dashboard készítése:

  * `streamlit` / `gradio` / `flask` / `dash`
* Vizualizáció: trendgrafikon, hírek hangulata, model output

**Kimenet:**

* `main.py` vagy webapp
* Felhasználóbarát “AI Advisor” nézet

**Tech stack:**

* Python + `streamlit` vagy `flask`
* Frontendhez: `plotly`, `matplotlib`

---

## 🔄 Párhuzamos munkaszervezés

| Hét | Tevékenység                                                    | Résztvevők |
| --- | -------------------------------------------------------------- | ---------- |
| 1.  | Projekt setup (GitHub repo, mappastruktúra, API-k kipróbálása) | mindenki   |
| 2.  | Adatgyűjtés kódolása + szöveg-feldolgozás alapok               | 1. + 2.    |
| 3.  | NLP modell tanítása + árfolyam-előrejelzés modellezés          | 2. + 3.    |
| 4.  | Eredmények integrálása + UI építés                             | 4.         |
| 5.  | Tesztelés, prezentáció, finomhangolás                          | mindenki   |

---

## 📂 Példa mappastruktúra

```
crypto_ai_project/
├── data/
│   ├── market_data.csv
│   ├── news_data.csv
│   ├── sentiment_data.csv
├── models/
│   ├── sentiment_model.pkl
│   ├── forecast_model.h5
├── modules/
│   ├── data_collector.py
│   ├── sentiment_analyzer.py
│   ├── forecast_model.py
│   ├── advisor.py
├── app/
│   ├── dashboard.py
│   ├── templates/
│   └── static/
├── README.md
└── main.py
```

---

## 💬 Kommunikáció és integráció

* **GitHub repository** (branch: data, nlp, model, ui)
* **Egységes CSV formátum:** minden modul `datetime` mezőt használjon
* **Interfészek:** minden modul függvényként exportálja az eredményét pl.

  ```python
  def get_latest_forecast(symbol="BTC"):
      return {"trend": "up", "confidence": 0.78}
  ```

---

