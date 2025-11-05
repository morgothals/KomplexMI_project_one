
### Cél:
Egy AI-alapú befektetési tanácsadó rendszer fejlesztése, ami:

automatikusan adatokat gyűjt (pl. kriptovaluta, deviza, cég, ország gazdasági adatok),

ezekből következtetéseket és ajánlásokat készít,

és neurális hálók + természetes nyelvű modellek segítségével értelmezi a híreket, trendeket.

Ez tehát egy narrow AI (szűk célú AI) projekt: nem általános mesterséges intelligencia, hanem egy konkrét problémára – befektetési döntések segítésére – tanított rendszer.

---

## 🧩 Összefoglaló architektúra

```
[Adatgyűjtés modul] → [Szövegelemző modell] → [Idősor-előrejelző modell] → [Tanácsadó (döntéstámogató) modul]
```

Minden modul önállóan fejleszthető, és később egy `main.py` vagy webes felület integrálja őket.

---

## 👥 Szereposztás és feladatok

### 🧑‍💻 **1. személy (DANI) – Adatgyűjtés és API integráció (Data Engineer)**

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

### 🤖 **2. személy (ÁDÁM) – Szövegelemzés és hangulatelemzés (NLP specialist)**

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

### 📈 **3. személy (PETI)– Idősor-előrejelzés (ML engineer / Data Scientist)**

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

### 🧠 **4. személy (SZABI) – Tanácsadó és front-end integráció (AI logic & UI)**

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


szuper — íme az **1. fázis (4 hét)** részletes, tanár-barát projektterv, 4 főre bontva, párhuzamosítható feladatokkal, konkrét kimenetekkel és mérőszámokkal. A dátumok Budapest szerint értendők.

# 🗓 Ütemezés áttekintés (2025)

* **1. hét:** nov 5 – nov 11
* **2. hét:** nov 12 – nov 18
* **3. hét:** nov 19 – nov 25
* **4. hét:** nov 26 – dec 2

# 👥 Szerepek (fix felelősség + helyettesíthetőség)

* **A – Data Engineer (Adatgyűjtés & ETL):** API-k, adatminőség, tárolás
* **B – NLP Specialist (Szövegelemzés):** sentiment, kulcsszavak, kiértékelés
* **C – ML Engineer (Idősor-előrejelzés):** feature-ök, modell, validáció
* **D – Integrátor & UI (Tanácsadó logika + Dashboard):** pipeline, UX, vizualizáció

---

# 📂 Kötelező egységes interfészek (már az 1. héten lefektetve)

**Közös időbélyeg formátum:** `UTC ISO8601` (pl. `2025-11-05T08:00:00Z`)
**Szimbólum kulcs:** `symbol ∈ {BTC-USD, ETH-USD, SOL-USD}`

**Fájl-sémák**

* `data/market_data.csv`

  * oszlopok: `timestamp, symbol, open, high, low, close, volume`
* `data/news_raw.csv`

  * oszlopok: `timestamp, source, title, text, url, symbol_tags`
* `data/sentiment.csv`

  * oszlopok: `timestamp, doc_id, symbol, sentiment_score[-1..1], sentiment_label{neg,neu,pos}, keywords[list]`
* `data/features.csv` (model input C-nek)

  * oszlopok: `timestamp, symbol, close, rsi14, sma20, sma50, sent_mean_3h, sent_mean_24h, ... , target_dir{down,flat,up}`

**Függvény-szerződések (Python)**

* `modules/data_collector.py::collect_market(symbol: str, start: str, end: str) -> pd.DataFrame`
* `modules/news_collector.py::collect_news(symbols: list[str], start: str, end: str) -> pd.DataFrame`
* `modules/sentiment_analyzer.py::score_news(df_news: pd.DataFrame) -> pd.DataFrame`
* `modules/feature_builder.py::build_timeseries(df_mkt, df_sent) -> pd.DataFrame`
* `modules/forecast_model.py::train(df_feat) -> TrainedModel; predict(model, horizon_h:int=6) -> dict`
* `modules/advisor.py::advise(pred, context) -> {"action": "buy|hold|sell", "confidence": float, "rationale": str}`

---

# ✅ Mérőszámok (elfogadási kritériumok)

* **NLP (B):**

  * *Label-szintű ellenőrzés:* min. **70%** pontosság kézzel ellenőrzött 100 minta-cikken
  * *Stabilitás:* ugyanazon hír szentimentje ±0.1-nél jobban ne ingadozzon újrafutáskor
* **Idősor (C):**

  * *Irányhelyesség 6h horizonton:* **≥ 55%** (baseline felett)
  * *MAPE (ha regressziós előrejelzés):* **≤ 8–12%** piloton
* **Rendszer (D + mindenki):**

  * *End-to-end futtatás:* egy gombos (CLI/Streamlit) pipeline lefut hiba nélkül
  * *Dashboard:* grafikonok + akciójavaslat + indoklás látható, frissíthető
* **Adatminőség (A):**

  * *Hiányzók aránya:* kritikus feature-ökben **< 1%**, imputálás dokumentálva
  * *Időszinkron:* piac és hírfolyam összeillesztés drifte **< 1 perc** átlag

---

# 🧭 1. hét (nov 5–11) – Alapok, adatút és prototípusok

**Mindenki**

* GitHub repo, issue sablonok, branch-stratégia (`feat/*`, `fix/*`, `docs/*`), CI lint
* `.env.example` (API kulcsok helye), `README` v0, adatvédelmi/etikai megjegyzések

**A – Data Engineer**

* API-próbák: egy választott árfolyamforrás (pl. yfinance / CoinGecko) + 2 hírforrás (RSS vagy könnyen elérhető feed)
* `collect_market()` és `collect_news()` kezdeti implementáció, CSV-mentés
* Időzóna-normalizálás, duplikátum-szűrés, rate-limit kezelési terv

**B – NLP**

* Baseline sentiment: VADER/TextBlob **és** egy finomhangolatlan FinBERT/BERT modell összevetése 30–50 cikken
* `score_news()` prototípus: `sentiment_score`, `sentiment_label`, `keywords`
* Kézi címkézésre minta CSV (min. 100 sor) – ez lesz a későbbi validáció alapja

**C – ML**

* Feature-katalógus tervezet (TA, műszaki indikátorok + aggregált szentiment)
* `feature_builder()` váz: RSI, SMA, gördülő sent_mean (3h/24h)
* Train/test split stratégia időalapon (no leakage), baseline (naiv irányjelző)

**D – Integrátor & UI**

* Streamlit váz: 3 tab (Piac, Hírek & Szentiment, Tanács)
* Adatbetöltés gomb, egyszerű grafikonok (close, sent_mean)
* Egységes hibaüzenetek, loading állapotok

**Deliverable (1. hét vége):**

* Futó **adatletöltés + baseline sentiment + baseline feature**
* Streamlit app v0 (grafikon + táblázat), rövid **tech demo** 5 percben

---

# 🔧 2. hét (nov 12–18) – NLP finomítás + Feature-rendszer + Adatminőség

**A – Data Engineer**

* Stabilizálás: visszatérési kódok, retry/backoff, logolás (`logs/etl_*.jsonl`)
* Szimbólum-tagelés hírekben (cím/URL alapján), egyszerű NER/regex kulcsszűrés
* Időbeli join ellenőrzése (hír ➜ megfelelő gyertya/ablak)

**B – NLP**

* Finomhangolás (ha idő engedi): kis kézi címkézett mintán *light* fine-tune vagy prompt-alapú normalizálás
* Kulcsszó-pipeline: “ETF”, “regulation”, “halving”, “hack”, “SEC”, stb. (top-N tf-idf + kézi stoplista)
* Validáció: 100 minta, pontosság/konzisztencia jelentés

**C – ML**

* Feature-rendszer kibővítése (volatility, ATR, z-score, sent_volatility)
* Célváltozó: **irány (up/flat/down 6h)** + alternatív regressziós cél (Δ% 6h)
* Modellkísérletek: **GRU/LSTM** baseline **vs.** XGBoost/RandomForest (irány)
* Keresztvalidáció időablakokkal (rolling origin)

**D – Integrátor & UI**

* Modell-pluginek: `predict()` integrálása az appba, kimeneti kártya: *Action + Confidence + Why*
* Vizualizációk:

  * gyertya + előrejelzés sáv
  * 24h szentiment idősor
  * kulcsszó felhő / top-kulcsszavak lista

**Deliverable (2. hét vége):**

* **NLP jelentés** (pontosság, döntési példák)
* **Model comparison** jegyzet (irányhelyesség, baseline felett)
* App v1: előrejelzés + akciókártya megjelenik

---

# 📈 3. hét (nov 19–25) – Modell stabilizálás + Backtesting + Tanácsadói szabályok

**A – Data Engineer**

* Backfill 3–6 hónap adatra (legalább BTC-USD), uniform CSV-k
* Adatminőség dashboard (missing, outlier, időcsúszás)

**B – NLP**

* Driftszonda: kulcsszavak/források szerepének változása (heti snapshot)
* Hibaanalízis: félrecímkézett minták katalógusa (tanárnak nagyon jó pont)

**C – ML**

* **Backtesting**: gördülő ablakos teszt 6h horizonton, metrikák összesítése
* **Feature importance** (klasszikus modellnél), LSTM-nél SHAP mintasorokra
* Threshold-optimalizálás “no-trade” sávra (bizonytalanság esetén HOLD)

**D – Integrátor & UI**

* **Tanácsadói szabálymotor** (ensemble):

  * ha `pred_dir=up` & `sent_mean_3h>0` & vol nem extrém → **BUY**
  * ha `pred_dir=down` & negatív hangulat → **SELL**
  * ha bizonytalan → **HOLD**
* Jelmagyarázat + kockázati disclaimer, *paper trade* gomb (nem köt valódi ügyletet)

**Deliverable (3. hét vége):**

* **Backtest riport** (irányhelyesség, MAPE, confusion matrix)
* App v2: szabályalapú tanácsadó, részletes indoklással

---

# 🚀 4. hét (nov 26–dec 2) – Finiselés, prezentáció, dokumentáció

**A – Data Engineer**

* Reprodukálható `make data`/`python run_etl.py` parancs
* Végső adat-dokumentáció (források, korlátok, etika)

**B – NLP**

* Végső validáció (új 50 cikk), hibakategóriák és javaslatok
* Rövid “model card” a sentiment modulhoz

**C – ML**

* Végső modell mentése (`models/forecast_lstm_v1.h5` + `models/meta.json`)
* Tanárbarát ábra: *predikció vs. valóság* + irányhelyesség időben

**D – Integrátor & UI**

* Polírozott dashboard (egységes design, dark mode ok), “Demo flow” gomb
* **1-kattintásos demo:** `python main.py --symbol BTC-USD --horizon 6`

**Közös deliverablek:**

* **Végső preziszlajd** (10–12 dia): cél, architektúra, metrikák, demo GIF
* **README (végleges):** telepítés, futtatás, mappastruktúra, eredmények, korlátok
* **Etikai/jogi megjegyzés:** nem valós befektetési tanács

---

# 🧱 Kockázatok & mitigáció

* **API rate-limit / változó elérhetőség:** cache-elés, retry/backoff, forrás-fallback
* **NLP zajos adat:** több forrás, szabályos kulcsszűrés, kézi validációs minta
* **Idősor drift:** rendszeres backtesting, threshold-alapú HOLD
* **Integrációs csúszás:** korai függvény-szerződések, dummy adapterek a másik fél helyett

---

# 🛠 Technológiai csomag (javaslat)

* **Python 3.11**, `pandas`, `numpy`, `scikit-learn`, `tensorflow/keras` vagy `pytorch`
* NLP: `transformers`, `nltk`/`spacy`, baseline: `vaderSentiment`
* UI: `streamlit`, grafikon: `plotly`/`matplotlib`
* Orkesztráció: egyszerű `make` vagy `tox`; log: `loguru`
* Formázás: `black`, `ruff`; típusok: `mypy`

---

# 📌 Issue-szintű teendőlista (rövid, megnyitható a GitHub-ban)

**Hét 1**

* [A] `collect_market()` + minta CSV BTC-USD 14 nap
* [A] `collect_news()` 2 forrásból, 7 nap, symbol tag
* [B] `score_news()` baseline VADER + FinBERT próba
* [C] `feature_builder()` váz: RSI, SMA, sent_mean
* [D] Streamlit váz + grafikonok + betöltés

**Hét 2**

* [A] Retry/backoff + logolás + időszinkron ellenőrző
* [B] 100 cikk kézi validáció + kulcsszavak
* [C] LSTM/GRU vs. XGBoost irányhelyesség teszt
* [D] `predict()` integráció + akciókártya

**Hét 3**

* [A] Backfill 3–6 hó adat
* [B] Drift/hibaanalízis jegyzet
* [C] Backtesting riport (rolling window)
* [D] Szabálymotor + indoklókártya, no-trade sáv

**Hét 4**

* [A] ETL runbook + adatdoksi
* [B] NLP model card + végső valid
* [C] Modell mentések + ábrák
* [D] Demo flow, preziszlajd, README v1.0

---

