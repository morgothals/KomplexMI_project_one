Persze, kiegészítem a régi összefoglalót azzal, amit most hozzáépítettünk (all-time sentiment + long-term features + log-görbe + dashboard bővítés). A régi szöveget meghagyom, csak beépítem az új részeket.

---

## 🔷 Projekt összefoglaló – `crypto_ai_project`

Ez egy Python alapú **crypto befektetési tanácsadó rendszer** Bitcoinra fókuszálva.
Fő funkciók:

* Piaci adatok gyűjtése (Binance OHLCV + intraday 1m).
* On-chain adatok (Blockchain.com charts API).
* Makró adatok (S&P500, DXY – Yahoo Finance).
* Hír- és sentiment elemzés (CoinDesk, Reddit, Cointelegraph + Fear & Greed index + saját all-time news dataset).
* Feature engineering (technikai indikátorok, on-chain, makró, esemény feature-ök).
* **LSTM modell**, ami a következő 1 órás **log-return-t** tanulja, ebből számolunk következő árat.
* **Hosszútávú, 15 napos felbontású BTC feature-dataset** (2012-től), több idősíkú log-return, SMA, vol, drawdown, on-chain + makró + sentiment aggregált feature-ökkel.
* **Logaritmikus regressziós long-term BTC trend görbe**, amely a teljes history alapján trendet illeszt, de az utolsó biztos árpontra van „ráhorgonyozva”, és szórássávot is ad.
* Rule-based advisor (BUY / HOLD / SELL).
* Egyszerű Flask + Chart.js dashboard, amin már a hosszútávú görbe és a szórássáv is megjelenik.

---

## 📁 Könyvtárstruktúra (lényeges részek)

Projekt gyökér: `crypto_ai_project/`

Fontos mappák és fájlok:

crypto_ai_project/
├── app/
│   ├── dashboard.py              # Flask app (API + HTML dashboard)
│   └── templates/
│       └── dashboard.html        # Frontend UI (Tailwind + Chart.js)
├── data/
│   ├── raw/
│   │   ├── bitcoin_kaggle.csv    # Kaggle BTC history (kézzel letöltve)
│   │   └── news_alltime.csv      # SAJÁT all-time hírdataset (2012-től, oszlopok: date, news)
│   ├── processed/
│   │   ├── market_data.csv              # Binance 1h OHLCV (inkrementális, „operatív”)
│   │   ├── market_data_full.csv         # Kaggle + Binance 1h merge (többéves teljes history)
│   │   ├── onchain_data.csv             # Blockchain.com teljes history napi on-chain
│   │   ├── macro_data.csv               # S&P500 + DXY napi zárók (teljes history)
│   │   ├── sentiment_data.csv           # Rövid (kb. 60 nap) napi sentiment idősor (dashboardhoz)
│   │   ├── news_data.csv                # Max 30 nap nyers hírek (CoinDesk, Reddit, CT rss/scraper)
│   │   ├── training_features_1h.csv     # LSTM train feature store (1h)
│   │   ├── training_sentiment_features.csv
│   │   │                                # Hosszú távú napi sentiment feature store (2012-től)
│   │   └── longterm_features_15d.csv    # 15 napos long-term BTC feature-dataset (2012-től)
│   └── runtime/
│       └── market_intraday_1m.csv       # Aznapi 1m Binance OHLCV (naponta resetelve)
├── models/
│   ├── forecast_model.keras             # Keras LSTM modell (1h log-return target)
│   └── forecast_scalers.pkl             # MinMaxScaler-ek X-re és y-ra (joblib)
├── modules/
│   ├── **init**.py
│   ├── config.py                        # Útvonalak, konstansok, API URL-ek
│   ├── data_collector.py                # Adatletöltés (Binance, on-chain, makró)
│   ├── feature_engineering.py           # Technikai indikátorok (MA, RSI, stb.)
│   ├── feature_assembler.py             # Market + on-chain + macro + sentiment összejoin 1h-ra
│   ├── sentiment_analyzer.py            # Hírek + Fear&Greed → napi sentiment (all-time + friss rss)
│   ├── forecast_model.py                # LSTM train/predict logika (1h log-return)
│   ├── advisor.py                       # Rule-based BUY/HOLD/SELL jelzés
│   ├── longterm_features.py             # 15 napos long-term BTC feature-képzés
│   └── log_curve_forecaster.py          # Log-regressziós hosszútávú BTC trend + szórássáv
├── predictions/
│   └── btc_log_curve_prediction.csv     # Hosszútávú (éves) BTC log-görbe:
│                                        #   - timestamp (év vége, pl. 2012-12-31…2030-12-31)
│                                        #   - pred_log_price (trend szerinti ln(ár))
│                                        #   - pred_price (várható BTC ár az adott ponton)
│                                        #   - pred_price_low / pred_price_high (≈ ±1σ szórássáv)
│                                        #
│                                        # Később ide kerülhetnek más forecast outputok is,
│                                        # pl. long-horizon modellek, LLM által felülvizsgált pályák stb.
├── bootstrap_market_data.py             # Kaggle + Binance 1H history összefűzés
├── build_training_features.py           # Végső training_features_1h.csv előállítása
├── main.py                              # CLI: update_data, build_features,
│                                        #      build_all_features, train, advise, log_curve
└── venv/                                # Virtuális env (lokális)


---

## ⚙️ `modules/config.py`

Fontos beállítások:

* Alappathok:

  * `BASE_DIR`
  * `DATA_DIR`
  * `PROCESSED_DIR`
  * `MODELS_DIR`
* Konkrét fájlok:

  * `MARKET_DATA_CSV` → `data/processed/market_data.csv`
  * `MARKET_DATA_FULL_CSV` → `data/processed/market_data_full.csv` (**teljes 1h history**)
  * `ONCHAIN_DATA_CSV` → `data/processed/onchain_data.csv`
  * `MACRO_DATA_CSV` → `data/processed/macro_data.csv`
  * `SENTIMENT_DATA_CSV` → `data/processed/sentiment_data.csv`
  * `NEWS_DATA_CSV` → `data/processed/news_data.csv`
  * `NEWS_ALLTIME_CSV` → `data/raw/news_alltime.csv` (**új**)
  * `TRAINING_FEATURES_CSV` → `data/processed/training_features_1h.csv`
  * `TRAINING_SENTIMENT_FEATURES_CSV` → `data/processed/training_sentiment_features.csv`
  * `LONGTERM_FEATURES_15D_CSV` → `data/processed/longterm_features_15d.csv` (**új**)
  * `MARKET_INTRADAY_1M_CSV` → `data/runtime/market_intraday_1m.csv`
  * `FORECAST_MODEL_PATH` → `models/forecast_model.keras`
  * `FORECAST_SCALER_PATH` → `models/forecast_scalers.pkl`
* API / URL konstansok:

  * `BINANCE_BASE_URL`, `SYMBOL="BTCUSDT"`, `INTERVAL="1h"`
  * `FEAR_GREED_API_URL` (Alternative.me)
  * `BLOCKCHAIN_CHARTS_BASE` (Blockchain.com charts)
  * `COINDESK_RSS_URL`
  * `REDDIT_CRYPTO_RSS_URL`
  * `COINTELEGRAPH_TAG_URLS` (markets, bitcoin)
* Modell paraméterek:

  * `LOOKBACK` (pl. 60 → 60 óra visszatekintő ablak LSTM-hez)

---

## 🧲 Adatgyűjtés – `modules/data_collector.py`

(változatlanul, csak röviden)

* **Binance OHLCV 1H**: `update_market_data_csv`
  → inkrementális frissítés `market_data.csv`-re.
* **Binance intraday 1m**: `update_intraday_minute_data`
  → aznapi 1m gyertyák `market_intraday_1m.csv`-be.
* **On-chain**: `update_onchain_data`
  → `onchain_data.csv` (n-transactions, n-unique-addresses, hash-rate, avg-block-size, miners-revenue).
* **Makró**: `update_macro_data`
  → `macro_data.csv` (sp500_close, dxy_close).

---

## 📰 Hírek & Sentiment – `modules/sentiment_analyzer.py`

### Hírforrások (friss, magas frekvenciás rész)

Ugyanaz, mint korábban:

* `fetch_coindesk_rss()` – CoinDesk RSS.
* `fetch_reddit_crypto_rss()` – r/CryptoCurrency RSS.
* `fetch_cointelegraph_all_tags()` – Cointelegraph (markets/bitcoin) HTML scraper.

Ezekből **cikk-szintű sentiment** jön létre (VADER `compound` score) és napi aggregáció: átlag, szórás, bullish/bearish arány **a friss napokra**, ahol ténylegesen van több cikk.

### `news_data.csv` – rövid nyers hírtár (max ~30 nap)

`update_news_store()`:

* Összegyűjti a friss RSS/scraper híreket.
* Összefésüli a régi `news_data.csv`-vel.
* URL szerint deduplikál.
* Csak utolsó ~30 nap marad.
* Mentés: `data/processed/news_data.csv`.

### ÚJ: All-time hírdataset – `news_alltime.csv`

A `data/raw/news_alltime.csv` egy **kézzel/extern forrásból összeállított** hosszú idősorú hír-összefoglaló:

* Oszlopok: `date`, `news`
* 2012-től indul, jellemzően **havi szintű** „aggregált” hírszövegek (kulcsesemények).

Erre épül:

#### `build_news_sentiment_from_alltime_csv()`

* Beolvassa a `NEWS_ALLTIME_CSV`-t.
* Minden sorra VADER-rel számít `compound`-ot.
* A havi `date` pontokra így kapsz egy **ritka, de hosszú idősorú** `news_sentiment` idősor.
* Ezután:

  * felvesz egy **napi indexet** a legkisebb dátumtól „ma-ig”,
  * a ritka pontok közé **lineáris interpolációval** számít köztes napokra sentimentet.

Ezzel kapsz egy **2012-től induló napi sentiment bázist** (all-time).

### Fear & Greed index – all-time jelleggel

* A `FEAR_GREED_API_URL`-lel lekérjük a Fear & Greed adatokat (limitet úgy választod, hogy több évre visszamenő legyen).
* A kapott sorozat:

  * timestamp → napra resample,
  * csatlakozik a napi sentiment idősorhoz (`fear_greed` oszlop).

### Napi sentiment idősor – kombinált logika

`build_sentiment_timeseries()` most **két forrást kombinál**:

1. All-time bázis (`news_alltime.csv` → interpolált napi `news_sentiment`).
2. Friss RSS-alapú cikkek (`news_data.csv` → napi aggregált `news_sentiment`, `bullish_ratio`, `bearish_ratio`),
   amelyek **felülírják** az adott nap all-time becslését, ha vannak valós cikkek aznap.

A pipeline:

* `df_base` = all-time napi `news_sentiment` (2012-től).

* `df_recent` = friss cikkekből számolt napi aggregált `news_sentiment`, `news_sentiment_std`, `bullish_ratio`, `bearish_ratio`.

* A kettő összejoinolása úgy, hogy:

  * friss napokon a tényleges cikk-alapú aggregált értékek élnek,
  * régi napokon marad az interpolált all-time bázis.

* Fear & Greed idősor hozzájoinolása (`fear_greed`).

Eredmények:

* **`TRAINING_SENTIMENT_FEATURES_CSV`**

  → többéves, **napi** indexelésű idősor, oszlopokkal:

  * `news_sentiment`
  * `news_sentiment_std` (ahol van elég cikk; régi napokon 0 vagy NaN)
  * `bullish_ratio`, `bearish_ratio` (praktikusan csak a friss időszakra releváns, ahol napi több cikk van)
  * `fear_greed`

* **`SENTIMENT_DATA_CSV`**

  → ebből vágott, **kb. 60 napos** részlet, amit a dashboard használ:

  * gyakorlatban csak azokat a napokat tartalmazza, ahol van **nem teljesen üres** vagy nullás adat (news_sentiment / fear_greed),
  * így nem szerepel egy nagy „0-ákkal tele” szakasz, hanem ténylegesen értelmes a short idősor.

---

## 📐 Feature engineering & training store (1H) – rövid távú modellhez

Ez a rész ugyanaz, csak röviden:

* `feature_engineering.py` → 1H technikai indikátorok (MA-k, EMA-k, RSI, volatilitás, volume-based feature-ök).
* `feature_assembler.py` → összejoinolja:

  * `market_data.csv` (1H),
  * `onchain_data.csv` (napi → 1H align),
  * `macro_data.csv` (napi → 1H align),
  * `training_sentiment_features.csv` (napi → 1H align),
  * esemény feature-ök (halving, nagy események).
* `build_training_features.py` → `TRAINING_FEATURES_CSV` (`training_features_1h.csv`), ami az LSTM-hez megy.

---

## 🧱 ÚJ: Hosszútávú BTC feature-dataset – `modules/longterm_features.py`

Cél: **lassabb időlépcsőjű (15 napos) dataset** hosszútávú trend/előrejelzéshez, LLM-ekhez, stb.

`build_longterm_btc_features()`:

* Kiindulás:

  * `market_data_full.csv` (teljes 1H BTC history, Kaggle + Binance),
  * `onchain_data.csv` (napi),
  * `macro_data.csv` (napi),
  * `training_sentiment_features.csv` (napi, all-time sentiment + F&G).

* Lépések:

  1. A `market_data_full`-t **napi** szintre resample-öli (pl. napi záróár).
  2. Kiszámít:

     * `price_close` (napi záróár),
     * több idősíkú log-return:

       * `log_return_15d`, `log_return_30d`, `log_return_90d`
     * simított árak:

       * `sma_30d`, `sma_90d`, `sma_180d`
     * volatilitás:

       * `vol_30d`, `vol_90d` (rolling std a napi log-returnre)
     * drawdown:

       * pl. `drawdown_180d` (180 napos lokális max-hoz mért visszaesés).
  3. On-chain és makró adatok joinolása.
  4. `training_sentiment_features` hozzájoinolása:

     * `news_sentiment`, `fear_greed` + 15 napos rolling aggregátumok:

       * pl. `news_sentiment_15d_mean`, `fear_greed_15d_mean`,
       * egy egyszerű `news_sentiment_15d_trend` (pl. különbség az utolsó és az első 15 napos átlag között).
  5. Az egészet **15 napos rácsra** mappeli (pl. minden 15. napra egy sor, a köztelévő napok aggregációival).
  6. Long-horizon targetek (ha használod):

     * `target_log_return_1y` (kb. 365 nappal későbbi log-return),
     * `target_vol_1y` (következő ~1 év volatilitása).

* Mentés: `LONGTERM_FEATURES_15D_CSV` → `data/processed/longterm_features_15d.csv`.
  Ez lesz az alapja bármilyen **„hosszú távú (években mérhető)”** modellnek / LLM inputnak, ahol már nem 1H idősíkban gondolkodsz.

A `main.py` `update_data` parancsa a végén meghívja:

```python
df_long = build_longterm_btc_features()
print(f"Hosszútávú feature shape: {df_long.shape}")
```

---

## 🤖 LSTM modell – `modules/forecast_model.py`

Ugyanaz: 1H log-return-ök, sliding window, MinMaxScaler, Keras LSTM, target: következő 1 órás log-return → `forecast_model.keras`, `forecast_scalers.pkl`.

---

## 💡 Advisor – `modules/advisor.py`

`generate_advice()`:

* Meghívja a rövid távú LSTM modellt (`predict_next_close()`).
* Számolja a relatív változást.
* Mellé csatolja az aktuális `fear_greed` és `news_sentiment` értékeket.
* Egy egyszerű szabályrendszer alapján `BUY` / `HOLD` / `SELL` jelzést ad.

---

## 🌈 ÚJ: Logaritmikus regressziós hosszútávú BTC trend – `modules/log_curve_forecaster.py`

Ez a modul **nem neurális háló**, hanem egy „statisztikai” modell:

1. Beolvassa a **teljes BTC napi history-t** `market_data_full.csv`-ből.

2. Kiszámítja a napok számát a legelső dátumtól: `t = (timestamp - start).days`.

3. Logaritmikus ár:

   ```python
   log_price = ln(close)
   ```

4. **Súlyozott lineáris regresszió**:

   * `log_price ~ a + b * t`
   * a minták súlya nő az idővel (régi évek: kisebb súly, friss évek: nagyobb súly),
   * így a trendet jobban a közelmúlt befolyásolja.

5. **Re-anchoring az utolsó biztos pontra**:

   * A regresszió meredeksége: `b`.
   * Az interceptet úgy állítjuk be (`a_adj`), hogy a modell **pontosan átmenjen az utolsó valódi árponton** (pl. 2025 végi BTC ár).
   * Ez garantálja, hogy **2025-ben a pred_price ≈ valós utolsó ár**, és innen indul a jövő extrapoláció.

6. Residualok szórása:

   * `std = std(log_price - pred_log_price)`
   * ebből képezzük a szórássávot.

7. Éves pontok generálása:

   * 2012-től `end_year`-ig (pl. 2030),

   * minden év végére (dec 31) egy pont:

     ```text
     timestamp, pred_log_price, pred_price,
     pred_price_low, pred_price_high
     ```

   * ahol:

     ```python
     pred_price      = exp(pred_log)
     pred_price_low  = exp(pred_log - sigma_mult * std)
     pred_price_high = exp(pred_log + sigma_mult * std)
     ```

8. Mentés:

   * `predictions/btc_log_curve_prediction.csv`.

`run_log_regression_curve(end_year=2030, sigma_mult=1.0)`:

* lefuttatja az egészet,
* kinyomtatja a paramétereket (a_adj, b, std),
* elmenti a CSV-t.

A CLI-ben:

```bash
python main.py log_curve
```

---

## 🌐 Flask dashboard – `app/dashboard.py` + `templates/dashboard.html`

### Backend: `app/dashboard.py`

`/api/state` most már ezeket adja vissza:

* `candles_1h` – 1H OHLCV (utolsó ~200 gyertya) a `market_data.csv`-ből.
* `intraday_1m` – aznapi 1m árak a `market_intraday_1m.csv`-ből.
* `sentiment` – a `sentiment_data.csv` ~60 napos idősora:

  * `timestamps`
  * `news_sentiment`
  * `fear_greed`
  * `latest` (utolsó értékek).
* `advice` – a `generate_advice()` outputja.
* **ÚJ: `long_curve`** – a log-görbe és szórássáv:

  ```json
  {
    "labels": ["2012", "2013", ..., "2030"],
    "pred_price": [...],
    "pred_price_low": [...],
    "pred_price_high": [...]
  }
  ```

Ez a `load_longterm_curve()` helperben olvassa be a `predictions/btc_log_curve_prediction.csv`-t.

### Frontend: `templates/dashboard.html`

* Tailwind + Chart.js.

* Felső grid (3 kártya):

  1. **Jelzés kártya**

     * BUY/HOLD/SELL
     * utolsó záróár
     * következő ár predikció
     * várható változás (%)
  2. **Hangulat kártya**

     * Fear & Greed aktuális érték
     * News sentiment aktuális érték
     * Chart.js vonaldiagram:

       * y1: news_sentiment,
       * y2: Fear & Greed index,
       * tengelyfeliratokkal: „Idő (napok)”, „News sentiment”, „Fear & Greed index”.
  3. **Intraday (1m) kártya**

     * vonaldiagram a mai 1m close árakról,
     * x tengely: „Idő (mai nap, percek)”, y: „BTC ár (USD)”.

* Alul:

  * 1H close chart (line chart):

    * label: „BTC záróár (1H, USD)”
    * x tengely: „Idő (utolsó ~200 óra)”
    * y tengely: „BTC ár (USD)”

* **ÚJ: hosszútávú BTC trend grafikon**

  * Canvas: `longCurveChart`.

  * Három dataset:

    1. `pred_price` → „Várható BTC ár 5 év múlva”
    2. `pred_price_low` → „Alsó sáv (≈ -1σ)” – szaggatott vonal
    3. `pred_price_high` → „Felső sáv (≈ +1σ)” – szaggatott vonal

  * X tengely: „Év (current_timestamp)” (az év, amelyhez a 5 éves horizontra számolt ár tartozik).

  * Y tengely: „Modellezett BTC ár 5 év múlva (USD)”

    * tickek formázása `toLocaleString()`-gel, hogy ezres elválasztó is legyen.

A JS-ben a `refresh()` 60 másodpercenként újra lehúzza az `/api/state`-et, és:

* `upsertCharts(state)` → frissíti a 4 Chart.js grafikont.
* `updateInfoPanels(state)` → frissíti a jelzés és hangulat panel szövegeit.

---

## 🧪 Tipikus futási sorrend (kiegészítve)

0. **Kaggle letöltés**

Kaggle adat bemásolása -> ehhez kell csinálni -> data/raw/bitcoin_kaggle.csv
Letöltés: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/data
Betenni és átnevezni bitcoin_kaggle.csv -ra


1. **Adatfrissítés** (piaci, on-chain, makró, sentiment, long-term dataset):

   ```bash
   python main.py update_data
   ```

   Ez most:

   * frissíti a 1H OHLCV-t (`market_data.csv`),
   * frissíti az on-chain, makró adatokat,
   * újraépíti a sentiment idősorokat (`training_sentiment_features.csv`, `sentiment_data.csv`),
   * frissíti az intraday 1m adatot,
   * **újraépíti a 15 napos long-term feature-datasetet** (`longterm_features_15d.csv`).

2. **Training feature store építés (1H)**:

   ```bash
   python build_training_features.py
   ```

3. **Rövid távú LSTM modell tanítása**:

   ```bash
   python main.py train --epochs 20
   ```

4. **Advisory jelzés CLI-ben**:

   ```bash
   python main.py advise
   ```

5. **Hosszútávú log-görbe frissítése**:

   ```bash
   python main.py log_curve
   # -> predictions/btc_log_curve_prediction.csv
   ```

6. **Flask dashboard** indítása:

   ```bash
   python -m app.dashboard
   # http://localhost:5000/
   ```

   Itt már látszik:

   * rövid távú (1H, 1m),
   * hangulat,
   * **valamint a hosszútávú BTC trend görbe szórássávval** 2012-től 2030-ig.
