# news_data.csv — Kriptovaluta hírek

Tárolja a forrásból származó híreket időbélyeggel, forrással és a hír rövid tartalmával.

| Oszlopnév  | Típus                                          | Jelentés                                                    |
| ---------- | ---------------------------------------------- | ----------------------------------------------------------- |
| `datetime` | ISO-formátumú UTC idő (`YYYY-MM-DDTHH:mm:ssZ`) | Hír publikálási ideje                                       |
| `symbol`   | `str`                                          | Melyik kriptóhoz kapcsolódik (`BTC`, `ETH`, `SOL`, `MULTI`) |
| `source`   | `str`                                          | Forrás (pl. `Yahoo Finance`, `CoinDesk`)                    |
| `title`    | `str`                                          | Hír címe                                                    |
| `summary`  | `str`                                          | Rövid összefoglaló / leírás                                 |
| `url`      | `str`                                          | Hír linkje                                                  |


# sentiment_data.csv — Elemzett hírek (AI/LLM kimenet)

A news_data.csv-ből elemzett érzelmi értékek tárolása, amelyeket a sentiment_analyzer.py modul állít elő.

Ezeket a modul automatikusan képes előállítani pl. TextBlob, VADER, vagy transformers (FinBERT) modellel.

| Oszlopnév         | Típus   | Jelentés                                               |
| ----------------- | ------- | ------------------------------------------------------ |
| `datetime`        | ISO UTC | A hír időpontja (öröklődik a `news_data.csv`-ből)      |
| `symbol`          | `str`   | Melyik kriptóhoz tartozik                              |
| `title`           | `str`   | Hír címe (azonosításra)                                |
| `sentiment_score` | `float` | Érzelmi polaritás (–1 negatív, 0 semleges, +1 pozitív) |
| `sentiment_label` | `str`   | `positive` / `neutral` / `negative`                    |


# market_data.csv — Árfolyam-idősor (kriptovaluták)

Tárolja az adott kriptók napi vagy órás árfolyamadatait (Open-High-Low-Close-Volume formátumban).

| Oszlopnév  | Típus   | Jelentés                                |
| ---------- | ------- | --------------------------------------- |
| `datetime` | ISO UTC | Időpont (gyertya nyitás ideje)          |
| `symbol`   | `str`   | Kriptovaluta (pl. `BTC-USD`, `ETH-USD`) |
| `open`     | `float` | Nyitóár                                 |
| `high`     | `float` | Legmagasabb ár                          |
| `low`      | `float` | Legalacsonyabb ár                       |
| `close`    | `float` | Záróár                                  |
| `volume`   | `float` | Forgalom mennyisége (volumen)           |
