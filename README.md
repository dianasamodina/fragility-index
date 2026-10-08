# Crypto Market Fragility Index

A daily 0–100 index of accumulated fragility in crypto markets, published with an open forecast log.

**Thesis:** a market is most vulnerable not when it panics, but when it is quiet. Calm breeds confidence, confidence breeds leverage, leverage breeds fragility.

Methodology by **Diana Samodina**.

[→ Live forecast log](logs/forecast_log.csv)

---

## What the data showed

The first version of this index was built on common sense: volatility rises, risk rises. Testing against 2017–2026 history showed that this is **wrong**.

| Component | AUC with the original sign | AUC inverted |
|---|---|---|
| DVOL (implied volatility) | 0.380 | **0.620** |
| Realised volatility, 30d | 0.445 | **0.555** |
| Composite index | 0.402 | — |

An AUC of 0.40 means the index ran backwards: the higher it read, the **lower** the probability of a drawdown.

The reason is economic, not technical. High volatility is a crash that has already happened, which usually marks a bottom. Prolonged calm is the period in which leverage accumulates and the system becomes vulnerable to a shock.

This is a documented mechanism: Minsky's financial instability hypothesis, also called the volatility paradox. Well studied in traditional markets, barely quantified in crypto.

## Components

| Component | AUC | Weight |
|---|---|---|
| DVOL (inverted) | 0.620 | 0.30 |
| Stablecoin supply change, 14d | 0.604 | 0.25 |
| Drawdown from 90d high | 0.561 | 0.15 |
| Realised volatility (inverted) | 0.555 | 0.15 |
| DeFi TVL change, 14d | 0.549 | 0.15 |
| Perpetual funding rate | 0.502 | dropped |
| ETH/BTC, 30d | 0.518 | dropped |

Stablecoin flows deserve a note: this is the only component whose power grows with lag — 0.604 on the signal day, 0.643 thirty days ahead of the event. A genuine leading indicator, where the others work closer to the moment.

## Walk-forward results

Trained only on the past, refit every 30 days, 1331 out-of-sample days:

- **ROC AUC 0.652**
- Top decile of signals: drawdowns occurred in **9.0%** of cases against a 4.1% base rate
- **Lift 2.21**

In the red zone, a drawdown past 15% within two weeks happens roughly twice as often as usual.

## Honest limitations

**Component signs were chosen with the full history in view.** That is partial fitting. The defence is that the mechanism is documented in the literature independently of this data — but the real test is the future.

**This is why the public forecast log exists.** It is the only way to demonstrate that the signal is real. Every entry is committed with a server-side timestamp and never edited afterwards.

**The signal is moderate.** AUC 0.652 is not crisis prediction; it is a doubling of probability. That is what this project claims, and nothing more.

**Weights are deliberately not optimised.** Tuning weights to maximise a metric on the same history produces an index that explains the past perfectly and says nothing about the future.

## Data sources

All free, no API keys. Deribit for implied volatility, Binance with a Coinbase fallback for prices, DefiLlama for stablecoin supply and DeFi TVL, Stooq with a FRED fallback for VIX.

## Running it

Install dependencies with `pip install -r requirements.txt`, then run `collect.py` to fetch history since 2017, `index_build.py` to build the index, `backtest.py` to validate, and `publish.py` to append a log entry.

A GitHub Actions workflow runs this daily at 00:20 UTC and commits the forecast automatically.

## Layout

- `config.py` — the entire methodology: components, signs, weights
- `sources.py` — collectors with fallback venues
- `collect.py` — fetch orchestration, per-source caching
- `index_build.py` — features to z-scores to composite to 0-100
- `backtest.py` — rule, walk-forward and known events
- `publish.py` — forecast log and post text
- `logs/` — the public forecast log, committed daily

## License

MIT for the code. Please cite as *Samodina, D. — Crypto Market Fragility Index*.
