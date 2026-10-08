"""
Backtest. Two independent tests.

TEST 1 — rule.
    Signal: the index crosses ALERT_THRESHOLD from below.
    Event:  BTC draws down past DRAWDOWN_THRESHOLD within HORIZON_DAYS.
    Reports precision, recall, false alarm count and mean lead time.

TEST 2 — walk-forward model.
    Logistic regression on the same z-scored components, trained only on
    the past, expanding window, refit every 30 days. This checks whether
    the components carry signal beyond the rule itself.

The point: neither test sees the future. Any backtest that breaks this
produces beautiful numbers and fails in production.

Usage:
    python src/backtest.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import (ALERT_THRESHOLD, COMPONENTS, DATA, DRAWDOWN_THRESHOLD,
                    HORIZON_DAYS, KNOWN_EVENTS)


# ---------------------------------------------------------------------------
def make_target(price: pd.Series) -> pd.Series:
    """
    1 if price fell past the threshold within HORIZON_DAYS, measured
    against TODAY's price.
    """
    fwd_min = price.shift(-1).rolling(HORIZON_DAYS, min_periods=1).min()
    fwd_min = fwd_min.shift(-(HORIZON_DAYS - 1))
    dd = fwd_min / price - 1
    return (dd <= DRAWDOWN_THRESHOLD).astype(float).where(dd.notna())


# ---------------------------------------------------------------------------
def test_rule(idx: pd.DataFrame) -> dict:
    df = idx.dropna(subset=["fragility", "btc"]).copy()
    df["target"] = make_target(df["btc"])
    df = df.dropna(subset=["target"])

    above = df["fragility"] >= ALERT_THRESHOLD
    signals = df.index[above & ~above.shift(1, fill_value=False)]

    if len(signals) == 0:
        return {"signals": 0}

    hits = df.loc[signals, "target"].sum()
    precision = hits / len(signals)

    # Recall: how many stress episodes were caught. An episode is a
    # contiguous block of target==1 days; it counts as caught if any
    # signal fired within HORIZON_DAYS before it started.
    ev = df["target"].astype(int)
    starts = df.index[(ev == 1) & (ev.shift(1, fill_value=0) == 0)]
    caught, leads = 0, []
    for start in starts:
        window = signals[(signals <= start) &
                         (signals >= start - pd.Timedelta(days=HORIZON_DAYS))]
        if len(window):
            caught += 1
            leads.append((start - window[-1]).days)
    recall = caught / len(starts) if len(starts) else np.nan
    base_rate = df["target"].mean()

    return {
        "signals": int(len(signals)),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "false_alarms": int(len(signals) - hits),
        "episodes": int(len(starts)),
        "mean_lead_days": round(float(np.mean(leads)), 1) if leads else None,
        "base_rate": round(float(base_rate), 3),
        "lift": round(float(precision / base_rate), 2) if base_rate else None,
    }


# ---------------------------------------------------------------------------
def test_walkforward(idx: pd.DataFrame, refit_every: int = 30,
                     min_train: int = 365) -> dict:
    cols = [c for c in COMPONENTS if c in idx.columns]
    df = idx[cols + ["btc"]].dropna().copy()
    df["target"] = make_target(df["btc"])
    df = df.dropna(subset=["target"])

    X_all, y_all = df[cols].values, df["target"].values
    preds = np.full(len(df), np.nan)
    model = scaler = None

    for i in range(min_train, len(df)):
        if (i - min_train) % refit_every == 0:
            # Train only where the outcome has already materialised.
            cut = i - HORIZON_DAYS
            if cut <= 50:
                continue
            Xtr, ytr = X_all[:cut], y_all[:cut]
            if len(np.unique(ytr)) < 2:
                continue
            scaler = StandardScaler().fit(Xtr)
            model = LogisticRegression(
                max_iter=1000, class_weight="balanced"
            ).fit(scaler.transform(Xtr), ytr)
        if model is not None:
            preds[i] = model.predict_proba(
                scaler.transform(X_all[i:i + 1])
            )[0, 1]

    mask = ~np.isnan(preds)
    if mask.sum() < 100 or len(np.unique(y_all[mask])) < 2:
        return {"status": "not enough data"}

    pd.Series(preds, index=df.index, name="model_prob").to_frame().to_csv(
        DATA / "walkforward_probs.csv")

    return {
        "status": "ok",
        "oos_days": int(mask.sum()),
        "roc_auc": round(float(roc_auc_score(y_all[mask], preds[mask])), 3),
        "note": "0.5 = useless, 0.65+ = real signal",
    }


# ---------------------------------------------------------------------------
def check_known_events(idx: pd.DataFrame) -> pd.DataFrame:
    """What the index read in the two weeks before each known crash."""
    rows = []
    series = idx["fragility"].dropna()
    for date, label in KNOWN_EVENTS.items():
        d = pd.Timestamp(date)
        window = series[(series.index >= d - pd.Timedelta(days=14)) &
                        (series.index <= d)]
        rows.append({
            "event": label,
            "date": date,
            "max_14d_before": round(float(window.max()), 1)
            if len(window) else None,
            "on_the_day": round(float(series.asof(d)), 1)
            if len(series) and series.index.min() <= d else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    idx = pd.read_csv(DATA / "index.csv", index_col=0, parse_dates=True)

    print("=" * 60)
    print("TEST 1 — threshold rule")
    print("=" * 60)
    for k, v in test_rule(idx).items():
        print(f"  {k:>18}: {v}")

    print("\n" + "=" * 60)
    print("TEST 2 — walk-forward model")
    print("=" * 60)
    for k, v in test_walkforward(idx).items():
        print(f"  {k:>18}: {v}")

    print("\n" + "=" * 60)
    print("KNOWN EVENTS")
    print("=" * 60)
    print(check_known_events(idx).to_string(index=False))


if __name__ == "__main__":
    main()
