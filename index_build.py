"""
Build the composite fragility index (0-100).

Pipeline:
  1. Derive features from raw series (RV, drawdown, rates of change).
  2. Turn each feature into a rolling z-score over Z_WINDOW days.
  3. Apply the sign so that higher always means more fragile.
  4. Clip, weight, sum.
  5. Map the composite to its percentile over RANK_WINDOW days → 0-100.

Every window is ROLLING AND BACKWARD-LOOKING ONLY. No point of the index
uses future data. This is not a detail: without it the backtest lies.

Usage:
    python src/index_build.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import COMPONENTS, DATA, RANK_WINDOW, Z_CLIP, Z_WINDOW


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw series to features, one per entry in COMPONENTS."""
    f = pd.DataFrame(index=raw.index)

    if "dvol" in raw:
        f["dvol"] = raw["dvol"]

    if "funding" in raw:
        f["funding"] = raw["funding"]

    if "btcusdt" in raw:
        px = raw["btcusdt"]
        logret = np.log(px).diff()
        # Annualised 30-day realised volatility, in percent.
        f["rv30"] = logret.rolling(30).std() * np.sqrt(365) * 100
        # Drawdown from the 90-day high (a negative number).
        f["dd90"] = px / px.rolling(90).max() - 1

    if "stable_supply" in raw:
        f["stable_14d"] = raw["stable_supply"].pct_change(14) * 100

    if "defi_tvl" in raw:
        f["tvl_14d"] = raw["defi_tvl"].pct_change(14) * 100

    if "ethusdt" in raw and "btcusdt" in raw:
        f["ethbtc_30d"] = (raw["ethusdt"] / raw["btcusdt"]).pct_change(30) * 100

    if "vix" in raw:
        f["vix"] = raw["vix"]

    return f


def rolling_z(s: pd.Series, window: int = Z_WINDOW) -> pd.Series:
    """Rolling z-score. min_periods is half the window so early data counts."""
    mean = s.rolling(window, min_periods=window // 2).mean()
    std = s.rolling(window, min_periods=window // 2).std()
    return (s - mean) / std.replace(0, np.nan)


def build_index(raw: pd.DataFrame) -> pd.DataFrame:
    feats = build_features(raw)

    zs = pd.DataFrame(index=feats.index)
    weights = {}
    for name, cfg in COMPONENTS.items():
        if name not in feats:
            print(f"[!] component {name} missing — skipped")
            continue
        zs[name] = (rolling_z(feats[name]) * cfg["sign"]).clip(-Z_CLIP, Z_CLIP)
        weights[name] = cfg["weight"]

    if zs.empty:
        raise RuntimeError("no components available")

    w = pd.Series(weights)
    w = w / w.sum()

    # Weighted mean over the components available on each day. If a source
    # drops out, the remaining weights renormalise automatically.
    available = zs.notna()
    composite = ((zs.fillna(0) * w).sum(axis=1)
                 / (available * w).sum(axis=1).replace(0, np.nan))

    # Require at least half the weight to be present, else skip the day.
    composite[(available * w).sum(axis=1) < 0.5] = np.nan

    # Percentile over a rolling window → 0-100.
    index = composite.rolling(RANK_WINDOW, min_periods=180).apply(
        lambda x: (x[:-1] < x[-1]).mean() * 100, raw=True
    )

    out = zs.copy()
    out["composite_z"] = composite
    out["fragility"] = index
    if "btcusdt" in raw:
        out["btc"] = raw["btcusdt"]
    return out


def main():
    raw = pd.read_csv(DATA / "raw.csv", index_col=0, parse_dates=True)
    idx = build_index(raw)
    idx.to_csv(DATA / "index.csv")

    last = idx["fragility"].dropna()
    print(f"Saved: {DATA / 'index.csv'}")
    if len(last):
        print(f"Latest fragility reading: {last.iloc[-1]:.1f} "
              f"on {last.index[-1].date()}")


if __name__ == "__main__":
    main()
