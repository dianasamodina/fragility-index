"""
Daily publication. The most important script in the project.

It does three things:
  1. Appends a row to logs/forecast_log.csv — the public forecast log.
     A row is written once per date and never edited afterwards.
  2. Resolves forecasts whose horizon has elapsed.
  3. Prints a ready-to-post summary.

Why this matters more than anything else here: a git-tracked log with
server-side timestamps cannot be backdated. Commit every day, no gaps.

Usage:
    python src/publish.py
    python src/publish.py --resolve-only
"""

from __future__ import annotations

import argparse

import pandas as pd

from config import (ALERT_THRESHOLD, COMPONENTS, DATA, DRAWDOWN_THRESHOLD,
                    HORIZON_DAYS, LOGS)

LOG_PATH = LOGS / "forecast_log.csv"

COLUMNS = ["date", "fragility", "zone", "btc_price", "horizon_days",
           "threshold_pct", "call", "resolved_on", "realized_min_pct",
           "outcome", "top_drivers"]


def zone_of(v: float) -> str:
    if v >= ALERT_THRESHOLD:
        return "red"
    if v >= 65:
        return "amber"
    return "green"


def top_drivers(row: pd.Series, k: int = 3) -> str:
    """The three components pushing the index up hardest today."""
    present = {c: row[c] for c in COMPONENTS if c in row and pd.notna(row[c])}
    if not present:
        return ""
    ranked = sorted(present.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return "; ".join(f"{COMPONENTS[c]['label']} (z={v:+.1f})"
                     for c, v in ranked)


def load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=COLUMNS)


def append_today(idx: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    live = idx.dropna(subset=["fragility"])
    if live.empty:
        raise RuntimeError("index is empty — run collect.py and index_build.py")

    row = live.iloc[-1]
    date = live.index[-1].date().isoformat()

    if date in set(log["date"].astype(str)):
        print(f"[=] entry for {date} already exists, log untouched")
        return log

    fragility = float(row["fragility"])
    zone = zone_of(fragility)
    call = (f"ELEVATED FRAGILITY: probability of a BTC drawdown past "
            f"{abs(DRAWDOWN_THRESHOLD):.0%} within {HORIZON_DAYS} days is "
            f"roughly twice the base rate") if zone == "red" else (
           "WATCH: fragility above normal but below the signal threshold"
           if zone == "amber" else
           "CALM: no accumulated fragility detected")

    new = {
        "date": date,
        "fragility": round(fragility, 1),
        "zone": zone,
        "btc_price": round(float(row.get("btc", float("nan"))), 2),
        "horizon_days": HORIZON_DAYS,
        "threshold_pct": DRAWDOWN_THRESHOLD * 100,
        "call": call,
        "resolved_on": "",
        "realized_min_pct": "",
        "outcome": "pending",
        "top_drivers": top_drivers(row),
    }
    print(f"[+] logged {date}: fragility {fragility:.1f} ({zone})")
    return pd.concat([log, pd.DataFrame([new])], ignore_index=True)


def resolve(idx: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """Mark the outcome of forecasts whose horizon has closed."""
    if log.empty or "btc" not in idx:
        return log

    px = idx["btc"].dropna()
    px.index = pd.to_datetime(px.index)
    today = px.index.max()

    for i, r in log.iterrows():
        if r.get("outcome") != "pending":
            continue
        d = pd.Timestamp(r["date"])
        end = d + pd.Timedelta(days=HORIZON_DAYS)
        if end > today:
            continue
        window = px[(px.index > d) & (px.index <= end)]
        if window.empty:
            continue
        realized = window.min() / float(r["btc_price"]) - 1
        hit = realized <= DRAWDOWN_THRESHOLD
        log.at[i, "resolved_on"] = end.date().isoformat()
        log.at[i, "realized_min_pct"] = round(realized * 100, 2)
        if r["zone"] == "red":
            log.at[i, "outcome"] = "true_positive" if hit else "false_positive"
        else:
            log.at[i, "outcome"] = "false_negative" if hit else "true_negative"
    return log


def scoreboard(log: pd.DataFrame) -> str:
    done = log[log["outcome"].isin(
        ["true_positive", "false_positive",
         "true_negative", "false_negative"])]
    if done.empty:
        return "No resolved forecasts yet."
    tp = (done["outcome"] == "true_positive").sum()
    fp = (done["outcome"] == "false_positive").sum()
    fn = (done["outcome"] == "false_negative").sum()
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return (f"Resolved forecasts: {len(done)} | signals: {tp + fp} | "
            f"precision: {prec:.0%} | recall: {rec:.0%}")


def post_text(log: pd.DataFrame) -> str:
    last = log.iloc[-1]
    emoji = {"red": "🔴", "amber": "🟡", "green": "🟢"}[last["zone"]]
    return (
        f"{emoji} Crypto Market Fragility: "
        f"{last['fragility']}/100 — {last['date']}\n\n"
        f"{last['call']}\n\n"
        f"Top drivers today: {last['top_drivers']}\n\n"
        f"{scoreboard(log)}\n"
        f"The full forecast log is public and never edited after the fact."
    )


def main(resolve_only: bool = False):
    idx = pd.read_csv(DATA / "index.csv", index_col=0, parse_dates=True)
    log = load_log()

    if not resolve_only:
        log = append_today(idx, log)
    log = resolve(idx, log)

    log.to_csv(LOG_PATH, index=False)
    print(f"[+] log saved: {LOG_PATH} ({len(log)} entries)\n")
    print("-" * 60)
    print(post_text(log))
    print("-" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolve-only", action="store_true")
    main(ap.parse_args().resolve_only)
