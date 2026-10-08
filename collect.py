"""
Fetch raw data and write it to data/raw.csv.

Usage:
    python src/collect.py              # refresh everything
    python src/collect.py --only dvol  # one source only

Each source is cached in its own file under data/raw/, so one failing
source neither breaks the build nor forces a full re-download.
"""

from __future__ import annotations

import argparse
import sys
import traceback

import pandas as pd

from config import DATA
from sources import ALL_SOURCES

RAW_DIR = DATA / "raw"
RAW_DIR.mkdir(exist_ok=True)


def collect(only: str | None = None) -> pd.DataFrame:
    names = [only] if only else list(ALL_SOURCES)
    failures = []

    for name in names:
        fn = ALL_SOURCES.get(name)
        if fn is None:
            print(f"[!] unknown source: {name}")
            continue
        try:
            print(f"[.] {name} ...", flush=True)
            s = fn()
            s.to_frame(name).to_csv(RAW_DIR / f"{name}.csv")
            print(f"[+] {name}: {len(s)} points, "
                  f"{s.index.min().date()} to {s.index.max().date()}")
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"[x] {name}: {exc}")
            traceback.print_exc(limit=1)

    # Merge whatever was collected, including previously cached sources.
    frames = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        frames.append(pd.read_csv(path, index_col=0, parse_dates=True))

    if not frames:
        print("No sources collected.")
        sys.exit(1)

    raw = pd.concat(frames, axis=1).sort_index()
    raw.to_csv(DATA / "raw.csv")
    print(f"\nSaved: {DATA / 'raw.csv'}  "
          f"({raw.shape[0]} rows, {raw.shape[1]} columns)")
    if failures:
        print(f"Failed: {', '.join(failures)} — "
              f"the index will be built from the remaining components.")
    return raw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="collect a single source")
    collect(ap.parse_args().only)
