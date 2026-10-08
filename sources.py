"""
Data collectors. Every source is free and needs no API key.

Each function returns a pandas Series with a DatetimeIndex (UTC,
normalised to midnight) and one numeric column. If a source is
unreachable it raises, and the caller decides what to do.

Run this on your own machine: sandboxed environments usually block
these domains.
"""

from __future__ import annotations

import time
from io import StringIO

import pandas as pd
import requests

from config import START_DATE

UA = {"User-Agent": "fragility-index/0.2 (research)"}
TIMEOUT = 30


def _get(url: str, params: dict | None = None, retries: int = 3):
    """GET with retries and a polite pause. Returns requests.Response."""
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=TIMEOUT)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


def _to_daily(s: pd.Series, how: str = "last") -> pd.Series:
    """Resample any series onto a daily UTC grid."""
    s = s.sort_index()
    s.index = pd.to_datetime(s.index, utc=True).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")]
    return s.resample("D").agg(how).ffill(limit=3)


# ---------------------------------------------------------------------------
# 1. Deribit DVOL — BTC implied volatility
# ---------------------------------------------------------------------------
def fetch_dvol(currency: str = "BTC") -> pd.Series:
    """Deribit implied volatility index, daily closes."""
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    start = int(pd.Timestamp(START_DATE).timestamp() * 1000)
    end = int(pd.Timestamp.utcnow().timestamp() * 1000)

    rows = []
    cursor = start
    # The API caps candles per request, so walk in 200-day windows.
    step = 200 * 24 * 3600 * 1000
    while cursor < end:
        chunk_end = min(cursor + step, end)
        r = _get(url, {
            "currency": currency,
            "start_timestamp": cursor,
            "end_timestamp": chunk_end,
            "resolution": "1D",
        })
        rows.extend(r.json().get("result", {}).get("data", []))
        cursor = chunk_end
        time.sleep(0.3)

    if not rows:
        raise RuntimeError("Deribit returned no DVOL data")

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
    s = pd.Series(df["close"].values,
                  index=pd.to_datetime(df["ts"], unit="ms"),
                  name="dvol")
    return _to_daily(s)


# ---------------------------------------------------------------------------
# 2. Binance — perpetual funding rates
# ---------------------------------------------------------------------------
def fetch_funding(symbols=("BTCUSDT", "ETHUSDT")) -> pd.Series:
    """
    Mean perpetual funding rate, annualised.

    Funding settles every 8 hours, so 1095 times per year. High positive
    funding means crowded longs, which means fragility.
    """
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    start = int(pd.Timestamp(START_DATE).timestamp() * 1000)

    per_symbol = []
    for sym in symbols:
        rows, cursor = [], start
        while True:
            r = _get(url, {"symbol": sym, "startTime": cursor, "limit": 1000})
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            last_ts = batch[-1]["fundingTime"]
            if last_ts <= cursor or len(batch) < 1000:
                break
            cursor = last_ts + 1
            time.sleep(0.25)

        if not rows:
            continue
        df = pd.DataFrame(rows)
        s = pd.Series(df["fundingRate"].astype(float).values,
                      index=pd.to_datetime(df["fundingTime"], unit="ms"))
        per_symbol.append(_to_daily(s * 1095, how="mean"))

    if not per_symbol:
        raise RuntimeError("Binance returned no funding history")

    out = pd.concat(per_symbol, axis=1).mean(axis=1)
    out.name = "funding"
    return out


# ---------------------------------------------------------------------------
# 3. Daily prices, with a fallback venue
# ---------------------------------------------------------------------------
def _klines_binance(symbol: str) -> pd.Series:
    url = "https://api.binance.com/api/v3/klines"
    start = int(pd.Timestamp(START_DATE).timestamp() * 1000)

    rows, cursor = [], start
    while True:
        r = _get(url, {"symbol": symbol, "interval": "1d",
                       "startTime": cursor, "limit": 1000})
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor or len(batch) < 1000:
            break
        cursor = last_ts + 1
        time.sleep(0.25)

    if not rows:
        raise RuntimeError("empty response")
    df = pd.DataFrame(rows)
    return pd.Series(df[4].astype(float).values,
                     index=pd.to_datetime(df[0], unit="ms"))


# Binance returns 451 to US IPs. Coinbase is a US venue and always works
# from there, so it serves as the fallback.
_COINBASE_PAIRS = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD"}


def _klines_coinbase(symbol: str) -> pd.Series:
    pair = _COINBASE_PAIRS.get(symbol)
    if pair is None:
        raise RuntimeError(f"no Coinbase pair for {symbol}")

    url = f"https://api.exchange.coinbase.com/products/{pair}/candles"
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp.utcnow().tz_localize(None)

    chunks, cursor = [], start
    # Coinbase caps a request at 300 candles, so walk in windows.
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=295), end)
        r = _get(url, {"granularity": 86400,
                       "start": cursor.isoformat(),
                       "end": chunk_end.isoformat()})
        chunks.extend(r.json() or [])
        cursor = chunk_end
        time.sleep(0.35)

    if not chunks:
        raise RuntimeError("empty response")
    df = pd.DataFrame(chunks, columns=["ts", "low", "high",
                                       "open", "close", "volume"])
    return pd.Series(df["close"].astype(float).values,
                     index=pd.to_datetime(df["ts"], unit="s"))


def fetch_klines(symbol: str = "BTCUSDT") -> pd.Series:
    """
    Daily closing prices. Binance first, Coinbase on failure.

    The fallback matters because Binance returns 451 to US IPs and the
    daily automated run happens on US servers.
    """
    errors = []
    for name, fn in (("binance", _klines_binance),
                     ("coinbase", _klines_coinbase)):
        try:
            s = fn(symbol)
            if len(s):
                s.name = symbol.lower()
                return _to_daily(s)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    raise RuntimeError(f"no prices for {symbol} — " + " | ".join(errors))


# ---------------------------------------------------------------------------
# 4. DefiLlama — aggregate stablecoin supply
# ---------------------------------------------------------------------------
def fetch_stablecoin_supply() -> pd.Series:
    """Total market cap of USD-pegged stablecoins."""
    r = _get("https://stablecoins.llama.fi/stablecoincharts/all")
    data = r.json()
    if not data:
        raise RuntimeError("DefiLlama returned no stablecoin data")

    idx, vals = [], []
    for row in data:
        total = row.get("totalCirculatingUSD") or {}
        v = total.get("peggedUSD")
        if v is None:
            # The key occasionally differs — sum whatever is there.
            v = sum(x for x in total.values() if isinstance(x, (int, float)))
        idx.append(pd.to_datetime(int(row["date"]), unit="s"))
        vals.append(float(v))

    return _to_daily(pd.Series(vals, index=idx, name="stable_supply"))


# ---------------------------------------------------------------------------
# 5. DefiLlama — aggregate TVL
# ---------------------------------------------------------------------------
def fetch_defi_tvl() -> pd.Series:
    """Total value locked across all chains."""
    r = _get("https://api.llama.fi/v2/historicalChainTvl")
    data = r.json()
    if not data:
        raise RuntimeError("DefiLlama returned no TVL data")

    s = pd.Series(
        [float(row["tvl"]) for row in data],
        index=[pd.to_datetime(int(row["date"]), unit="s") for row in data],
        name="defi_tvl",
    )
    return _to_daily(s)


# ---------------------------------------------------------------------------
# 6. VIX, with a fallback source
# ---------------------------------------------------------------------------
def _vix_from_stooq() -> pd.Series:
    r = _get("https://stooq.com/q/d/l/", {"s": "^vix", "i": "d"})
    df = pd.read_csv(StringIO(r.text))
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    return pd.Series(df["Close"].values,
                     index=pd.to_datetime(df["Date"]), name="vix")


def _vix_from_fred() -> pd.Series:
    r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv",
             {"id": "VIXCLS"})
    df = pd.read_csv(StringIO(r.text))
    date_col, val_col = df.columns[0], df.columns[1]
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])
    return pd.Series(df[val_col].values,
                     index=pd.to_datetime(df[date_col]), name="vix")


def fetch_vix() -> pd.Series:
    """
    CBOE volatility index. Trading days only, weekends forward-filled.

    Stooq first: FRED does not respond from Google Colab or GitHub
    Actions servers. FRED stays as the fallback.
    """
    errors = []
    for name, fn in (("stooq", _vix_from_stooq), ("fred", _vix_from_fred)):
        try:
            s = fn()
            s = s[s.index >= pd.Timestamp(START_DATE)]
            if len(s):
                return _to_daily(s).ffill(limit=5)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    raise RuntimeError("VIX unavailable — " + " | ".join(errors))


ALL_SOURCES = {
    "dvol": fetch_dvol,
    "funding": fetch_funding,
    "btcusdt": lambda: fetch_klines("BTCUSDT"),
    "ethusdt": lambda: fetch_klines("ETHUSDT"),
    "stable_supply": fetch_stablecoin_supply,
    "defi_tvl": fetch_defi_tvl,
    "vix": fetch_vix,
}
