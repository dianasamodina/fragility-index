"""
Configuration for the crypto market fragility index.

Methodology by Diana Samodina.

Version 2. Component signs and composition were corrected after
diagnostics on 2017-2026 history (see "What the data showed" in README).

Core idea: a market is most vulnerable not when it panics, but when it
is quiet. Calm breeds confidence, confidence breeds leverage, leverage
breeds fragility. In the literature this is Minsky's financial
instability hypothesis, also known as the volatility paradox. It is
well documented for traditional markets and barely quantified for crypto.

Changing the methodology means changing this file, not the code.
"""

from pathlib import Path

_here = Path(__file__).resolve().parent
# If this file sits in src/, the project root is one level up. If it sits
# next to the data (Colab, where every module lands in one folder), the
# root is that folder itself.
ROOT = _here.parent if _here.name == "src" else _here
DATA = ROOT / "data"
LOGS = ROOT / "logs"
DATA.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

# How far back to pull history.
START_DATE = "2017-01-01"

# Rolling window for each component's z-score, in days.
Z_WINDOW = 365

# Rolling window for mapping the composite onto a 0-100 scale.
RANK_WINDOW = 730

# Z-score clip, so a single outlier cannot drag the whole index.
Z_CLIP = 3.0

# ---------------------------------------------------------------------------
# Index components.
#
# sign = +1  : higher value means higher fragility
# sign = -1  : higher value means lower fragility (inverted)
# weight     : weight in the composite, normalised automatically
# auc        : measured standalone predictive power on history
#              (0.50 = useless). Kept as a record of why the component is
#              here, and so that degradation becomes visible.
#
# WEIGHTS ARE DELIBERATELY NOT OPTIMISED. Tuning weights to maximise a
# metric on the same history is the fastest way to build an index that
# explains the past perfectly and says nothing about the future. The two
# strongest components get more, the rest share equally. That is enough.
# ---------------------------------------------------------------------------
COMPONENTS = {
    # STRONGEST. Low implied volatility means market complacency.
    # Inverted: the original assumption was the opposite, and the data
    # showed that high IV accompanies a crash that already happened
    # rather than preceding one.
    "dvol": {"sign": -1, "weight": 0.30, "auc": 0.620,
             "label": "Options market calm (DVOL, inverted)"},

    # SECOND STRONGEST, and the only component whose power GROWS with
    # lag: AUC 0.604 on the signal day, 0.643 thirty days ahead of the
    # event. Contracting stablecoin supply means capital leaving.
    "stable_14d": {"sign": -1, "weight": 0.25, "auc": 0.604,
                   "label": "Stablecoin supply change, 14d"},

    # Low realised volatility. Same logic as DVOL.
    "rv30": {"sign": -1, "weight": 0.15, "auc": 0.555,
             "label": "Realised volatility 30d (inverted)"},

    # Drawdown from the 90-day high. Behaves as momentum.
    "dd90": {"sign": -1, "weight": 0.15, "auc": 0.561,
             "label": "Drawdown from 90-day high"},

    # Contracting DeFi TVL means deleveraging.
    "tvl_14d": {"sign": -1, "weight": 0.15, "auc": 0.549,
                "label": "DeFi TVL change, 14d"},
}

# ---------------------------------------------------------------------------
# Dropped components. Still collected (they may serve future versions and
# the paid tier) but excluded from the index.
#
#   funding      AUC 0.502 — the funding level alone carries no signal.
#                Worth retrying as rate of change rather than level.
#   ethbtc_30d   AUC 0.518 — too noisy as a risk-appetite proxy.
#   vix          not measured in time; revisit once data accumulates.
# ---------------------------------------------------------------------------
DROPPED = ("funding", "ethbtc_30d", "vix")

# ---------------------------------------------------------------------------
# Target event and signal threshold.
# ---------------------------------------------------------------------------
HORIZON_DAYS = 14            # forecast horizon
DRAWDOWN_THRESHOLD = -0.15   # what counts as a stress event
ALERT_THRESHOLD = 85         # index level that triggers a signal

# Walk-forward metrics measured for version 2 — a baseline to compare
# against when the methodology changes.
BASELINE = {
    "walkforward_auc": 0.652,
    "top_decile_lift": 2.21,
    "top_decile_hit_rate": 0.090,
    "base_rate": 0.041,
    "measured_on": "2017-09-27..2026-10-08",
}

# Known stress episodes, for eyeballing whether the index responds.
KNOWN_EVENTS = {
    "2020-03-12": "COVID crash",
    "2021-05-19": "May 2021 deleveraging",
    "2022-05-09": "Terra / UST collapse",
    "2022-06-13": "Celsius / 3AC",
    "2022-11-08": "FTX collapse",
    "2024-08-05": "Yen carry unwind",
}
