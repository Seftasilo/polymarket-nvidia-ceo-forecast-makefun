#!/usr/bin/env python3
"""Forecast Polymarket's next NVIDIA CEO prices, minute by minute, 4 hours ahead.

Reads every <id>.in case in the input directory and writes <id>.out forecasts:
for each market and each of the 240 minutes after the checkpoint, five quantiles
(5%, 25%, 50%, 75%, 95%) of that minute's Yes price.

Method (per market, from the case's own 72h of minute prices only):
  1. At nine anchor horizons h in {1, 2, 5, 10, 20, 40, 80, 160, 240} minutes,
     take the empirical quantiles of the h-step price change over the trailing
     history, with exponential recency weights (half-life 180 minutes) so the
     distribution tracks the current regime and forgets the market-launch
     transient. This captures what a normal random walk misses: prices sit
     still most minutes, then jump, and the jump size grows slower than
     sqrt(h) (mild mean reversion, lag-1 return autocorrelation ~ -0.12).
  2. Interpolate each quantile across h = 1..240 in log-h.
  3. Recentre the spreads on the checkpoint price x (persistence median:
     short-horizon drift is noise, and a drifting median scored far worse),
     shrinking the spreads by 0.9 (overlapping windows slightly inflate the
     empirical tails), and floor the outer quantiles at 0.4x the house
     baseline's spread so a quiet recent window never promises zero risk.
  4. Clamp to [0, 1], enforce non-decreasing quantiles, round to 6 decimals.

Practice (local scorer, item-weighted mean pinball loss over p00-p05):
  house baseline 0.363589; this model 0.286550 (-21%). It beats the baseline
  on every one of the six practice cases. Pure standard library, no network,
  fully deterministic: two runs on the same case write identical bytes.

Run locally: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from bisect import bisect_left

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
ANCHORS = (1, 2, 5, 10, 20, 40, 80, 160, 240)
LOG_ANCHORS = [math.log(h) for h in ANCHORS]

HALF_LIFE_MIN = 180.0   # recency half-life for the empirical change distribution
SPREAD_SCALE = 0.9      # shrink empirical spreads (overlapping windows inflate tails)
FLOOR_FRAC = 0.4        # outer quantiles at least this x the baseline's spread
BASE_SIGMA = 0.0025     # house baseline spread, price units per sqrt-minute
MIN_WINDOWS = 50        # need at least this many h-step changes, else baseline


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0.0)."""
    r = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return r + 0.0


def weighted_quantiles(changes, weights, taus):
    """Quantiles of `changes` under `weights` (both lists, same order).

    Sorts by value, walks the cumulative weight, and takes the first value
    whose cumulative weight reaches tau * total weight.
    """
    pairs = sorted(zip(changes, weights))
    total = 0.0
    cum = []
    for _, w in pairs:
        total += w
        cum.append(total)
    vals = [c for c, _ in pairs]
    out = []
    for tau in taus:
        i = bisect_left(cum, tau * total)
        if i >= len(vals):
            i = len(vals) - 1
        out.append(vals[i])
    return out


def interp_logh(xs, ys, x):
    """Linear interpolation of ys at x, with xs strictly increasing (log-h grid)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect_left(xs, x)
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def forecast_market(prices, horizon):
    """240 rows x 5 quantiles for one market, from its minute prices."""
    # The case holds at most 72 hours (4,320 minutes); cap defensively so a
    # longer history can never slow the 8-second run or stale the weights.
    prices = prices[-4320:]
    n = len(prices)
    if n == 0:
        raise ValueError("market has no prices")
    x = prices[-1]
    log2 = math.log(2.0)
    # Empirical quantiles of the h-step change at each anchor horizon.
    anchor_levels = []
    for h in ANCHORS:
        if n - h < MIN_WINDOWS:
            anchor_levels.append([x + z * BASE_SIGMA * math.sqrt(h) for z in Z])
            continue
        changes = [prices[i] - prices[i - h] for i in range(h, n)]
        # Weight by the change window's end time: newest windows weigh most.
        weights = [math.exp(-(n - 1 - i) * log2 / HALF_LIFE_MIN) for i in range(h, n)]
        qs = weighted_quantiles(changes, weights, QUANTILES)
        anchor_levels.append([x + q for q in qs])
    # Interpolate each quantile across h = 1..horizon in log-h. Hoist the
    # per-quantile anchor columns out of the row loop: rebuilding them for
    # every row would cost 240 x 5 list builds per market for nothing.
    cols = [[a[j] for a in anchor_levels] for j in range(5)]
    rows = []
    for step in range(1, horizon + 1):
        lh = math.log(step)
        row = [interp_logh(LOG_ANCHORS, cols[j], lh) for j in range(5)]
        # Recentre spreads on the checkpoint price and shrink them.
        med = row[2]
        row = [x + (row[j] - med) * SPREAD_SCALE for j in range(5)]
        row[2] = x
        # Floor the outer quantiles at a fraction of the baseline's spread.
        half = FLOOR_FRAC * BASE_SIGMA * math.sqrt(step)
        row[0] = min(row[0], x - half)
        row[4] = max(row[4], x + half)
        # Enforce non-decreasing quantiles, clamp, round.
        fixed = []
        prev = 0.0
        for v in row:
            v = price6(v)
            if v < prev:
                v = prev
            fixed.append(v)
            prev = v
        rows.append(fixed)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in inputs:
        case_id = name[:-len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        if case["quantiles"] != QUANTILES:
            raise ValueError(
                "%s asks for quantiles %s, not %s"
                % (name, case["quantiles"], QUANTILES)
            )
        horizon = case["horizon_minutes"]
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], horizon)}
            for m in case["markets"]
        ]
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
