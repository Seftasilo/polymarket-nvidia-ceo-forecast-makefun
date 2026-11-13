# Polymarket NVIDIA CEO forecast

Forecast each trading market of Polymarket's "Next CEO of NVIDIA?" event, minute
by minute, 4 hours ahead, as five quantiles (5%, 25%, 50%, 75%, 95%) per minute.

## How it works

`main.py` (Python 3.13, standard library only) reads each `<id>.in` case and
writes `<id>.out`. Per market, from that case's own 72 hours of minute prices:

1. At nine anchor horizons (1, 2, 5, 10, 20, 40, 80, 160, 240 minutes), the
   empirical quantiles of the h-step price change, with exponential recency
   weights (half-life 180 min) so the distribution tracks the current regime
   and forgets the market-launch transient. Prices sit still most minutes,
   then jump, and jump size grows slower than sqrt(h) (mild mean reversion).
2. Interpolate each quantile across h = 1..240 in log-h.
3. Recentre the spreads on the checkpoint price (persistence median — a
   drifting median scored far worse), shrink spreads by 0.9, and floor the
   outer quantiles at 0.4x the house baseline's spread so a quiet recent
   window never promises zero risk.
4. Clamp to [0, 1], enforce non-decreasing quantiles, round to 6 decimals.

## Practice results (local scorer, item-weighted mean pinball loss, p00–p05)

| model | loss |
|---|---|
| house baseline (persistence, sigma 0.0025) | 0.363589 |
| this entry | **0.286550** (-21%) |

It beats the baseline on every one of the six practice cases (p00 0.472 vs
0.536, p01 0.314 vs 0.380, p02 0.355 vs 0.408, p03 0.300 vs 0.369, p04 0.207
vs 0.299, p05 0.131 vs 0.238). Coverage 90% is 0.94 (nominal 0.9), median
error 0.77 — same as the baseline, since the median is persistence.

Deterministic: two runs on the same case write identical bytes. Runtime
~0.23 s per case (limit 8 s). Output ~200–260 KB per case (limit 1 MiB).

Run locally: `mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py`
(the kit's practice cases live in `kit/polymarket-nvidia-ceo-forecast-01-kit/input/`).
