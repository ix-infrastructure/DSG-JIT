# Trajectory-level error shape

## ATE versus the clean nondeterminism floor

| condition | valid | ATE mean (mm) | ATE std (mm) | Δ vs clean (mm) | Δ / clean std | Mann-Whitney p |
|---|---|---|---|---|---|---|
| clean | 5/5 | 39.19 | 3.96 | — | — | — |
| realistic_high | 5/5 | 41.23 | 4.75 | +2.05 | +0.52 | 0.421 |
| realistic_low | 5/5 | 42.37 | 4.04 | +3.18 | +0.80 | 0.310 |
| stress | 3/3 | 48.69 | 5.82 | +9.50 | +2.40 | 0.071 |

## Shape of the relative (edge) pose error — what the SE(3) denoiser sees

Per-run statistics, averaged over runs. Std ratio is vs. clean; p is Mann-Whitney on per-run std.

### trans (mm)

| condition | axis | mean | std | std ratio vs clean | std p | skew | ex. kurtosis | lag-1 ACF |
|---|---|---|---|---|---|---|---|---|
| clean | x | -0.0391 | 2.1681 | — | — | +0.44 | +38.50 | -0.049 |
| clean | y | +0.1137 | 1.1276 | — | — | -0.81 | +38.07 | +0.045 |
| clean | z | -0.1110 | 2.6269 | — | — | -0.08 | +29.11 | -0.044 |
| realistic_high | x | -0.0378 | 2.0967 | 0.967 | 0.008 | +0.19 | +33.16 | -0.041 |
| realistic_high | y | +0.1135 | 1.0945 | 0.971 | 0.548 | +0.07 | +25.81 | +0.053 |
| realistic_high | z | -0.1111 | 2.5583 | 0.974 | 0.222 | -0.01 | +23.84 | -0.040 |
| realistic_low | x | -0.0407 | 2.2362 | 1.031 | 0.548 | +0.18 | +77.08 | -0.041 |
| realistic_low | y | +0.1171 | 1.1434 | 1.014 | 0.841 | +0.41 | +63.86 | +0.049 |
| realistic_low | z | -0.1195 | 2.6501 | 1.009 | 0.690 | +0.65 | +50.22 | -0.038 |
| stress | x | -0.0388 | 2.2519 | 1.039 | 0.071 | +0.28 | +37.24 | -0.071 |
| stress | y | +0.1193 | 1.2085 | 1.072 | 0.143 | +0.61 | +30.05 | +0.015 |
| stress | z | -0.1151 | 3.8594 | 1.469 | 0.036 | -0.83 | +380.83 | -0.195 |

### rot (mrad)

| condition | axis | mean | std | std ratio vs clean | std p | skew | ex. kurtosis | lag-1 ACF |
|---|---|---|---|---|---|---|---|---|
| clean | x | -0.0155 | 0.1662 | — | — | -1.75 | +33.80 | -0.061 |
| clean | y | -0.0020 | 0.2733 | — | — | -0.65 | +33.96 | -0.086 |
| clean | z | +0.0050 | 0.1704 | — | — | +0.67 | +54.33 | -0.175 |
| realistic_high | x | -0.0157 | 0.1685 | 1.014 | 0.841 | -0.97 | +32.38 | -0.064 |
| realistic_high | y | -0.0021 | 0.2653 | 0.971 | 0.222 | -0.75 | +32.81 | -0.068 |
| realistic_high | z | +0.0051 | 0.1734 | 1.018 | 1.000 | +0.52 | +79.33 | -0.150 |
| realistic_low | x | -0.0158 | 0.1958 | 1.178 | 0.690 | -5.16 | +266.48 | -0.034 |
| realistic_low | y | -0.0021 | 0.2987 | 1.093 | 1.000 | -1.11 | +159.53 | -0.066 |
| realistic_low | z | +0.0053 | 0.1782 | 1.046 | 0.548 | +0.92 | +86.84 | -0.213 |
| stress | x | -0.0154 | 0.1759 | 1.058 | 0.143 | -1.46 | +44.23 | -0.079 |
| stress | y | -0.0015 | 0.2716 | 0.994 | 1.000 | -0.01 | +52.53 | -0.080 |
| stress | z | +0.0048 | 0.1843 | 1.082 | 0.393 | +1.06 | +61.24 | -0.199 |

## Absolute error memory (descriptive only)

Caution: absolute error integrates edge error, so even perfectly white edge noise
produces a random walk with long memory here. This column cannot distinguish white
from biased edge error -- use the lag-1 ACF of the RELATIVE error for whiteness.

| condition | mean correlation time of |abs err| (frames) | at 20 Hz (s) |
|---|---|---|
| clean | 118 | 5.9 |
| realistic_high | 120 | 6.0 |
| realistic_low | 124 | 6.2 |
| stress | 151 | 7.6 |

