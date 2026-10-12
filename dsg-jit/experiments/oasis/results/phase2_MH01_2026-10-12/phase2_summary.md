# Phase 2 results — low-light SLAM grid (MH01)

## Verdict (pre-registered decision rules)

**UNDETERMINED (outside pre-registered table) — denoising beats gain-only at L5, but L1-L2 equivalence to clean is NOT confirmed.**

Primary contrast, C4 vs C2 at L5: Wilcoxon p = 0.0098 (Holm 0.0293), valid 10/10 vs 9/10.

Runs: 200 darkened + 20 clean. Clean floor: 20/20 valid, ATE 39.8 ± 5.1 mm.

Machine-load check on clean runs: Spearman(load, ATE) = -0.42 (p = 0.06).

## Per-cell outcomes

| level | arm | valid | median tracked | median ATE valid (mm) | mean resets |
|---|---|---|---|---|---|
| L1 | C0 | 10/10 | 0.987 | 41.5 | 0.5 |
| L1 | C4 | 10/10 | 0.987 | 41.6 | 0.5 |
| L2 | C0 | 10/10 | 0.976 | 44.2 | 0.7 |
| L2 | C4 | 10/10 | 0.999 | 44.1 | 0.4 |
| L3 | C0 | 10/10 | 0.999 | 44.3 | 0.3 |
| L3 | C1 | 10/10 | 0.975 | 45.5 | 0.6 |
| L3 | C2 | 10/10 | 0.987 | 46.3 | 0.5 |
| L3 | C4 | 10/10 | 0.999 | 41.5 | 0.4 |
| L4 | C0 | 10/10 | 0.999 | 38.9 | 0.3 |
| L4 | C1 | 10/10 | 0.975 | 51.5 | 0.7 |
| L4 | C2 | 10/10 | 0.987 | 48.5 | 0.5 |
| L4 | C4 | 10/10 | 0.999 | 40.9 | 0.1 |
| L5 | C0 | 9/10 | 0.975 | 45.8 | 1.1 |
| L5 | C1 | 0/10 | 0.758 | nan | 3.8 |
| L5 | C2 | 9/10 | 0.987 | 51.6 | 0.7 |
| L5 | C4 | 10/10 | 0.999 | 43.9 | 0.1 |
| L6 | C0 | 0/10 | 0.161 | nan | 0.1 |
| L6 | C1 | 0/10 | 0.745 | nan | 7.5 |
| L6 | C2 | 0/10 | 0.758 | nan | 5.0 |
| L6 | C4 | 10/10 | 0.999 | 60.2 | 0.1 |

### C4 (gain + denoise) vs C2 (gain only) — negative favours denoising

| level | n | valid A/B | median Δ composite (A−B) | Wilcoxon p | Holm p | Fisher p (valid) |
|---|---|---|---|---|---|---|
| L3 | 10 | 10/10 | -2.0 mm | 0.131 | 0.131 | 1.000 |
| L4 | 10 | 10/10 | -7.0 mm | 0.027 | 0.055 | 1.000 |
| L5 | 10 | 10/9 | -11.2 mm | 0.010 | 0.029 | 1.000 |
| L6 | 10 | 10/0 | -884.2 mm | 0.002 | 0.008 | 0.000 |

### C2 (gain) vs C1 (lowered FAST threshold) — negative favours gain

| level | n | valid A/B | median Δ composite (A−B) | Wilcoxon p | Holm p | Fisher p (valid) |
|---|---|---|---|---|---|---|
| L3 | 10 | 10/10 | +2.9 mm | 0.131 | 0.262 | 1.000 |
| L4 | 10 | 10/10 | -2.0 mm | 0.695 | 0.695 | 1.000 |
| L5 | 10 | 9/0 | -889.4 mm | 0.004 | 0.016 | 0.000 |
| L6 | 10 | 0/0 | -12.9 mm | 0.016 | 0.047 | 1.000 |

### C2 (gain) vs C0 (none)

| level | n | valid A/B | median Δ composite (A−B) | Wilcoxon p | Holm p | Fisher p (valid) |
|---|---|---|---|---|---|---|
| L3 | 10 | 10/10 | +0.3 mm | 0.770 | 1.000 | 1.000 |
| L4 | 10 | 10/10 | +11.5 mm | 0.006 | 0.018 | 1.000 |
| L5 | 10 | 9/9 | +1.5 mm | 0.846 | 1.000 | 1.000 |
| L6 | 10 | 0/0 | -592.2 mm | 0.002 | 0.008 | 1.000 |

### C4 (gain + denoise) vs C0 (none)

| level | n | valid A/B | median Δ composite (A−B) | Wilcoxon p | Holm p | Fisher p (valid) |
|---|---|---|---|---|---|---|
| L1 | 10 | 10/10 | -0.2 mm | 0.846 | 1.000 | 1.000 |
| L2 | 10 | 10/10 | +1.6 mm | 0.922 | 1.000 | 1.000 |
| L3 | 10 | 10/10 | -1.2 mm | 0.322 | 0.967 | 1.000 |
| L4 | 10 | 10/10 | +2.0 mm | 0.193 | 0.773 | 1.000 |
| L5 | 10 | 10/9 | -7.8 mm | 0.037 | 0.186 | 1.000 |
| L6 | 10 | 10/0 | -1.47 (validity-driven) | 0.002 | 0.012 | 0.000 |

### C1 (threshold) vs C0 (none)

| level | n | valid A/B | median Δ composite (A−B) | Wilcoxon p | Holm p | Fisher p (valid) |
|---|---|---|---|---|---|---|
| L3 | 10 | 10/10 | -0.3 mm | 0.695 | 0.695 | 1.000 |
| L4 | 10 | 10/10 | +11.7 mm | 0.002 | 0.008 | 1.000 |
| L5 | 10 | 0/9 | +892.6 mm | 0.004 | 0.008 | 0.000 |
| L6 | 10 | 0/0 | -578.5 mm | 0.002 | 0.008 | 1.000 |

## Realistic regime (gain-compensated): equivalence to clean

TOST, margin ±5 mm on valid-run ATE. p < 0.05 means equivalent.

| level | arm | valid | mean ATE (mm) | Δ vs clean (mm) | TOST p |
|---|---|---|---|---|---|
| L1 | C0 | 10/10 | 40.8 | +1.1 | 0.019 |
| L1 | C4 | 10/10 | 42.1 | +2.4 | 0.084 |
| L2 | C0 | 10/10 | 43.7 | +3.9 | 0.281 |
| L2 | C4 | 10/10 | 43.7 | +4.0 | 0.289 |

