# Degradation framework roadmap

Drafted 2026-10-19. Companion to `Degradation correction framework.md` (design and evidence) and `Degradation models handoff.pdf` (intern brief). Durations are rough estimates; run counts are exact.

## Where we stand

| Degradation | Real evidence | Verdict | Correction |
|---|---|---|---|
| Calibration / timing (stereo rotation, time offset, IMU scale) | 380-run EuRoC sweep: hurts at realistic tiers, silently (tracking stays valid) | **Main target** | Post-SLAM batch re-estimation in DSG-JIT + residual monitor |
| Uniform darkness (4Seasons night) | Night 12/30 good vs day 10/10; saturation-safe gain → 20/30, failures 260 → 46 (p = 0.0003) | **Viable, partly solved** | Gated gain (works); quantization loss remains (27 m vs 7 m day) |
| Patchy darkness (EuRoC MH04/05) | Gain cuts dark-window drift 17–20% | Solved by the same gain | Same module |
| Image denoising, motion blur, flicker, FPN, IMU white noise | No measurable SLAM cost at realistic levels | Dropped (report as negatives) | None |
| Trajectory-level Gaussian noise | errchar/errnorm: correlated multi-timescale drift with jumps; AR(1) fails in 40/40 | Not a valid model | None |

## Phase 0: Close out characterization (≈ 1–2 weeks, lambda only)

| # | Task | Size | Done when |
|---|---|---|---|
| 0.1 | Commit current code, results and reports | – | On `experiments/denoising` |
| 0.2 | **TUM-VI real miscalibration baseline**: room1–6, three conditions (calibrated / scale+misalignment removed / fully raw incl. biases), 10 seeds | 9.2 GB download; 180 runs | Real (not injected) IMU miscalibration cost measured |
| 0.3 | Re-sweep IMU scale with sourced tiers (gyro 1/3/9%, accel 1/3/4%) on MH01/MH03 | 60 runs | Tier table no longer provisional |
| 0.4 | Trace the "bad imu" map resets in C1/C5 on 4Seasons night | Log analysis + ~10 runs | Cause known; explains why only gain helps |
| 0.5 | Rebuild synthetic darkening (scale + black clip + 8-bit quantization, measured σ), calibrate to 4Seasons night statistics | Script + 30 runs | Synthetic night reproduces real failure rate (~60%) |
| 0.6 | Second 4Seasons night recording (different route) for gain generalization | ~15 GB, 60 runs | Gain effect holds on a held-out sequence |

**Gate:** 0.2 decides whether IMU-scale recovery is worth building (if raw TUM-VI is barely worse, drop isf from Phase 1).

## Phase 1: Calibration re-estimation in DSG-JIT (core contribution, ≈ 4–6 weeks)

1. **Export from ORB-SLAM3** (engineering prerequisite): keyframe poses, map points, keyframe-to-point observations, and IMU preintegration intervals. Today we save only the trajectory. Requires changes in the OASIS fork.
2. **Identifiability first** (intern Q3): which of stereo yaw, accel scale, time offset are separable under EuRoC MAV motion vs car motion (4Seasons)? Fisher information on the exported problems.
3. **Bilevel estimator**: outer = calibration parameters (rst, toff, isf; then rbc, imis); inner = bundle adjustment + IMU factors in DSG-JIT; objective = measurement self-consistency, no ground truth; gradients by implicit differentiation.
4. **Validate on the existing 380 sweep runs**, then on TUM-VI raw (0.2), whose correct values are published.

**Success criteria:** recovered error within 0.05° (rst), 1 ms (toff), 0.2% (isf); re-run ATE within ~5 mm of clean; no change on clean runs. Baseline to beat: Kalibr-style offline recalibration and OpenVINS online calibration (see `Calibration recovery literature check.md`).

## Phase 2: Silent-miscalibration monitor (≈ 2–3 weeks, can overlap Phase 1)

- Per-parameter residual statistics (reprojection by image row/column, IMU preintegration residuals by axis, time-shift cross-correlation of gyro vs visual rotation).
- ROC on the 380-run sweep with clean runs as negatives; target ≥ 90% detection at tier 2 with ≤ 5% false alarms.
- Output feeds Phase 1 (which parameter to re-estimate) and the pre-SLAM gating.

## Phase 3: Photometric module (≈ 2–3 weeks)

- Package the gated saturation-safe gain (brightness monitor → gain; identity on bright frames).
- Use the 0.5 synthetic model to test what can still be gained beyond gain (quantization-aware feature extraction, temporal accumulation). Only pursue if synthetic and real agree.
- Report the upstream message honestly: most of the night loss happens before 8-bit quantization (exposure/gain control).

## Phase 4: Integration and evaluation (≈ 3–4 weeks)

- Full pipeline: monitors → gated pre-corrections → ORB-SLAM3 → DSG-JIT quality-weighted re-optimization + calibration re-estimation.
- Datasets: EuRoC (injected sweep + MH04/05), 4Seasons day/night, TUM-VI raw.
- Ablation per module; validity-first reporting (§5 of the design doc); every module must leave clean input unchanged.

## Publication plan

| Paper | Content | Ready after |
|---|---|---|
| A | Calibration errors silently degrade stereo-inertial SLAM; GT-free recovery with differentiable bilevel optimization | Phases 0.2–0.3, 1, 2 |
| B (or section of A) | Measured degradation study: which degradations matter for VI-SLAM, with negative results (denoising, blur, Gaussian trajectory noise) | Phase 0 (data mostly in hand) |
| C | Low-light module | Only if Phase 3 finds a gain beyond simple brightness correction |

The old IEEE paper's Gaussian-trajectory-noise results are **not** salvageable as-is (errnorm, 2026-10-16).

## Decisions needed

1. **Paper A first?** It has the strongest evidence and the clearest DSG-JIT contribution.
2. **OK to modify the OASIS ORB-SLAM3 fork** for the keyframe/observation export (Phase 1.1)?
3. **Intern split**: suggested Q3 (identifiability) and Q4 (residual monitor) to interns; Phase 1 estimator in-house.
4. **Lambda storage**: `/` (holding `/home`) is at 100% (9.6 GB free); `/data` has 314 GB. Phase 0 needs ~25 GB, all on `/data`.
