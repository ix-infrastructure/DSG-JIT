# DSG-JIT denoising audit

*Audit of the recent denoising experiments (exp40–exp43) and the DSG-JIT library, against the planned upgraded ORB-SLAM3 pipeline. Repository state: branch `experiments/denoising`, HEAD `c763fb8`, 2026-10-06.*

---

## 1. Verdict on the recent experiments

The four most recent experiments split into two families with fundamentally different objectives.

| Experiment | Outer objective | Ground truth used | Reported metric | Verdict |
|---|---|---|---|---|
| exp40 (two-phase) | fidelity to noisy measurements + smoothness + anchors | exact GT poses at 2 of every 100 frames (anchors) | per-edge measurement RMSE | **Valid denoiser, sparse oracle** — disclose anchors |
| exp41 (three-phase) | same as exp40 | same as exp40 | per-edge measurement RMSE | **Valid denoiser, sparse oracle** — disclose anchors |
| exp42 (SE-Sync inner solver) | squared error to **GT pose at every frame** | dense GT, every window | ATE / ARE | **Not a denoising result** |
| exp43 (SE-Sync, unbatched) | squared error to **GT pose at every frame** | dense GT, every window, **plus precision pooled over all 22 noise seeds** | ATE / ARE | **Not a denoising result** |

### 1.1 exp42 and exp43 optimize directly against ground truth

The outer loss is the distance between the reconstructed trajectory and the window's ground-truth poses:

```python
# exp43_sesync_structured_gn.py:586  (exp42: line 478)
def loss_fn(theta):
    R_star, t_star = sesync_inner_solve(theta, noisy_odom, kappa, omega, n, inner_cfg)
    loss_t = jnp.mean(jnp.sum((t_star - gt_t_rel) ** 2, axis=-1))
    loss_r = jnp.mean(jax.vmap(lambda Ra, Rb: jnp.sum((Ra - Rb) ** 2))(R_star, gt_R_rel))
    return loss_t + loss_r
```

Each window is passed the full ground-truth slice (`win_gt = gt_global[lo:hi]`, exp43:831). The optimizer adjusts the measurement corrections so the trajectory matches the answer; only the inner solver's structure and the 70-step Adam budget prevent it from reaching zero error. The improvement figures therefore measure how far a constrained optimizer moves toward ground truth it is given, not the ability to remove noise. A deployed system has no ground truth to optimize against.

**Provenance.** The dense-GT loss is present in exp42's first commit (`e088f12`, 2026-06-26, Rica Calubaquib) and was inherited by exp43 (`c4fa27d`, 2026-07-08). A later commit (`781843a`, 2026-06-29) changed only the line's coordinate frame and rotation metric. It may have been intended as an oracle upper bound; that should be confirmed with the author before drawing conclusions about intent.

**Secondary leak (exp43).** `_sigma_and_precision` estimates per-edge precision from diffs **pooled across all S = 22 noise realizations** of the same trajectory (exp43:505–517). A deployed system observes one realization, so this gives the precision estimate 22× more data than is ever available.

### 1.2 exp40 and exp41 are clean apart from disclosed anchors

Checked explicitly for subtler leaks:

- **Inner-solve initialization** is forward composition of the *noisy* window measurements (`x_init = _forward_compose_jit(origin, w_noisy)`, exp41:822) — not ground truth.
- **Noise model and auto-weights** are estimated from the noisy measurements only (`estimate_noise_*(noisy_np)`, exp41:666–673).
- **Anchors** are exact ground-truth poses at window positions `range(0, window, spacing)` plus the last frame — with window 100 and spacing 100, two per window (2% density). This is a genuine oracle and must be disclosed. A deployable variant should use noisy anchors (GNSS-like fixes) or loop closures instead of exact poses.

### 1.3 Consequences for work built on exp43

| Artifact | Status |
|---|---|
| Manuscript duplicate `paper/` (zips v3–v5): Claim 5 rewritten as "the low-noise boundary is a weighting artifact" | **Unsupported.** With GT in the loss, low noise is not hard. The original Claim 5 (exp41-based identifiability boundary) is the defensible one. Variant P columns in Table I, §E, and the per-edge heatmaps must be removed. The untouched original is at `~/Development/Ix/Real_Time_Odometry_Denoising/`. |
| YC application: "up to 96%" (seq 20, seed 16) and "~41% average" | **Both from exp43.** Defensible replacement: exp41, ~40% per-edge RMSE reduction at σ_t = 0.03, with sparse GT anchors stated. |
| 10.4 ms/pose throughput | A real runtime measurement, but of a method that cannot be deployed as written. |
| SE-Sync inner solver (rotation-only GN + closed-form translation, IFT backward) | **Still sound and worth keeping.** The problem is the objective around it, not the solver. |

The seed divergences recorded in `exp42_results_log.md` (e.g. ΔT ≈ −260% on isolated seeds) were observed under the GT-supervised objective; whether they persist under a proper objective is unknown.

---

## 2. Structural findings

### 2.1 Copy-pasted experiment lineage

exp30–exp43 are 15 standalone scripts totalling **14,472 lines** (600–1,316 each). Core machinery is redefined rather than shared:

| Symbol | Defined in |
|---|---|
| `build_denoiser` | 11 files |
| `compute_ate` | **10 files** |
| `@jax.custom_vjp` (IFT backward) | 9 files |
| `estimate_noise*` | 9 files |
| `compute_auto_weights` | 7 files |

The experiments import almost nothing from the library: exp42/43 import only `so3_exp` and `so3_log`; exp40/41 add a few `math3d` helpers and `sigma_to_weight`.

Both serious problems found so far are this failure mode — copies drifting apart with nothing checking them:

- **Metric drift.** exp41 reports per-edge *measurement* RMSE (`compute_per_pose_meas_error`); exp43 reports *trajectory* ATE/ARE (`delta_metric`). The manuscript's Table I placed them in shared columns as if they were the same quantity.
- **Objective drift.** exp42 replaced exp41's denoising objective with a GT-supervised one in a fresh copy of the code.

### 2.2 Testing

- Library: **27 tests** across `math3d`, manifold GN, SE(3) SLAM, loop closure, sliding window, voxel residuals and the world model.
- Denoising experiments: **0 tests.** None of the code that produced the paper's numbers is tested.

### 2.3 Numerics

- `jax_enable_x64` is never set; all library and experiment code runs in **float32** (exp41 casts explicitly in 15 places). Adequate for millimetre-scale per-edge errors, but a plausible contributor to failures in the SE-Sync line's dense 300×300 normal-equation solves. *Hypothesis, not demonstrated.*
- `so3_log` (`math3d.py:204`) handles small angles but has **no θ ≈ π branch**: as θ → π the factor θ/(2 sin θ) diverges while vee(R − Rᵀ) → 0, giving an ill-conditioned axis and unbounded gradients. This is the singularity `exp42_results_log.md` records working around. Harmless for frame-to-frame relative rotations; a real defect for absolute or window-relative orientations.

---

## 3. Library inventory

| Module | What it provides | Tested | Reuse value for the pipeline |
|---|---|---|---|
| `core/math3d.py` | `so3_exp/log`, `se3_exp`, compose, relative pose, left retraction | yes | **High** — Lie-group core for every stage (fix θ ≈ π first) |
| `optimization/solvers.py`, `jit_wrappers.py` | GD, damped Newton, GN, manifold GN; `JittedGN`, `JittedGNManifold` | yes | **High** — estimator scaffolding for recalibration and post-SLAM |
| `slam/measurements.py` | SE(3) odometry (incl. geodesic), temporal smoothness, landmark bearing/relative, range, prior residuals | yes | **High** — residual library for post-SLAM refinement |
| `world/voxel_grid.py` + voxel residuals | voxel grid spec, pose–voxel–point, voxel smoothness, point-observation residuals | partly | **Medium** — starting point for map refinement; only a 1-D (x-axis) neighbour-connectivity helper exists |
| `world/model.py`, `scene_graph.py`, `dynamic_scene_graph.py` | world model, scene graph | yes | Medium — map container if a mapping stage is added |
| `sensors/fusion.py` | `SensorFusionManager`, async multi-sensor dispatch | no | Medium — possible orchestrator for triggered modules |
| `sensors/imu.py` | `integrate_imu_naive` | no | **Low** — explicitly "not a replacement for proper IMU preintegration" |
| `sensors/camera.py` | intrinsics, frame containers, grayscale conversion | no | Low — no image processing |

### Gaps

1. **No robust loss functions** anywhere in the library (no Huber, Cauchy, Geman–McClure, Barron). ORB-SLAM3's real per-edge error is heavy-tailed (excess kurtosis +29 to +54), so the post-SLAM stage needs them.
2. **Implicit differentiation is not packaged.** The `custom_vjp` IFT backward — the method's core contribution — exists only inside 9 experiment scripts.
3. **No IMU preintegration.** Required for IMU-derived blur kernels and for online calibration.
4. **No canonical trajectory metrics.** ATE has 10 implementations in experiments. New, validated versions exist in `experiments/oasis/` (`eval_ate.py`, `analyze_error_shape.py`) but are not in the library.
5. **No image or photometric processing.** All pre-SLAM image modules are new code.

---

## 4. Recycle map onto the planned pipeline

| Pipeline stage | Reusable now | Must be built |
|---|---|---|
| **Pre-SLAM: low light** (first) | `experiments/oasis/`: `inject_image_noise.py` (frozen + per-frame Gaussian, tested), `run_fpn_suite.sh` harness pattern, `eval_ate.py`, `analyze_error_shape.py`; `run_oasis.sh` + `oasis-orbslam3:24.04` image on lambda | Photon-transfer low-light simulator (exposure stops, Poisson shot, gain-scaled read noise, quantization, bounded by MT9V034's 1×–4× gain); correction arms (lowered-FAST-threshold control, gamma/brightness, brightness + denoise); degradation trigger |
| **Pre-SLAM: recalibration** | `math3d`, manifold GN, `JittedGN`, IFT pattern from experiments | IMU preintegration; observability/degeneracy checks; drift detector; a hook to rebuild ORB-SLAM3's stereo rectification (it reads calibration once at startup) |
| **Pre-SLAM: motion blur** | `math3d` for trajectory interpolation | IMU preintegration; blur synthesis along GT trajectory; EuRoC exposure time (unknown — check rosbag headers). Lowest priority per the research |
| **Post-SLAM: trajectory** | exp41 objective (fidelity + smoothness + anchors); SE-Sync inner solver and IFT backward from exp42/43; windowed MAD noise estimation and MAP auto-weights; `analyze_error_shape.py` | Robust loss; realistic anchors (noisy fixes or loop closures, not exact GT); leakage test; library packaging |
| **Post-SLAM: map** | voxel grid + voxel residuals; world model; web viewer | ORB-SLAM3 map export (sparse points + keyframes); full 3-D voxel connectivity; densification if a mesh is wanted (ORB-SLAM3 produces no mesh) |

---

## 5. Recommendations, in order

1. **Settle the exp42/43 question with the author**, then correct the manuscript duplicate (revert Claim 5, remove Variant P) and the YC text if not yet submitted.
2. **Add a leakage test before any new denoising result is reported**: run the denoiser with ground truth replaced by random poses and assert identical output. This catches both failure modes in §1.
3. **Extract a `dsg_jit.denoise` module**: one objective definition, one metric set (per-edge RMSE *and* ATE, clearly named), the IFT solver, and tests. Port exp41 into it.
4. **Re-run "Variant P" honestly**: SE-Sync inner solver + exp41 objective + single-realization precision. This is a small change with existing code and is the only way to learn what the SE-Sync line actually achieves.
5. **Add an anchors-only baseline for exp41.** Its exact ground-truth anchors (2 per 100 frames) can reduce error on their own. Without a correction that uses only the anchors and no denoising objective, the gain from denoising cannot be separated from the gain from the oracle. (Raised in the companion research report, *Upgraded ORB SLAM3 correction pipeline*.)
6. **Add robust kernels** to `slam/measurements.py` before applying the denoiser to real ORB-SLAM3 output.
7. **Fix `so3_log` near θ = π** and add a test at θ ∈ {π − 10⁻³, π − 10⁻⁶}.
8. **Implement IMU preintegration** before starting recalibration or blur work.
9. **Raise run counts to ≥10 per condition**, reporting medians and trajectory validity separately, per the framework research.
