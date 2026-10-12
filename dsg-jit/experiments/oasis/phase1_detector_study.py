#!/usr/bin/env python3
"""Phase 1 of the low-light experiment: detector-level study, no SLAM runs.

For each frame, noise seed, darkness level and correction arm, reproduce what
ORB-SLAM3's front end would see and measure it against the clean frame.
Synthetic darkening is pixel-aligned with the clean frame, so repeatability
and descriptor agreement are computed EXACTLY, with no geometry.

DARKENING (EMVA/ELD linear camera model, measured MH01 constants)
  Scene dimmed by dEV stops; analog gain g = min(2^dEV, gain_max) compensates
  as far as it can; output brightness scales by a = g / 2^dEV. The clean frame
  already carries sensor noise, so we add only the variance DIFFERENCE between
  the target and the (scaled) inherited noise:
     dvar = g^2 K x (2^-dEV - 4^-dEV) + (g K r)^2 (1 - 4^-dEV)
  Gaussian with exact variance (shape approximate at very low photon counts).
  Applied to the RAW frames (sensor domain), same dEV on both cameras,
  independent noise per camera; then quantized at 10 bits (0.25 DN) or 8 bits
  and clipped to [1, 255] (MH01 floors at DN 1). Rectification follows.

CORRECTION ARMS (applied to the darkened frame, before rectification)
  C0 none                        C1 FAST thresholds scaled by a (config only)
  C2 linear gain 1/a             C3 gamma, shared across stereo, median -> 75 DN
  C4 C2 + non-local means        C5 C2 + heavier NLM  (stand-in for BM3D ceiling)
  C6 CLAHE                       C8 10-bit path: quantize at 0.25 DN, then gain

EXTRACTION, matched to the OASIS/ORB-SLAM3 fork's source
  * rectification: cv::stereoRectify(..., CALIB_ZERO_DISPARITY, alpha=-1) with
    the inverse of Stereo.T_c1_c2, initUndistortRectifyMap, INTER_LINEAR remap
    (Settings.cc precomputeRectificationMaps)
  * 8-level pyramid, scale 1.2, INTER_LINEAR resize from the previous level
  * FAST per 35-px cell at iniThFAST, falling back to minThFAST when a cell is
    empty (ORBextractor.cc:889-910) -- EXACT, including cell geometry
  * per-level feature budget N(1-f)/(1-f^L), f = 1/1.2 (ORBextractor.cc:440)
  * spatial distribution: APPROXIMATION of DistributeOctTree (best response per
    grid cell, grid sized to the level budget, topped up by response)
  * orientation by intensity centroid (r = 15) on the unblurred level
  * rBRIEF via cv2.ORB.compute on its 7x7 sigma=2 blurred pyramid -- the same
    blur ORB-SLAM3 uses (ORBextractor.cc:1197); verified to keep given angles
  * stereo: descriptor distance < (TH_HIGH+TH_LOW)/2 = 75, row band
    +/-2*scale(octave_R), octave within +/-1, 0 <= disparity <= fx
    (Frame.cc ComputeStereoMatches). SAD refinement and the distance-outlier
    filter are NOT reproduced, so counts are upper bounds on accepted matches.

METRICS per (frame, seed, level, arm, camera)
  n_kp, cells forced to the fallback threshold, cells empty even after fallback,
  repeatability (clean keypoints recovered within 2*scale px, octave +/-1),
  spurious keypoints (no clean counterpart), descriptor Hamming distance at the
  CLEAN keypoint locations (median, fraction <= 50 = TH_LOW), stereo matches.
"""
import argparse, csv, glob, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2

EDGE, CELL, PATCH, HALF = 19, 35, 31, 15
ARMS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C8"]
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# --------------------------------------------------------------- config ---

def load_cfg(yaml):
    fs = cv2.FileStorage(yaml, cv2.FILE_STORAGE_READ)
    g = lambda k: fs.getNode(k).real()
    K1 = np.array([[g("Camera1.fx"), 0, g("Camera1.cx")], [0, g("Camera1.fy"), g("Camera1.cy")], [0, 0, 1]])
    K2 = np.array([[g("Camera2.fx"), 0, g("Camera2.cx")], [0, g("Camera2.fy"), g("Camera2.cy")], [0, 0, 1]])
    D1 = np.array([g("Camera1.k1"), g("Camera1.k2"), g("Camera1.p1"), g("Camera1.p2")])
    D2 = np.array([g("Camera2.k1"), g("Camera2.k2"), g("Camera2.p1"), g("Camera2.p2")])
    # Everything float64: OpenCV 4.6's stereoRectify rejects mixed precisions
    # (cvRodrigues2 type check), and FileStorage may return the matrix as float32.
    return dict(K1=K1.astype(np.float64), K2=K2.astype(np.float64),
                D1=D1.astype(np.float64), D2=D2.astype(np.float64),
                T=np.asarray(fs.getNode("Stereo.T_c1_c2").mat(), dtype=np.float64),
                w=int(g("Camera.width")), h=int(g("Camera.height")),
                nfeat=int(g("ORBextractor.nFeatures")), sf=g("ORBextractor.scaleFactor"),
                nlev=int(g("ORBextractor.nLevels")),
                ini=int(g("ORBextractor.iniThFAST")), mn=int(g("ORBextractor.minThFAST")))


def rect_maps(c):
    Ti = np.linalg.inv(c["T"]); size = (c["w"], c["h"])
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(c["K1"], c["D1"], c["K2"], c["D2"], size,
                                                Ti[:3, :3], Ti[:3, 3],
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1,
                                                newImageSize=size)
    ml = cv2.initUndistortRectifyMap(c["K1"], c["D1"], R1, P1[:3, :3], size, cv2.CV_32FC1)
    mr = cv2.initUndistortRectifyMap(c["K2"], c["D2"], R2, P2[:3, :3], size, cv2.CV_32FC1)
    return ml, mr, float(P1[0, 0])

# ------------------------------------------------------------- darkening ---

def darken(x, ev, rng, K, read_e, gain_max):
    """Continuous-valued darkened frame and its output attenuation a."""
    if ev == 0:
        return x.astype(np.float64), 1.0
    g = min(2.0 ** ev, gain_max); a = g / 2.0 ** ev
    xs = np.maximum(x.astype(np.float64), 0.0)
    dvar = g * g * K * xs * (2.0 ** -ev - 4.0 ** -ev) + (g * K * read_e) ** 2 * (1 - 4.0 ** -ev)
    return xs * a + rng.standard_normal(x.shape) * np.sqrt(dvar), a


def quant(y, step):
    return np.clip(np.round(y / step) * step, 1.0, 255.0)


def restored_sigma(ev, K, read_e, gain_max, mu=75.0):
    """Model noise (DN) at intensity mu after brightness is restored."""
    g = min(2.0 ** ev, gain_max); a = g / 2.0 ** ev
    return g * K * np.sqrt(mu / K / 2.0 ** ev + read_e ** 2) / a


def to_u8(img):
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def nlm(img_u8, h, search):
    return cv2.fastNlMeansDenoising(img_u8, None, h=float(max(h, 1.0)),
                                    templateWindowSize=7, searchWindowSize=search)


_CLAHE = None
def apply_arms(yL, yR, a, sig_r, cfg):
    """Return {arm: (left_u8, right_u8, iniTh, minTh)} for one darkened pair."""
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ini, mn = cfg["ini"], cfg["mn"]
    l8, r8 = quant(yL, 1.0), quant(yR, 1.0)
    out = {"C0": (to_u8(l8), to_u8(r8), ini, mn),
           "C1": (to_u8(l8), to_u8(r8), max(1, round(ini * a)), max(1, round(mn * a)))}
    gl, gr = to_u8(l8 / a), to_u8(r8 / a)
    out["C2"] = (gl, gr, ini, mn)
    m = float(np.median(np.concatenate([l8.ravel(), r8.ravel()])))
    gam = np.log(75 / 255) / np.log(max(m, 1.0) / 255) if m < 75 else 1.0
    out["C3"] = (to_u8(255 * (l8 / 255) ** gam), to_u8(255 * (r8 / 255) ** gam), ini, mn)
    out["C4"] = (nlm(gl, sig_r, 21), nlm(gr, sig_r, 21), ini, mn)
    out["C5"] = (nlm(gl, 1.3 * sig_r, 35), nlm(gr, 1.3 * sig_r, 35), ini, mn)
    out["C6"] = (_CLAHE.apply(to_u8(l8)), _CLAHE.apply(to_u8(r8)), ini, mn)
    out["C8"] = (to_u8(quant(yL, 0.25) / a), to_u8(quant(yR, 0.25) / a), ini, mn)
    return out

# ------------------------------------------------------------ extraction ---

_UMAX_OFF = None
def _circle():
    global _UMAX_OFF
    if _UMAX_OFF is None:
        dy, dx = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1]
        m = dx * dx + dy * dy <= HALF * HALF
        _UMAX_OFF = (dy[m], dx[m])
    return _UMAX_OFF


def ic_angle(img, xs, ys):
    dy, dx = _circle()
    v = img[ys[:, None] + dy[None, :], xs[:, None] + dx[None, :]].astype(np.float64)
    return np.degrees(np.arctan2((v * dy).sum(1), (v * dx).sum(1))) % 360.0


def distribute(pts, resp, n, w, h):
    """Approximate DistributeOctTree: best response per grid cell, then top-up."""
    if len(pts) <= n:
        return np.arange(len(pts))
    nx = max(1, int(round(np.sqrt(n * w / h)))); ny = max(1, int(round(n / nx)))
    cid = (np.minimum((pts[:, 1] * ny / h).astype(int), ny - 1) * nx
           + np.minimum((pts[:, 0] * nx / w).astype(int), nx - 1))
    order = np.lexsort((-resp, cid))
    first = order[np.r_[True, cid[order][1:] != cid[order][:-1]]]
    keep = first[np.argsort(-resp[first])][:n]
    if len(keep) < n:
        rest = np.setdiff1d(np.argsort(-resp), keep, assume_unique=False)
        keep = np.r_[keep, rest[:n - len(keep)]]
    return keep


def pyramid(img, nlev, sf):
    """Successive INTER_LINEAR resize from the previous level, as ORB-SLAM3
    ComputePyramid does. Shared by extract() and desc_at() so both see
    identical level images."""
    lv = [img]
    for i in range(1, nlev):
        lv.append(cv2.resize(lv[-1], (int(round(img.shape[1] / sf ** i)), int(round(img.shape[0] / sf ** i))),
                             interpolation=cv2.INTER_LINEAR))
    return lv


def extract(img, ini, mn, cfg, orb):
    nlev, sf = cfg["nlev"], cfg["sf"]
    f = 1.0 / sf
    per = cfg["nfeat"] * (1 - f) / (1 - f ** nlev)
    budget, tot = [], 0
    for lv in range(nlev - 1):
        budget.append(int(round(per))); tot += budget[-1]; per *= f
    budget.append(max(cfg["nfeat"] - tot, 0))

    kps, cells, fallback, dead = [], 0, 0, 0
    pyr = pyramid(img, nlev, sf)
    for lv in range(nlev):
        lvl = pyr[lv]
        H, Wd = lvl.shape
        x0 = y0 = EDGE - 3; x1, y1 = Wd - EDGE + 3, H - EDGE + 3
        nC, nR = int((x1 - x0) / CELL), int((y1 - y0) / CELL)
        if nC < 1 or nR < 1:
            continue
        wc, hc = int(np.ceil((x1 - x0) / nC)), int(np.ceil((y1 - y0) / nR))
        pts, rsp = [], []
        for i in range(nR):
            iy = y0 + i * hc; my = min(iy + hc + 6, y1)
            if iy >= y1 - 3:
                continue
            for j in range(nC):
                ix = x0 + j * wc; mx = min(ix + wc + 6, x1)
                if ix >= x1 - 6:
                    continue
                cells += 1
                sub = lvl[iy:my, ix:mx]
                k = cv2.FastFeatureDetector_create(ini, True).detect(sub)
                if not k:
                    fallback += 1
                    k = cv2.FastFeatureDetector_create(mn, True).detect(sub)
                    if not k:
                        dead += 1
                for p in k:
                    pts.append((p.pt[0] + ix, p.pt[1] + iy)); rsp.append(p.response)
        if not pts:
            continue
        pts, rsp = np.array(pts), np.array(rsp)
        keep = distribute(pts - [x0, y0], rsp, budget[lv], x1 - x0, y1 - y0)
        p = pts[keep]; xs = np.round(p[:, 0]).astype(int); ys = np.round(p[:, 1]).astype(int)
        ok = (xs >= HALF) & (ys >= HALF) & (xs < Wd - HALF) & (ys < H - HALF)
        ang = np.zeros(len(p)); ang[ok] = ic_angle(lvl, xs[ok], ys[ok])
        s = sf ** lv
        for q, a, r in zip(p[ok], ang[ok], rsp[keep][ok]):
            kps.append(cv2.KeyPoint(float(q[0] * s), float(q[1] * s), PATCH * s, float(a), float(r), lv, len(kps)))
    if not kps:
        return np.zeros((0, 2)), np.zeros(0, int), np.zeros((0, 32), np.uint8), dict(
            cells=cells, fallback=fallback, dead=dead)
    kps, des = orb.compute(img, kps)
    if des is None:
        des = np.zeros((0, 32), np.uint8); kps = []
    xy = np.array([k.pt for k in kps]) if kps else np.zeros((0, 2))
    oc = np.array([k.octave for k in kps], int) if kps else np.zeros(0, int)
    return xy, oc, des, dict(cells=cells, fallback=fallback, dead=dead, angles=[k.angle for k in kps])


def desc_at(img, xy, oc, angles, orb, sf):
    """Descriptors of `img` at given keypoints (orientation recomputed on img)."""
    if len(xy) == 0:
        return np.zeros((0, 32), np.uint8), np.zeros(0, bool)
    kps = [cv2.KeyPoint(float(x), float(y), PATCH * sf ** o, 0.0, 0, int(o), i)
           for i, ((x, y), o) in enumerate(zip(xy, oc))]
    # orientation on the corrupted level image, as ORB-SLAM3 would compute it
    pyr = pyramid(img, int(oc.max()) + 1, sf)
    for o in np.unique(oc):
        s = sf ** o
        lvl = pyr[o]
        idx = np.nonzero(oc == o)[0]
        xs = np.round(xy[idx, 0] / s).astype(int); ys = np.round(xy[idx, 1] / s).astype(int)
        ok = (xs >= HALF) & (ys >= HALF) & (xs < lvl.shape[1] - HALF) & (ys < lvl.shape[0] - HALF)
        ang = np.zeros(len(idx)); ang[ok] = ic_angle(lvl, xs[ok], ys[ok])
        for t, a_ in zip(idx, ang):
            kps[t].angle = float(a_)
    out, des = orb.compute(img, kps)
    full = np.zeros((len(xy), 32), np.uint8); has = np.zeros(len(xy), bool)
    if des is not None:
        for k, d in zip(out, des):
            full[k.class_id] = d; has[k.class_id] = True
    return full, has


def hamming(a, b):
    return POPCOUNT[np.bitwise_xor(a, b)].sum(-1).astype(int)


def stereo_matches(L, R, sf, maxd, th=75):
    (xl, ol, dl), (xr, orr, dr) = L, R
    if len(xl) == 0 or len(xr) == 0:
        return np.zeros((0, 3))
    rband = 2.0 * sf ** orr
    dy = np.abs(xl[:, None, 1] - xr[None, :, 1]) <= rband[None, :]
    disp = xl[:, None, 0] - xr[None, :, 0]
    m = dy & (disp >= 0) & (disp <= maxd) & (np.abs(ol[:, None] - orr[None, :]) <= 1)
    i, j = np.nonzero(m)
    if len(i) == 0:
        return np.zeros((0, 3))
    d = hamming(dl[i], dr[j])
    order = np.lexsort((d, i))
    best = order[np.r_[True, i[order][1:] != i[order][:-1]]]
    best = best[d[best] < th]
    # (left x, left y, disparity) of every accepted match
    return np.column_stack([xl[i[best], 0], xl[i[best], 1], xl[i[best], 0] - xr[j[best], 0]])


def correct_matches(M, Mref, tol_xy=2.0, tol_d=1.0):
    """Matches whose left keypoint and disparity agree with a CLEAN-frame match.
    Pixel alignment makes the clean match a per-feature ground truth."""
    if len(M) == 0 or len(Mref) == 0:
        return 0
    from scipy.spatial import cKDTree
    t = cKDTree(Mref[:, :2]); n = 0
    for x, y, d in M:
        c = t.query_ball_point((x, y), tol_xy)
        n += any(abs(Mref[k, 2] - d) <= tol_d for k in c)
    return int(n)


def repeat_stats(cxy, coc, xy, oc, sf):
    if len(cxy) == 0 or len(xy) == 0:
        return 0.0, (1.0 if len(xy) else 0.0)
    from scipy.spatial import cKDTree
    t = cKDTree(xy)
    tol = 2.0 * sf ** coc
    found = np.zeros(len(cxy), bool)
    for idx, (p, o, r) in enumerate(zip(cxy, coc, tol)):
        cand = t.query_ball_point(p, r)
        found[idx] = any(abs(oc[c] - o) <= 1 for c in cand)
    tc = cKDTree(cxy)
    spur = np.array([len(tc.query_ball_point(p, 2.0 * sf ** o)) == 0 for p, o in zip(xy, oc)])
    return float(found.mean()), float(spur.mean())

# ------------------------------------------------------------------ work ---

G = {}
def _init(cfg, maps, a):
    G.update(cfg=cfg, ml=maps[0], mr=maps[1], fx=maps[2], args=a,
             orb=cv2.ORB_create(nfeatures=cfg["nfeat"], scaleFactor=cfg["sf"], nlevels=cfg["nlev"],
                                edgeThreshold=EDGE, patchSize=PATCH))


def rect(img, m):
    return cv2.remap(img, m[0], m[1], cv2.INTER_LINEAR)


def process_frame(job):
    fi, fL, fR = job
    cfg, a, orb, sf = G["cfg"], G["args"], G["orb"], G["cfg"]["sf"]
    rawL, rawR = cv2.imread(fL, 0), cv2.imread(fR, 0)
    cL, cR = rect(rawL, G["ml"]), rect(rawR, G["mr"])
    ref = {cam: extract(img, cfg["ini"], cfg["mn"], cfg, orb) for cam, img in (("L", cL), ("R", cR))}
    ref_M = stereo_matches(ref["L"][:3], ref["R"][:3], sf, G["fx"])
    rows = []
    for seed in range(a.seeds):
        for ev in a.levels:
            rngL = np.random.default_rng([seed, fi, ev, 0]); rngR = np.random.default_rng([seed, fi, ev, 1])
            yL, att = darken(rawL, ev, rngL, a.K, a.read_e, a.gain_max)
            yR, _ = darken(rawR, ev, rngR, a.K, a.read_e, a.gain_max)
            sig = restored_sigma(ev, a.K, a.read_e, a.gain_max)
            for arm, (l8, r8, ini, mn) in apply_arms(yL, yR, att, sig, cfg).items():
                if arm not in a.arms:
                    continue
                L, R = rect(l8, G["ml"]), rect(r8, G["mr"])
                ex = {"L": extract(L, ini, mn, cfg, orb), "R": extract(R, ini, mn, cfg, orb)}
                M = stereo_matches(ex["L"][:3], ex["R"][:3], sf, G["fx"])
                st_ok = correct_matches(M, ref_M)
                for cam, img in (("L", L), ("R", R)):
                    cxy, coc, cdes, cst = ref[cam]
                    xy, oc, des, est = ex[cam]
                    rep, spur = repeat_stats(cxy, coc, xy, oc, sf)
                    dd, has = desc_at(img, cxy, coc, cst.get("angles", []), orb, sf)
                    hd = hamming(cdes[has], dd[has]) if has.any() else np.array([256])
                    rows.append(dict(frame=fi, seed=seed, level=ev, arm=arm, cam=cam,
                                     atten=att, ini=ini, mn=mn,
                                     n_kp=len(xy), n_kp_clean=len(cxy),
                                     frac_fallback=est["fallback"] / max(est["cells"], 1),
                                     frac_dead=est["dead"] / max(est["cells"], 1),
                                     repeat=rep, spurious=spur,
                                     desc_med=float(np.median(hd)),
                                     desc_le50=float((hd <= 50).mean()),
                                     stereo=len(M), stereo_correct=st_ok,
                                     stereo_clean=len(ref_M)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True); ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=10, help="use every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--levels", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--K", type=float, default=0.0186, help="DN per electron (measured, MH01)")
    ap.add_argument("--read-e", type=float, default=30.0, help="read noise, electrons (measured 28-33)")
    ap.add_argument("--gain-max", type=float, default=4.0, help="MT9V034 analog gain ceiling")
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 4))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = load_cfg(a.yaml); maps = rect_maps(cfg)
    fL = sorted(glob.glob(os.path.join(a.seq, "mav0", "cam0", "data", "*.png")))
    names = {os.path.basename(f) for f in glob.glob(os.path.join(a.seq, "mav0", "cam1", "data", "*.png"))}
    fL = [f for f in fL if os.path.basename(f) in names][::a.stride]
    if a.max_frames:
        fL = fL[:a.max_frames]
    jobs = [(i, f, os.path.join(a.seq, "mav0", "cam1", "data", os.path.basename(f))) for i, f in enumerate(fL)]

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(cfg, maps, a)) as ex:
        for r in ex.map(process_frame, jobs, chunksize=1):
            rows.extend(r)
    with open(os.path.join(a.out, "phase1_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # aggregate: mean over frames, seeds, cameras -- per (level, arm)
    agg = {}
    keys = ["n_kp", "frac_fallback", "frac_dead", "repeat", "spurious", "desc_med", "desc_le50"]
    sk = ["stereo_ratio", "stereo_correct_ratio", "stereo_precision"]
    for r in rows:
        k = (r["level"], r["arm"]); agg.setdefault(k, {m: [] for m in keys + sk})
        for m in keys:
            agg[k][m].append(r[m])
        if r["cam"] == "L":
            agg[k]["stereo_ratio"].append(r["stereo"] / max(r["stereo_clean"], 1))
            agg[k]["stereo_correct_ratio"].append(r["stereo_correct"] / max(r["stereo_clean"], 1))
            agg[k]["stereo_precision"].append(r["stereo_correct"] / max(r["stereo"], 1))
    summary = {f"L{lv}_{arm}": {m: float(np.mean(v)) for m, v in d.items()} for (lv, arm), d in sorted(agg.items())}
    meta = dict(frames=len(jobs), seeds=a.seeds, levels=a.levels, arms=a.arms, K=a.K, read_e=a.read_e,
                gain_max=a.gain_max, fx_rect=maps[2],
                restored_sigma_dn={f"L{ev}": restored_sigma(ev, a.K, a.read_e, a.gain_max) for ev in a.levels})
    with open(os.path.join(a.out, "phase1_summary.json"), "w") as f:
        json.dump({"meta": meta, "by_level_arm": summary}, f, indent=1)

    print(f"frames={len(jobs)} seeds={a.seeds}  rows={len(rows)}")
    hdr = f"{'cell':<8}" + "".join(f"{m:>10}" for m in ["n_kp", "fallback", "dead", "repeat", "spurious",
                                                        "desc_le50", "stereo", "st_corr", "st_prec"])
    print(hdr)
    for (lv, arm), d in sorted(agg.items()):
        print(f"L{lv}_{arm:<5}" + "".join(f"{np.mean(d[m]):>10.3f}" if m != "n_kp" else f"{np.mean(d[m]):>10.0f}"
                                         for m in ["n_kp", "frac_fallback", "frac_dead", "repeat", "spurious",
                                                   "desc_le50", "stereo_ratio", "stereo_correct_ratio",
                                                   "stereo_precision"]))


if __name__ == "__main__":
    main()
