#!/usr/bin/env python3
"""Quantitative evaluation of 3D ultrasound reconstruction quality.

Provides reproducible numbers so different reconstruction pipelines /
hyper-parameters can be compared without "eyeballing" slices.

Three families of metrics:

1. Coverage statistics (intrinsic)
   - Fill ratio, mean / std of integration weight, ratio of high-weight
     voxels (well-sampled vs barely sampled).

2. Sharpness statistics (intrinsic)
   - Tenengrad (mean |∇I|^2) — higher means crisper edges.
   - Volume entropy of the non-zero histogram — higher means richer
     intensity distribution (less smeared).
   - Per-axis slice SSIM — average SSIM between consecutive slices;
     too low = noisy, too high = over-smoothed; useful as a *relative*
     indicator across runs.

3. Self-consistency (extrinsic, requires --split mode)
   - Rebuild two volumes from disjoint halves of the frame set
     (interleaved odd/even or random split), compare them on the
     intersection mask via NCC and RMSE.
   - This is the most informative single metric for "is the new pipeline
     actually putting energy in the same place?".

Usage:
  # Intrinsic only, single volume
  python3 evaluate_volume.py --volume data/reconstructions/pbm_v3_07mm.npy

  # With explicit weight file
  python3 evaluate_volume.py --volume V.npy --weight V_weight.npy

  # Compare two volumes (e.g. v3 baseline vs v4 candidate, same bounds)
  python3 evaluate_volume.py --volume V_new.npy --baseline V_old.npy

  # Self-consistency split (rebuilds twice with PBM)
  python3 evaluate_volume.py --split \
      --scan_dir  data/scans/scan_20260407_150839 \
      --calib_dir data/calibration \
      --voxel_size 0.0007 --elevation_sigma_mm 2.5 --intensity_min 15
"""

import argparse
import json
import os
import sys
import time
import numpy as np

# Allow running from any cwd (sibling imports)
_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsic metrics
# ─────────────────────────────────────────────────────────────────────────────

def coverage_stats(weight):
    nz_mask = weight > 0
    n_nz = int(nz_mask.sum())
    n_total = int(weight.size)
    fill = n_nz / max(n_total, 1)

    nz_w = weight[nz_mask]
    well_thr = np.percentile(nz_w, 50) if n_nz else 0.0
    well = float(np.mean(nz_w >= well_thr)) if n_nz else 0.0

    return dict(
        fill_ratio=float(fill),
        n_nonzero=n_nz,
        n_total=n_total,
        weight_mean=float(nz_w.mean()) if n_nz else 0.0,
        weight_median=float(np.median(nz_w)) if n_nz else 0.0,
        weight_p90=float(np.percentile(nz_w, 90)) if n_nz else 0.0,
        well_sampled_frac=well,
    )


def tenengrad(volume, mask):
    """Mean squared gradient magnitude over masked voxels.

    Uses central finite differences on the volume (not just mask), then
    averages |∇I|^2 inside mask. Higher → crisper.
    """
    vol = volume.astype(np.float32)
    gx = np.zeros_like(vol)
    gy = np.zeros_like(vol)
    gz = np.zeros_like(vol)
    gx[1:-1, :, :] = 0.5 * (vol[2:, :, :] - vol[:-2, :, :])
    gy[:, 1:-1, :] = 0.5 * (vol[:, 2:, :] - vol[:, :-2, :])
    gz[:, :, 1:-1] = 0.5 * (vol[:, :, 2:] - vol[:, :, :-2])
    g2 = gx * gx + gy * gy + gz * gz
    if mask.sum() == 0:
        return 0.0
    return float(g2[mask].mean())


def entropy_nonzero(volume, mask, bins=64):
    if mask.sum() == 0:
        return 0.0
    v = volume[mask].astype(np.float64)
    if v.max() <= v.min() + 1e-9:
        return 0.0
    hist, _ = np.histogram(v, bins=bins, range=(v.min(), v.max()))
    p = hist.astype(np.float64) / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def slice_ssim_axis(volume, mask, axis):
    """Average SSIM between successive slices along `axis`.

    Light-weight SSIM (no Gaussian window) for speed. Each pair only
    counts if both slices have at least 0.5% mask overlap.
    """
    n = volume.shape[axis]
    if n < 2:
        return 0.0
    vals = []
    for i in range(n - 1):
        if axis == 0:
            a = volume[i]; b = volume[i + 1]
            ma = mask[i]; mb = mask[i + 1]
        elif axis == 1:
            a = volume[:, i]; b = volume[:, i + 1]
            ma = mask[:, i]; mb = mask[:, i + 1]
        else:
            a = volume[:, :, i]; b = volume[:, :, i + 1]
            ma = mask[:, :, i]; mb = mask[:, :, i + 1]
        m = ma & mb
        if m.sum() < 0.005 * m.size:
            continue
        x = a[m].astype(np.float64); y = b[m].astype(np.float64)
        if len(x) < 8:
            continue
        mux, muy = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        cov = ((x - mux) * (y - muy)).mean()
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        ssim = (((2 * mux * muy) + c1) * ((2 * cov) + c2)) / \
               (((mux ** 2 + muy ** 2) + c1) * ((vx + vy) + c2))
        vals.append(float(ssim))
    return float(np.mean(vals)) if vals else 0.0


def intrinsic_metrics(volume, weight):
    nz = weight > 0
    out = dict(
        shape=list(volume.shape),
        intensity_min=float(volume[nz].min()) if nz.any() else 0.0,
        intensity_max=float(volume[nz].max()) if nz.any() else 0.0,
        intensity_mean=float(volume[nz].mean()) if nz.any() else 0.0,
    )
    out.update(coverage_stats(weight))
    out['tenengrad'] = tenengrad(volume, nz)
    out['entropy'] = entropy_nonzero(volume, nz)
    out['slice_ssim_axis0'] = slice_ssim_axis(volume, nz, axis=0)
    out['slice_ssim_axis1'] = slice_ssim_axis(volume, nz, axis=1)
    out['slice_ssim_axis2'] = slice_ssim_axis(volume, nz, axis=2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Extrinsic: pairwise comparison on shared mask
# ─────────────────────────────────────────────────────────────────────────────

def cross_metrics(vol_a, w_a, vol_b, w_b):
    if vol_a.shape != vol_b.shape:
        return dict(error=f'shape mismatch {vol_a.shape} vs {vol_b.shape}')
    common = (w_a > 0) & (w_b > 0)
    n = int(common.sum())
    if n < 100:
        return dict(error='overlap mask too small (<100 voxels)')
    a = vol_a[common].astype(np.float64)
    b = vol_b[common].astype(np.float64)
    a0 = a - a.mean(); b0 = b - b.mean()
    denom = np.sqrt((a0 ** 2).sum() * (b0 ** 2).sum())
    ncc = float((a0 * b0).sum() / denom) if denom > 0 else 0.0
    rmse = float(np.sqrt(((a - b) ** 2).mean()))
    mae = float(np.abs(a - b).mean())
    return dict(
        overlap_voxels=n,
        overlap_frac=float(n / common.size),
        ncc=ncc,
        rmse=rmse,
        mae=mae,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Split-and-rebuild self-consistency
# ─────────────────────────────────────────────────────────────────────────────

def split_consistency(scan_dir, calib_dir, voxel_size,
                      elevation_sigma_mm, intensity_min,
                      force_threshold=1.0, pose_percentile=0.0,
                      seed=0,
                      smooth_window=0, smooth_order=3,
                      time_lag_ms=0.0, auto_lag=False,
                      algo='v1', trilinear=True,
                      adaptive_elev=False, adaptive_alpha=0.05,
                      n_sigma_bins=4,
                      pixel_weight_fn=None,
                      tgc_eq=False, tgc_gamma=0.5,
                      despeckle_method='none', despeckle_ksize=3,
                      shadow_w=False, edge_w=False,
                      poses_npz=None):
    """Rebuild two PBM volumes from disjoint halves, compare on overlap.

    Frames are interleaved (even / odd) so each half samples the trajectory
    uniformly — purer test of whether the algorithm produces consistent
    geometry given less data.
    """
    from voxel_reconstruct import load_calibration, load_scan_data, auto_bounds
    from pbm_reconstruct import (pbm_compound, pbm_compound_v2,
                                  filter_pose_outliers)
    from pose_utils import load_scan_data_smooth, estimate_image_pose_lag
    from image_preproc import (apply_tgc_eq, despeckle, shadow_weight,
                                edge_weight, compose_xform, compose_weight,
                                transform_frames)

    print('Loading calibration...')
    T_tool0_probe, T_probe_us, px_x, px_y, roi = load_calibration(calib_dir)
    print(f'  Pixel size: {px_x*1000:.4f} × {px_y*1000:.4f} mm')

    print('Loading scan data...')
    use_smooth = smooth_window > 0 or time_lag_ms != 0.0 or auto_lag
    if use_smooth:
        lag_s = time_lag_ms / 1000.0
        if auto_lag:
            lag_s = estimate_image_pose_lag(scan_dir, roi=roi, max_lag_s=0.2,
                                            verbose=True)
        win = smooth_window if smooth_window > 0 else 1
        frames, poses = load_scan_data_smooth(
            scan_dir, roi=roi, force_threshold=force_threshold,
            sg_window=win, sg_order=smooth_order, time_lag_s=lag_s)
    else:
        frames, poses = load_scan_data(scan_dir, roi=roi,
                                       force_threshold=force_threshold)
    if pose_percentile > 0:
        frames, poses = filter_pose_outliers(frames, poses,
                                              percentile=pose_percentile)

    if poses_npz:
        d = np.load(poses_npz, allow_pickle=True)
        poses_np = d['poses']
        if len(poses_np) != len(frames):
            return dict(error=f'poses_npz length {len(poses_np)} != '
                              f'frames {len(frames)}')
        poses = [poses_np[i] for i in range(len(frames))]
        print(f'  Loaded refined poses from {poses_npz}')

    print(f'  Total frames: {len(frames)}')
    if len(frames) < 50:
        return dict(error=f'not enough frames ({len(frames)})')

    # Image-domain preprocessing
    xform_ops = []
    if tgc_eq:
        xform_ops.append(lambda im: apply_tgc_eq(
            im, intensity_min=intensity_min, gamma=tgc_gamma))
    if despeckle_method != 'none':
        xform_ops.append(lambda im: despeckle(
            im, despeckle_method, despeckle_ksize))
    if xform_ops:
        print(f'  Pre-processing frames ({len(xform_ops)} ops)...')
        frames = transform_frames(frames, compose_xform(*xform_ops))

    if pixel_weight_fn is None:
        weight_ops = []
        if shadow_w:
            weight_ops.append(shadow_weight)
        if edge_w:
            weight_ops.append(edge_weight)
        if weight_ops:
            pixel_weight_fn = compose_weight(*weight_ops)

    bounds = auto_bounds(poses, T_tool0_probe, T_probe_us,
                         frames[0].shape, px_x, px_y, margin=0.005)

    # Interleaved split (preserves spatial coverage in each half)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(frames))
    half = len(frames) // 2
    idx_a = sorted(perm[:half].tolist())
    idx_b = sorted(perm[half:].tolist())
    frames_a = [frames[i] for i in idx_a]; poses_a = [poses[i] for i in idx_a]
    frames_b = [frames[i] for i in idx_b]; poses_b = [poses[i] for i in idx_b]

    def _build(fr, ps):
        if algo == 'v2':
            return pbm_compound_v2(
                fr, ps, T_tool0_probe, T_probe_us, px_x, px_y,
                bounds, voxel_size,
                elevation_sigma_mm=elevation_sigma_mm,
                intensity_min=intensity_min,
                trilinear=trilinear,
                adaptive_elev=adaptive_elev,
                adaptive_alpha=adaptive_alpha,
                n_sigma_bins=n_sigma_bins,
                pixel_weight_fn=pixel_weight_fn,
                verbose=False)
        return pbm_compound(fr, ps, T_tool0_probe, T_probe_us,
                            px_x, px_y, bounds, voxel_size,
                            elevation_sigma_mm=elevation_sigma_mm,
                            intensity_min=intensity_min, verbose=False)

    print(f'\n=== Half A: {len(frames_a)} frames (algo={algo}) ===')
    t0 = time.time()
    vol_a, w_a = _build(frames_a, poses_a)
    print(f'  built in {time.time()-t0:.1f}s')

    print(f'\n=== Half B: {len(frames_b)} frames (algo={algo}) ===')
    t0 = time.time()
    vol_b, w_b = _build(frames_b, poses_b)
    print(f'  built in {time.time()-t0:.1f}s')

    cm = cross_metrics(vol_a, w_a, vol_b, w_b)
    cm['half_a_intrinsic'] = intrinsic_metrics(vol_a, w_a)
    cm['half_b_intrinsic'] = intrinsic_metrics(vol_b, w_b)
    return cm


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v):
    if isinstance(v, float):
        if abs(v) < 1e-3 or abs(v) > 1e5:
            return f'{v:.3e}'
        return f'{v:.4f}'
    if isinstance(v, list):
        return str(v)
    return str(v)


def print_metrics(name, m):
    print(f'\n─── {name} ───')
    skip_keys = {'half_a_intrinsic', 'half_b_intrinsic'}
    for k, v in m.items():
        if k in skip_keys:
            continue
        print(f'  {k:24s} {fmt(v)}')
    if 'half_a_intrinsic' in m:
        print('  [half A intrinsic]')
        for k, v in m['half_a_intrinsic'].items():
            print(f'    {k:22s} {fmt(v)}')
        print('  [half B intrinsic]')
        for k, v in m['half_b_intrinsic'].items():
            print(f'    {k:22s} {fmt(v)}')


def main():
    ap = argparse.ArgumentParser(description='3D US reconstruction QA metrics')
    ap.add_argument('--volume', help='Path to volume .npy')
    ap.add_argument('--weight', help='Path to weight .npy '
                                     '(default: replace .npy → _weight.npy)')
    ap.add_argument('--baseline', help='Optional second volume .npy for '
                                       'pairwise comparison')
    ap.add_argument('--baseline_weight', help='Weight for baseline')

    ap.add_argument('--split', action='store_true',
                    help='Self-consistency split-and-rebuild mode')
    ap.add_argument('--scan_dir')
    ap.add_argument('--calib_dir')
    ap.add_argument('--voxel_size', type=float, default=0.0007)
    ap.add_argument('--elevation_sigma_mm', type=float, default=2.5)
    ap.add_argument('--intensity_min', type=int, default=15)
    ap.add_argument('--force_threshold', type=float, default=1.0)
    ap.add_argument('--pose_percentile', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smooth_window', type=int, default=0,
                    help='SG smoothing window for split mode (odd, 0=off)')
    ap.add_argument('--smooth_order', type=int, default=3)
    ap.add_argument('--time_lag_ms', type=float, default=0.0,
                    help='Image-vs-pose lag in ms for split mode')
    ap.add_argument('--auto_lag', action='store_true',
                    help='Estimate image-pose lag automatically')
    ap.add_argument('--algo', choices=['v1', 'v2'], default='v1')
    ap.add_argument('--no_trilinear', action='store_true')
    ap.add_argument('--adaptive_elev', action='store_true')
    ap.add_argument('--adaptive_alpha', type=float, default=0.05)
    ap.add_argument('--n_sigma_bins', type=int, default=4)
    ap.add_argument('--tgc_eq', action='store_true')
    ap.add_argument('--tgc_gamma', type=float, default=0.5)
    ap.add_argument('--despeckle', choices=['none', 'median',
                                              'bilateral', 'gauss'],
                    default='none')
    ap.add_argument('--despeckle_ksize', type=int, default=3)
    ap.add_argument('--shadow_w', action='store_true')
    ap.add_argument('--edge_w', action='store_true')
    ap.add_argument('--poses_npz', help='Optional .npz from ibsr_refine.py '
                                         '(key "poses", same length as frames)')

    ap.add_argument('--json', help='Optionally dump metrics dict to JSON')
    args = ap.parse_args()

    report = {}

    if args.volume:
        weight_path = args.weight or args.volume.replace('.npy', '_weight.npy')
        vol = np.load(args.volume)
        w = np.load(weight_path) if os.path.exists(weight_path) \
            else (vol > 0).astype(np.float32)
        m = intrinsic_metrics(vol, w)
        print_metrics(f'Intrinsic — {os.path.basename(args.volume)}', m)
        report['volume'] = m

        if args.baseline:
            bw_path = args.baseline_weight or \
                      args.baseline.replace('.npy', '_weight.npy')
            vb = np.load(args.baseline); wb = np.load(bw_path) \
                if os.path.exists(bw_path) else (vb > 0).astype(np.float32)
            mb = intrinsic_metrics(vb, wb)
            print_metrics(f'Intrinsic — {os.path.basename(args.baseline)}', mb)
            report['baseline'] = mb
            cm = cross_metrics(vol, w, vb, wb)
            print_metrics('Cross (volume vs baseline)', cm)
            report['cross'] = cm

    if args.split:
        if not (args.scan_dir and args.calib_dir):
            ap.error('--split requires --scan_dir and --calib_dir')
        cm = split_consistency(args.scan_dir, args.calib_dir,
                               voxel_size=args.voxel_size,
                               elevation_sigma_mm=args.elevation_sigma_mm,
                               intensity_min=args.intensity_min,
                               force_threshold=args.force_threshold,
                               pose_percentile=args.pose_percentile,
                               seed=args.seed,
                               smooth_window=args.smooth_window,
                               smooth_order=args.smooth_order,
                               time_lag_ms=args.time_lag_ms,
                               auto_lag=args.auto_lag,
                               algo=args.algo,
                               trilinear=not args.no_trilinear,
                               adaptive_elev=args.adaptive_elev,
                               adaptive_alpha=args.adaptive_alpha,
                               n_sigma_bins=args.n_sigma_bins,
                               tgc_eq=args.tgc_eq,
                               tgc_gamma=args.tgc_gamma,
                               despeckle_method=args.despeckle,
                               despeckle_ksize=args.despeckle_ksize,
                               shadow_w=args.shadow_w,
                               edge_w=args.edge_w,
                               poses_npz=args.poses_npz)
        print_metrics('Split self-consistency', cm)
        report['split'] = cm

    if not args.volume and not args.split:
        ap.error('Provide --volume and/or --split mode')

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or '.',
                    exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'\nSaved report → {args.json}')


if __name__ == '__main__':
    main()
