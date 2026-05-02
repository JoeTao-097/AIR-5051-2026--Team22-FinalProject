#!/usr/bin/env python3
"""Lightweight IBSR: per-frame 1-D pose refinement along elevation axis.

For each B-mode frame we search a small translation of ``tool0`` along
the instantaneous ultrasound *elevation* direction (the 3rd column of
``T_base_us`` in ``pbm_reconstruct`` — out-of-plane normal of the image)
and pick the shift that maximizes NCC between the *forward-warped*
reference volume and the live frame.

This is a **pragmatic** substitute for full 6-DOF image-based slice
registration: it fixes the dominant residual error (slice thickness /
pose jitter along elevation) at ~O(N·K) cost where K is the number of
shift samples (~9–15).

Outputs ``poses_ibsr.npz`` with keys:
    poses       — (N, 4, 4) float64 refined ``T_base_tool0``
    shifts_mm   — (N,) float64 best shift in mm (+ = moved along +elev)
    ncc_init    — (N,) NCC at 0 shift
    ncc_best    — (N,) NCC at best shift

Usage:
  python3 ibsr_refine.py \\
      --scan_dir  data_backup/scans/scan_20260407_150839 \\
      --calib_dir data_backup/calibration \\
      --output    data_backup/scans/scan_20260407_150839/poses_ibsr.npz \\
      --ref_voxel 0.0015 --ref_stride 2 \\
      --search_mm 1.5 --n_shifts 9 \\
      --smooth_window 11 --auto_lag \\
      --tgc_eq --despeckle median
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from voxel_reconstruct import load_calibration, auto_bounds
from pose_utils import load_scan_data_smooth, estimate_image_pose_lag
from pbm_reconstruct import pbm_compound_v2
from image_preproc import (
    apply_tgc_eq, despeckle, compose_xform, transform_frames)


def sample_trilinear(vol, bounds, voxel, pts):
    """Trilinear sample ``vol`` at ``pts`` (N,3) in base frame. OOB → 0."""
    ijk = (pts - bounds[0]) / voxel  # continuous voxel coords (x, y, z)
    coords = np.vstack([ijk[:, 0], ijk[:, 1], ijk[:, 2]])
    return map_coordinates(vol, coords, order=1, mode='constant',
                           cval=0.0, prefilter=False).astype(np.float32)


def _ncc(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if len(a) < 32:
        return -1.0
    a0 = a - a.mean()
    b0 = b - b.mean()
    den = np.sqrt((a0 ** 2).sum() * (b0 ** 2).sum())
    if den < 1e-9:
        return -1.0
    return float((a0 * b0).sum() / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan_dir', required=True)
    ap.add_argument('--calib_dir', required=True)
    ap.add_argument('--output', required=True, help='.npz path')
    ap.add_argument('--ref_voxel', type=float, default=0.0015)
    ap.add_argument('--ref_stride', type=int, default=2,
                    help='Use every Nth frame to build reference volume')
    ap.add_argument('--search_mm', type=float, default=1.5)
    ap.add_argument('--n_shifts', type=int, default=9)
    ap.add_argument('--intensity_min', type=int, default=15)
    ap.add_argument('--force_threshold', type=float, default=1.0)
    ap.add_argument('--smooth_window', type=int, default=11)
    ap.add_argument('--smooth_order', type=int, default=3)
    ap.add_argument('--time_lag_ms', type=float, default=0.0)
    ap.add_argument('--auto_lag', action='store_true')
    ap.add_argument('--tgc_eq', action='store_true')
    ap.add_argument('--tgc_gamma', type=float, default=0.5)
    ap.add_argument('--despeckle', choices=['none', 'median'],
                    default='none')
    ap.add_argument('--max_frames', type=int, default=0,
                    help='0 = all frames')
    args = ap.parse_args()

    T_tool0_probe, T_probe_us, px_x, px_y, roi = load_calibration(
        args.calib_dir)
    T_tool0_us = T_tool0_probe @ T_probe_us

    lag_s = args.time_lag_ms / 1000.0
    if args.auto_lag:
        lag_s = estimate_image_pose_lag(args.scan_dir, roi=roi,
                                        max_lag_s=0.2, verbose=True)
    win = args.smooth_window if args.smooth_window > 0 else 1
    frames, poses = load_scan_data_smooth(
        args.scan_dir, roi=roi, force_threshold=args.force_threshold,
        sg_window=win, sg_order=args.smooth_order, time_lag_s=lag_s)
    print(f'Loaded {len(frames)} frames')

    if args.tgc_eq or args.despeckle != 'none':
        ops = []
        if args.tgc_eq:
            ops.append(lambda im: apply_tgc_eq(
                im, intensity_min=args.intensity_min, gamma=args.tgc_gamma))
        if args.despeckle != 'none':
            ops.append(lambda im: despeckle(im, args.despeckle, 3))
        frames = transform_frames(frames, compose_xform(*ops))

    if args.max_frames > 0 and len(frames) > args.max_frames:
        frames = frames[:args.max_frames]
        poses = poses[:args.max_frames]
        print(f'Truncated to {len(frames)} frames')

    bounds = auto_bounds(poses, T_tool0_probe, T_probe_us,
                         frames[0].shape, px_x, px_y, margin=0.005)

    # Reference volume (subsampled frames for speed)
    idx = np.arange(0, len(frames), args.ref_stride)
    fr_r = [frames[i] for i in idx]
    ps_r = [poses[i] for i in idx]
    print(f'Building reference volume ({len(fr_r)} frames, '
          f'voxel={args.ref_voxel*1000:.2f} mm)...')
    t0 = time.time()
    vol, w = pbm_compound_v2(
        fr_r, ps_r, T_tool0_probe, T_probe_us, px_x, px_y,
        bounds, args.ref_voxel, elevation_sigma_mm=2.5,
        intensity_min=args.intensity_min, trilinear=True, verbose=False)
    print(f'  ref volume done in {time.time()-t0:.1f}s, '
          f'shape={vol.shape}')

    shifts_grid = np.linspace(-args.search_mm, args.search_mm,
                              args.n_shifts)

    h, w_img = frames[0].shape
    uu, vv = np.meshgrid(np.arange(w_img, dtype=np.float64) * px_x,
                         np.arange(h, dtype=np.float64) * px_y)
    ones = np.ones_like(uu)
    zeros = np.zeros_like(uu)
    p_us_flat = np.stack([uu.ravel(), vv.ravel(), zeros.ravel(),
                          ones.ravel()], axis=0)  # (4, H*W)

    shifts_out = []
    ncc_init = []
    ncc_best = []
    poses_new = []

    print(f'Per-frame 1-D search ({args.n_shifts} shifts)...')
    t0 = time.time()
    for fi, (img, T_base_tool0) in enumerate(zip(frames, poses)):
        mask = img.ravel() >= args.intensity_min
        if mask.sum() < 100:
            poses_new.append(T_base_tool0.copy())
            shifts_out.append(0.0)
            ncc_init.append(-1.0)
            ncc_best.append(-1.0)
            continue

        T_base_us = T_base_tool0 @ T_tool0_us
        elev = T_base_us[:3, 2].astype(np.float64)
        elev /= max(np.linalg.norm(elev), 1e-9)

        best_s = 0.0
        best_n = -2.0
        n0 = -2.0
        p_sel = p_us_flat[:, mask]

        for s_mm in shifts_grid:
            T_try = T_base_tool0.copy()
            T_try[:3, 3] = T_try[:3, 3] + (s_mm / 1000.0) * elev
            T_try_us = T_try @ T_tool0_us
            pts = (T_try_us @ p_sel)[:3].T
            samp = sample_trilinear(vol, bounds, args.ref_voxel, pts)
            gt = img.ravel()[mask].astype(np.float32)
            nc = _ncc(samp, gt)
            if abs(s_mm) < 1e-9:
                n0 = nc
            if nc > best_n:
                best_n = nc
                best_s = float(s_mm)

        T_fix = T_base_tool0.copy()
        T_fix[:3, 3] = T_fix[:3, 3] + (best_s / 1000.0) * elev
        poses_new.append(T_fix)
        shifts_out.append(best_s)
        ncc_init.append(float(n0))
        ncc_best.append(float(best_n))

        if (fi + 1) % 50 == 0:
            dt = time.time() - t0
            print(f'  {fi+1}/{len(frames)}  ({(fi+1)/dt:.2f} fps)')

    shifts_out = np.array(shifts_out, dtype=np.float64)
    ncc_init = np.array(ncc_init, dtype=np.float64)
    ncc_best = np.array(ncc_best, dtype=np.float64)
    poses_arr = np.stack(poses_new, axis=0)

    print(f'\nShift stats [mm]: mean={shifts_out.mean():.3f}, '
          f'std={shifts_out.std():.3f}, max|.|={np.abs(shifts_out).max():.3f}')
    print(f'NCC: mean init={ncc_init[ncc_init > -0.5].mean():.4f} → '
          f'best={ncc_best[ncc_best > -0.5].mean():.4f}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.',
                exist_ok=True)
    np.savez_compressed(
        args.output,
        poses=poses_arr,
        shifts_mm=shifts_out,
        ncc_init=ncc_init,
        ncc_best=ncc_best,
        ref_voxel=args.ref_voxel,
        ref_stride=args.ref_stride,
        search_mm=args.search_mm,
        n_shifts=args.n_shifts,
    )
    print(f'\nSaved → {args.output}')


if __name__ == '__main__':
    main()
