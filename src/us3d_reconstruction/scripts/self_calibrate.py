#!/usr/bin/env python3
"""Image-based self-calibration of probe spatial transform + pixel scale.

Why
───
``T_probe_us`` in ``probe_calibration.yaml`` was derived analytically
(90° flip about X) and has zero residual from any actual data — any
small mounting offset, axis tilt, or pixel-size error gets baked into
the volume as systematic blur along the elevation direction.

We refine 8 parameters using only the data already on disk:

    delta_tx, delta_ty, delta_tz   (mm,  T_probe_us translation)
    delta_rx, delta_ry, delta_rz   (deg, T_probe_us rotation)
    log_sx, log_sy                  (log-scale, multiplies pixel_size)

Loss = -sharpness(volume) where sharpness = mean |∇I|^2 (Tenengrad)
restricted to the well-sampled bounding region. When calibration is
correct, frames register to the same edges → high gradients;
when wrong, edges smear along elevation → low gradients.

Optimizer: **Powell** by default (gradient-free, tolerates voxelized
Tenengrad). ``L-BFGS-B`` is supported with ``--fd_eps`` (~0.15) because
the default finite-difference step is far too small on this loss.

By default **pixel scale is frozen** (6-DOF only); pass
``--no_freeze_scale`` to fit scale inside ±``--bounds_scale_pct``.

Usage:
  python3 self_calibrate.py \
      --scan_dir   data_backup/scans/scan_20260407_150839 \
      --calib_dir  data_backup/calibration \
      --output     data_backup/calibration/probe_calibration_refined.yaml \
      --n_frames   80 --voxel_size 0.0015 --max_iter 80 \
      --smooth_window 11 --auto_lag --algo v1
"""

import argparse
import os
import sys
import time
import numpy as np
import yaml

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from voxel_reconstruct import load_calibration, auto_bounds
from pbm_reconstruct import pbm_compound, pbm_compound_v2
from pose_utils import load_scan_data_smooth, estimate_image_pose_lag
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, Bounds


# ─────────────────────────────────────────────────────────────────────────────
# Parameter ↔ transform plumbing
# ─────────────────────────────────────────────────────────────────────────────

def apply_delta(base_T_probe_us, base_px_x, base_px_y, params):
    """params layout (8): [tx_mm, ty_mm, tz_mm, rx_deg, ry_deg, rz_deg,
                            log_sx, log_sy]"""
    tx, ty, tz, rx, ry, rz, lsx, lsy = params
    dT = np.eye(4)
    dT[:3, :3] = R.from_rotvec(np.deg2rad([rx, ry, rz])).as_matrix()
    dT[:3, 3] = np.array([tx, ty, tz]) / 1000.0
    T_new = base_T_probe_us @ dT
    px_x = base_px_x * float(np.exp(lsx))
    px_y = base_px_y * float(np.exp(lsy))
    return T_new, px_x, px_y


# ─────────────────────────────────────────────────────────────────────────────
# Loss: tenengrad of well-sampled region
# ─────────────────────────────────────────────────────────────────────────────

def sharpness_loss(volume, weight, weight_thr_pct=50):
    """Mean |∇I|^2 inside well-sampled region. Negated for minimization."""
    if not np.any(weight > 0):
        return 0.0
    thr = np.percentile(weight[weight > 0], weight_thr_pct)
    mask = weight >= thr
    if mask.sum() < 50:
        return 0.0
    v = volume.astype(np.float32)
    gx = np.zeros_like(v); gy = np.zeros_like(v); gz = np.zeros_like(v)
    gx[1:-1, :, :] = 0.5 * (v[2:, :, :] - v[:-2, :, :])
    gy[:, 1:-1, :] = 0.5 * (v[:, 2:, :] - v[:, :-2, :])
    gz[:, :, 1:-1] = 0.5 * (v[:, :, 2:] - v[:, :, :-2])
    g2 = gx * gx + gy * gy + gz * gz
    return float(g2[mask].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Build a volume from a fixed frame subset, using current candidate params
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(frames, poses, T_tool0_probe, base_T_probe_us,
                   base_px_x, base_px_y,
                   voxel_size, elevation_sigma_mm, intensity_min,
                   pose_subset, roi_shape,
                   bounds_t_mm=5.0, bounds_r_deg=5.0,
                   bounds_log_scale=None,  # default = ±log(1.05) ≈ 0.0488
                   reg_t=0.001, reg_r=0.001, reg_s=10.0,
                   freeze_scale=False,
                   use_v2=False,
                   trilinear=True,
                   adaptive_elev=False,
                   adaptive_alpha=0.05,
                   n_sigma_bins=4,
                   verbose=False):
    """Return ``loss(x)`` closure with bounds + regularization.

    ``x`` is either 6-D (translation + rotation, scales fixed at 0) or
    full 8-D. Prefer ``method='L-BFGS-B'`` + ``Bounds`` so the solution
    cannot drift outside the box (Powell only sees a penalty, not true
    constraints).

    Two failure modes of pure Tenengrad maximization that this guards:

    1) **Pixel-scale collapse** — if scale → 0, all pixels project to a
       few voxels and Tenengrad inflates trivially. ``--freeze_scale``
       removes the variable (default **on**). Otherwise bound
       |log_scale| and penalize with ``reg_s``.

    2) **Translation/rotation drift** — bound ±5mm / ±5° and light
       ``reg_t`` / ``reg_r``.
    """
    if bounds_log_scale is None:
        bounds_log_scale = float(np.log(1.05))

    iter_state = dict(n=0, best=None, best_params=None,
                      t0=time.time(), history=[],
                      base_sharp=None)

    BOUND_T = bounds_t_mm
    BOUND_R = bounds_r_deg
    BOUND_S = bounds_log_scale
    n_x = 6 if freeze_scale else 8

    def to_params8(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if freeze_scale:
            return np.concatenate([x[:6], np.zeros(2, dtype=np.float64)])
        return x[:8].copy()

    def in_bounds8(p8):
        if abs(p8[0]) > BOUND_T or abs(p8[1]) > BOUND_T or abs(p8[2]) > BOUND_T:
            return False
        if abs(p8[3]) > BOUND_R or abs(p8[4]) > BOUND_R or abs(p8[5]) > BOUND_R:
            return False
        if abs(p8[6]) > BOUND_S or abs(p8[7]) > BOUND_S:
            return False
        return True

    def regularizer(p8):
        return (reg_t * (p8[0]**2 + p8[1]**2 + p8[2]**2) +
                reg_r * (p8[3]**2 + p8[4]**2 + p8[5]**2) +
                reg_s * (p8[6]**2 + p8[7]**2))

    def splat(fr, ps, Tpu, pxx, pyy, bnds):
        if use_v2:
            return pbm_compound_v2(
                fr, ps, T_tool0_probe, Tpu, pxx, pyy, bnds, voxel_size,
                elevation_sigma_mm=elevation_sigma_mm,
                intensity_min=intensity_min,
                trilinear=trilinear,
                adaptive_elev=adaptive_elev,
                adaptive_alpha=adaptive_alpha,
                n_sigma_bins=n_sigma_bins,
                verbose=False)
        return pbm_compound(
            fr, ps, T_tool0_probe, Tpu, pxx, pyy, bnds, voxel_size,
            elevation_sigma_mm=elevation_sigma_mm,
            intensity_min=intensity_min, verbose=False)

    def loss(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        params = to_params8(x)

        if not in_bounds8(params):
            # Soft-reject for Powell / Nelder-Mead (no true box constraints)
            penalty = 1e3
            for v, lim in zip(params,
                              [BOUND_T]*3 + [BOUND_R]*3 + [BOUND_S]*2):
                excess = max(abs(v) - lim, 0.0)
                penalty += 1e3 * excess * excess
            return penalty

        T_probe_us, px_x, px_y = apply_delta(base_T_probe_us,
                                             base_px_x, base_px_y, params)
        try:
            bounds = auto_bounds(pose_subset, T_tool0_probe, T_probe_us,
                                 roi_shape, px_x, px_y, margin=0.005)
            vol, w = splat(frames, poses, T_probe_us, px_x, px_y, bounds)
            sharp = sharpness_loss(vol, w)
        except Exception as e:
            if verbose:
                print(f'  eval error: {e}')
            return 1e6

        if iter_state['base_sharp'] is None:
            iter_state['base_sharp'] = sharp

        # Normalize sharpness so loss is O(1); regularizer is in same scale.
        norm_sharp = sharp / max(iter_state['base_sharp'], 1e-9)
        L = -norm_sharp + regularizer(params)

        iter_state['n'] += 1
        iter_state['history'].append((iter_state['n'], list(params), sharp))
        if iter_state['best'] is None or L < iter_state['best']:
            iter_state['best'] = L
            iter_state['best_params'] = list(params)
            iter_state['best_sharp'] = sharp
            tag = '★'
        else:
            tag = ' '
        if verbose:
            elapsed = time.time() - iter_state['t0']
            ps = (f't=({params[0]:+.2f},{params[1]:+.2f},{params[2]:+.2f})mm '
                  f'r=({params[3]:+.2f},{params[4]:+.2f},{params[5]:+.2f})° '
                  f's=({(np.exp(params[6])-1)*100:+.2f}%,'
                  f'{(np.exp(params[7])-1)*100:+.2f}%)')
            print(f'  {tag} {iter_state["n"]:3d} '
                  f'sharp={sharp:7.2f} L={L:+.4f} {ps} [{elapsed:.0f}s]')
        return L

    iter_state['n_x'] = n_x
    return loss, iter_state


def build_lbfgsb_bounds(freeze_scale, bounds_t_mm, bounds_r_deg,
                        bounds_log_scale):
    """Hard box for L-BFGS-B (translation mm, rotation deg, log-scale)."""
    lo = ([-bounds_t_mm] * 3 + [-bounds_r_deg] * 3 +
          ([-bounds_log_scale, -bounds_log_scale] if not freeze_scale else []))
    hi = ([bounds_t_mm] * 3 + [bounds_r_deg] * 3 +
          ([bounds_log_scale, bounds_log_scale] if not freeze_scale else []))
    return Bounds(lo, hi)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan_dir', required=True)
    ap.add_argument('--calib_dir', required=True)
    ap.add_argument('--output', required=True,
                    help='Refined YAML output path')
    ap.add_argument('--n_frames', type=int, default=80,
                    help='Subsample to N frames for speed')
    ap.add_argument('--voxel_size', type=float, default=0.0015)
    ap.add_argument('--elevation_sigma_mm', type=float, default=2.5)
    ap.add_argument('--intensity_min', type=int, default=15)
    ap.add_argument('--force_threshold', type=float, default=1.0)
    ap.add_argument('--max_iter', type=int, default=80)
    ap.add_argument('--smooth_window', type=int, default=0)
    ap.add_argument('--smooth_order', type=int, default=3)
    ap.add_argument('--time_lag_ms', type=float, default=0.0)
    ap.add_argument('--auto_lag', action='store_true')
    ap.add_argument('--init', nargs='+', type=float, default=None,
                    help='Initial 8-param vector (tx ty tz mm, rx ry rz deg, '
                         'log_sx log_sy)')
    ap.add_argument('--method', default='Powell',
                    choices=['L-BFGS-B', 'Powell', 'Nelder-Mead'],
                    help='Powell is robust on voxelized Tenengrad; L-BFGS-B '
                         'needs --fd_eps (default diff step is too small).')
    ap.add_argument('--fd_eps', type=float, default=0.15,
                    help='[L-BFGS-B] finite-diff step (mm/deg/log-scale); '
                         'try 0.1–0.25 if the optimizer exits at 0 iters.')
    ap.add_argument('--bounds_t_mm', type=float, default=5.0)
    ap.add_argument('--bounds_r_deg', type=float, default=5.0)
    ap.add_argument('--bounds_scale_pct', type=float, default=5.0,
                    help='Bound on |scale change|, percent (default 5)')
    ap.add_argument('--reg_t', type=float, default=0.001)
    ap.add_argument('--reg_r', type=float, default=0.001)
    ap.add_argument('--reg_s', type=float, default=10.0,
                    help='Heavy penalty on scale changes — Tenengrad is '
                         'gameable by shrinking pixels (default 10).')
    ap.add_argument('--no_freeze_scale', action='store_true',
                    help='Also optimize log pixel scale (8-D, bounded); '
                         'default is 6-DOF with scale frozen.')
    ap.add_argument('--algo', choices=['v1', 'v2'], default='v1',
                    help='PBM backend inside the objective (v2 = trilinear splat).')
    ap.add_argument('--trilinear', action='store_true', default=True,
                    help='[v2] trilinear splat (default on for --algo v2)')
    ap.add_argument('--no_trilinear', action='store_true')
    ap.add_argument('--adaptive_elev', action='store_true',
                    help='[v2] depth-adaptive elevation σ (slower)')
    ap.add_argument('--adaptive_alpha', type=float, default=0.05)
    ap.add_argument('--n_sigma_bins', type=int, default=4)
    args = ap.parse_args()
    args.freeze_scale = not args.no_freeze_scale

    print('Loading calibration...')
    T_tool0_probe, T_probe_us, px_x, px_y, roi = load_calibration(
        args.calib_dir)
    print(f'  Base T_probe_us:\n{T_probe_us}')
    print(f'  Base pixel size: {px_x*1000:.4f} × {px_y*1000:.4f} mm')

    print('\nLoading frames (smoothed)...')
    lag_s = args.time_lag_ms / 1000.0
    if args.auto_lag:
        lag_s = estimate_image_pose_lag(args.scan_dir, roi=roi, max_lag_s=0.2)
    win = args.smooth_window if args.smooth_window > 0 else 1
    frames_full, poses_full = load_scan_data_smooth(
        args.scan_dir, roi=roi, force_threshold=args.force_threshold,
        sg_window=win, sg_order=args.smooth_order, time_lag_s=lag_s)
    print(f'  Total: {len(frames_full)} frames')

    if len(frames_full) > args.n_frames:
        idx = np.linspace(0, len(frames_full) - 1, args.n_frames).astype(int)
        frames = [frames_full[i] for i in idx]
        poses = [poses_full[i] for i in idx]
    else:
        frames, poses = frames_full, poses_full
    print(f'  Subsampled: {len(frames)} frames for optimization')

    init8 = np.array(args.init) if args.init else np.zeros(8)
    assert len(init8) == 8
    n_x = 6 if args.freeze_scale else 8
    x0 = init8[:n_x].astype(np.float64)
    print(f'  Initial params (8): {init8.tolist()}  optimize_dim={n_x}')

    print(f'\nObjective: maximize Tenengrad sharpness, '
          f'voxel={args.voxel_size*1000:.2f}mm  PBM={args.algo}')
    print(f'freeze_scale={args.freeze_scale}  '
          f'Optimizer: {args.method}, max_iter={args.max_iter}\n')

    b_log = float(np.log(1 + args.bounds_scale_pct / 100.0))
    loss_fn, state = make_objective(
        frames, poses, T_tool0_probe, T_probe_us, px_x, px_y,
        args.voxel_size, args.elevation_sigma_mm, args.intensity_min,
        poses, frames[0].shape,
        bounds_t_mm=args.bounds_t_mm,
        bounds_r_deg=args.bounds_r_deg,
        bounds_log_scale=b_log,
        reg_t=args.reg_t, reg_r=args.reg_r, reg_s=args.reg_s,
        freeze_scale=args.freeze_scale,
        use_v2=(args.algo == 'v2'),
        trilinear=not args.no_trilinear,
        adaptive_elev=args.adaptive_elev,
        adaptive_alpha=args.adaptive_alpha,
        n_sigma_bins=args.n_sigma_bins,
        verbose=True)

    L0 = loss_fn(x0)
    sharp0 = state['base_sharp']
    print(f'\nInitial sharpness: {sharp0:.2f}, normalized loss L0={L0:.4f}\n')

    bounds_scipy = build_lbfgsb_bounds(
        args.freeze_scale, args.bounds_t_mm, args.bounds_r_deg, b_log)

    if args.method == 'L-BFGS-B':
        fd = np.full(n_x, args.fd_eps, dtype=np.float64)
        opts = dict(maxiter=args.max_iter, maxfun=max(500, args.max_iter * 25),
                    disp=1, ftol=1e-5, eps=fd)
        res = minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds_scipy,
                       options=opts)
    elif args.method == 'Powell':
        opts = dict(maxiter=args.max_iter,
                    maxfev=max(500, args.max_iter * 40),
                    xtol=0.05, ftol=0.005, disp=True)
        res = minimize(loss_fn, x0, method='Powell', options=opts)
    else:
        step = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0] +
                        ([] if args.freeze_scale else [0.02, 0.02]))
        x0_simplex = np.vstack([x0, x0 + np.eye(n_x) * step[:n_x]])
        opts = dict(maxiter=args.max_iter, fatol=0.5, xatol=0.01,
                    initial_simplex=x0_simplex, disp=True)
        res = minimize(loss_fn, x0, method='Nelder-Mead', options=opts)

    # Prefer tracker best (always in-bounds); never trust raw res.x for Powell.
    if state['best_params'] is not None:
        best = list(state['best_params'])
    elif n_x == 6:
        best = list(res.x) + [0.0, 0.0]
    else:
        best = list(res.x)
    sharp_final = state.get('best_sharp', sharp0)

    print(f'\n=== Optimization done (n_eval={state["n"]}) ===')
    print(f'Initial sharpness:  {sharp0:.3f}')
    print(f'Final sharpness:    {sharp_final:.3f}'
          f'  ({100.0*(sharp_final - sharp0) / max(sharp0, 1e-9):+.1f}%)')
    print(f'Best params: {[round(x, 4) for x in best]}')

    T_refined, px_x_new, px_y_new = apply_delta(T_probe_us, px_x, px_y,
                                                np.array(best))
    print(f'\nRefined T_probe_us:\n{T_refined}')
    print(f'Refined pixel size: {px_x_new*1000:.4f} × {px_y_new*1000:.4f} mm '
          f'(Δ {(px_x_new/px_x-1)*100:+.2f}%, '
          f'{(px_y_new/px_y-1)*100:+.2f}%)')

    # Write refined YAML (preserve other fields verbatim)
    with open(os.path.join(args.calib_dir, 'probe_calibration.yaml')) as f:
        cal = yaml.safe_load(f)
    cal['T_probe_us'] = T_refined.flatten().tolist()
    cal['pixel_size_x'] = float(px_x_new)
    cal['pixel_size_y'] = float(px_y_new)
    cal['_self_calibrated'] = dict(
        method=args.method, n_frames=len(frames),
        voxel_size=args.voxel_size,
        bounds_t_mm=args.bounds_t_mm, bounds_r_deg=args.bounds_r_deg,
        bounds_scale_pct=args.bounds_scale_pct,
        reg_t=args.reg_t, reg_r=args.reg_r, reg_s=args.reg_s,
        freeze_scale=bool(args.freeze_scale),
        sharpness_init=float(sharp0),
        sharpness_final=float(sharp_final),
        delta_params=dict(
            tx_mm=float(best[0]), ty_mm=float(best[1]), tz_mm=float(best[2]),
            rx_deg=float(best[3]), ry_deg=float(best[4]),
            rz_deg=float(best[5]),
            scale_x_pct=float((np.exp(best[6]) - 1) * 100),
            scale_y_pct=float((np.exp(best[7]) - 1) * 100),
        ),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.',
                exist_ok=True)
    with open(args.output, 'w') as f:
        yaml.safe_dump(cal, f, sort_keys=False)
    print(f'\nSaved refined calibration → {args.output}')


if __name__ == '__main__':
    main()
