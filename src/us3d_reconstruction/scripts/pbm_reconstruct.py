#!/usr/bin/env python3
"""PLUS-style 3D ultrasound reconstruction (PBM + hole filling).

Pipeline:
  Pass 1 — Pixel-Based Method (PBM) compounding
      For each pixel, splat into the voxel grid with an anisotropic
      Gaussian kernel along the probe elevation (out-of-plane) axis
      to model the physical slice thickness (~5 mm for a 1D linear
      array). Lateral/depth use nearest-neighbor since image pixels
      (~0.11 mm) are smaller than typical voxel sizes (0.3-1 mm).

  Pass 2 — Normalized Gaussian hole filling
      Empty voxels inside the swept slab are filled by normalized
      convolution of the partial volume (PLUS toolkit's
      FILL_HOLES_GAUSSIAN strategy). Iterated up to ``max_iter``
      times to bridge sparse gaps.

References:
  - Solberg O.V. et al. "Freehand 3D Ultrasound Reconstruction
    Algorithms - A Review", Ultrasound Med Biol, 2007.
  - PLUS Toolkit: vtkPasteSliceIntoVolume + vtkFillHolesInVolume.

Usage:
    python3 pbm_reconstruct.py \
        --scan_dir data/scans/scan_20260407_150839 \
        --calib_dir data/calibration \
        --output   data/reconstructions/pbm_v1.npy \
        --voxel_size 0.0005 \
        --elevation_sigma_mm 2.5 \
        --intensity_min 15 \
        --hole_fill_sigma_mm 1.5 --hole_fill_iter 2
"""

import argparse
import os
import time
import numpy as np

from voxel_reconstruct import load_calibration, load_scan_data, auto_bounds
from pose_utils import load_scan_data_smooth, estimate_image_pose_lag
from image_preproc import (apply_tgc_eq, despeckle, shadow_weight,
                            edge_weight, compose_xform, compose_weight,
                            transform_frames)


def filter_pose_outliers(frames, poses, percentile=10.0, verbose=True):
    """Keep only frames whose tool0 position lies in the central
    percentile-to-(100-percentile) window of all axes.

    Drops "transient" frames (initial contact, retraction, accidental
    moves) whose pose is far from the main scan cluster.
    """
    if percentile <= 0:
        return frames, poses
    pos = np.array([T[:3, 3] for T in poses])
    lo = np.percentile(pos, percentile, axis=0)
    hi = np.percentile(pos, 100 - percentile, axis=0)

    keep = np.all((pos >= lo) & (pos <= hi), axis=1)
    n_drop = int((~keep).sum())
    if verbose:
        span_lo = lo * 1000
        span_hi = hi * 1000
        print(f"  Pose filter [{percentile:.0f}-{100-percentile:.0f} pct]: "
              f"X[{span_lo[0]:.1f},{span_hi[0]:.1f}]  "
              f"Y[{span_lo[1]:.1f},{span_hi[1]:.1f}]  "
              f"Z[{span_lo[2]:.1f},{span_hi[2]:.1f}] mm")
        print(f"  Dropped {n_drop} pose-outlier frames, "
              f"kept {int(keep.sum())} / {len(poses)}")
    frames = [f for f, k in zip(frames, keep) if k]
    poses = [p for p, k in zip(poses, keep) if k]
    return frames, poses


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1: PBM compounding
# ─────────────────────────────────────────────────────────────────────────────

def pbm_compound(frames, poses, T_tool0_probe, T_probe_us,
                 pixel_size_x, pixel_size_y,
                 volume_bounds, voxel_size,
                 elevation_sigma_mm=2.5,
                 elevation_truncate=2.5,
                 intensity_min=15,
                 verbose=True):
    """Compound ultrasound frames into a voxel volume with anisotropic
    Gaussian splatting along the elevation direction.

    Args:
        elevation_sigma_mm: Gaussian sigma for the out-of-plane kernel
            (mm). Set close to half the probe's elevation slice
            thickness (5-10 mm for a 1D linear array → sigma 2-3 mm).
        elevation_truncate: kernel half-width in units of sigma.
        intensity_min: ignore pixels with raw intensity below this
            threshold (drops black background and "no echo" regions).
    """
    grid_shape = np.ceil((volume_bounds[1] - volume_bounds[0]) / voxel_size).astype(int)
    Nx, Ny, Nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
    N_total = Nx * Ny * Nz
    yz_stride = Ny * Nz

    if verbose:
        mem_mb = (N_total * 8 * 2) / (1024 ** 2)
        print(f"  Grid: {Nx}×{Ny}×{Nz} = {N_total/1e6:.1f}M voxels  (~{mem_mb:.0f} MB)")

    volume_flat = np.zeros(N_total, dtype=np.float64)
    weight_flat = np.zeros(N_total, dtype=np.float64)

    # Elevation Gaussian kernel (in voxel units)
    sigma_vox = elevation_sigma_mm / 1000.0 / voxel_size
    half = int(np.ceil(elevation_truncate * sigma_vox))
    e_offsets = np.arange(-half, half + 1, dtype=np.float64)
    e_kernel = np.exp(-0.5 * (e_offsets / sigma_vox) ** 2)
    if verbose:
        print(f"  Elevation kernel: sigma={elevation_sigma_mm:.2f}mm "
              f"({sigma_vox:.2f} voxels), half-width={half} voxels, "
              f"{len(e_offsets)} layers")

    T_tool0_us = T_tool0_probe @ T_probe_us

    t0 = time.time()
    n_pixels_total = 0

    for fi, (img, T_base_tool0) in enumerate(zip(frames, poses)):
        T_base_us = T_base_tool0 @ T_tool0_us
        elev_dir = T_base_us[:3, 2]   # unit vector along out-of-plane (in base frame)

        # Mask dark pixels (background / no signal)
        mask = img >= intensity_min
        if not mask.any():
            continue

        v_idx, u_idx = np.where(mask)
        intensities = img[v_idx, u_idx].astype(np.float64)
        n_pixels_total += len(intensities)

        # Pixel positions in US plane → base frame
        u_phys = u_idx.astype(np.float64) * pixel_size_x
        v_phys = v_idx.astype(np.float64) * pixel_size_y
        N = len(u_phys)
        p_us = np.stack(
            [u_phys, v_phys, np.zeros(N), np.ones(N)], axis=0)
        p_base_center = (T_base_us @ p_us)[:3].T          # (N, 3) meters

        # Float voxel coords
        ijk_center = (p_base_center - volume_bounds[0]) / voxel_size

        for k_off, w_e in zip(e_offsets, e_kernel):
            # Shift along elevation in voxel coords (elev_dir is a unit
            # vector in physical space → numerically a unit vector in
            # voxel coords too, since both axes share the same metric).
            ijk = ijk_center + k_off * elev_dir
            ijk_int = np.round(ijk).astype(np.int32)

            valid = ((ijk_int[:, 0] >= 0) & (ijk_int[:, 0] < Nx) &
                     (ijk_int[:, 1] >= 0) & (ijk_int[:, 1] < Ny) &
                     (ijk_int[:, 2] >= 0) & (ijk_int[:, 2] < Nz))
            if not valid.any():
                continue

            iv = ijk_int[valid]
            it = intensities[valid]

            flat = iv[:, 0] * yz_stride + iv[:, 1] * Nz + iv[:, 2]
            volume_flat += np.bincount(flat, weights=it * w_e,
                                        minlength=N_total)
            weight_flat += np.bincount(flat,
                                        weights=np.full(len(flat), w_e),
                                        minlength=N_total)

        if verbose and (fi + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            eta = (len(frames) - (fi + 1)) / rate
            print(f"  Splat {fi+1:4d}/{len(frames)}  "
                  f"({rate:.1f} fps, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    if verbose:
        print(f"  PBM compound done in {elapsed:.1f}s "
              f"({n_pixels_total/1e6:.1f}M valid pixels)")

    volume = volume_flat.reshape(grid_shape)
    weight = weight_flat.reshape(grid_shape)

    mean_volume = np.zeros_like(volume, dtype=np.float32)
    mask = weight > 0
    mean_volume[mask] = (volume[mask] / weight[mask]).astype(np.float32)

    if verbose:
        fill_ratio = 100.0 * np.count_nonzero(mask) / mask.size
        print(f"  Pass 1 fill ratio: {fill_ratio:.1f}% "
              f"({np.count_nonzero(mask):,} / {mask.size:,} voxels)")

    return mean_volume, weight.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1 v2: Trilinear splat + optional depth-adaptive elevation σ
# ─────────────────────────────────────────────────────────────────────────────

def _build_elev_kernel(sigma_mm, voxel_size, truncate=2.5):
    sigma_vox = sigma_mm / 1000.0 / voxel_size
    half = max(1, int(np.ceil(truncate * sigma_vox)))
    offs = np.arange(-half, half + 1, dtype=np.float64)
    ker = np.exp(-0.5 * (offs / sigma_vox) ** 2)
    return offs, ker, half


def pbm_compound_v2(frames, poses, T_tool0_probe, T_probe_us,
                    pixel_size_x, pixel_size_y,
                    volume_bounds, voxel_size,
                    elevation_sigma_mm=2.5,
                    elevation_truncate=2.5,
                    intensity_min=15,
                    trilinear=True,
                    adaptive_elev=False,
                    adaptive_alpha=0.05,
                    n_sigma_bins=4,
                    pixel_weight_fn=None,
                    verbose=True):
    """Improved PBM with trilinear splatting and optional depth-adaptive
    elevation kernel.

    Args:
        trilinear: if True, splat each pixel to its 8 voxel corners with
            barycentric weights. Removes the "stair-step" aliasing of
            nearest-neighbor splatting at the cost of ~6× more
            ``bincount`` calls. Single biggest quality win after pose
            smoothing.
        adaptive_elev: if True, the elevation σ varies linearly with
            depth (image row): ``sigma(v) = elevation_sigma_mm +
            adaptive_alpha * v_phys_mm``. Models the fact that a 1D
            linear array's elevation slice thickness grows with depth.
        n_sigma_bins: number of σ quantization levels for adaptive mode
            (more = closer to per-row σ, but slower).
        pixel_weight_fn: optional callable ``f(img) → (H, W) float`` that
            returns a per-pixel confidence weight (e.g. gradient
            magnitude, anti-shadow, anti-glare). Multiplied into the
            integration weight.
    """
    grid_shape = np.ceil(
        (volume_bounds[1] - volume_bounds[0]) / voxel_size).astype(int)
    Nx, Ny, Nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
    N_total = Nx * Ny * Nz
    yz_stride = Ny * Nz

    if verbose:
        mem_mb = (N_total * 8 * 2) / (1024 ** 2)
        print(f'  Grid: {Nx}×{Ny}×{Nz} = {N_total/1e6:.1f}M voxels '
              f'(~{mem_mb:.0f} MB)  trilinear={trilinear}  '
              f'adaptive_elev={adaptive_elev}')

    volume_flat = np.zeros(N_total, dtype=np.float64)
    weight_flat = np.zeros(N_total, dtype=np.float64)

    # Pre-build elevation kernels (one per σ bin)
    h, w = frames[0].shape
    if adaptive_elev:
        v_idx_all = np.arange(h)
        v_phys_mm = v_idx_all * pixel_size_y * 1000.0
        sigmas = elevation_sigma_mm + adaptive_alpha * v_phys_mm
        bin_edges = np.linspace(sigmas.min(), sigmas.max() + 1e-9,
                                n_sigma_bins + 1)
        bin_idx_per_row = np.clip(
            np.searchsorted(bin_edges, sigmas, side='right') - 1,
            0, n_sigma_bins - 1)
        bin_sigmas = []
        for b in range(n_sigma_bins):
            mask_b = bin_idx_per_row == b
            if mask_b.any():
                bin_sigmas.append(float(sigmas[mask_b].mean()))
            else:
                bin_sigmas.append(elevation_sigma_mm)
        kernels = [_build_elev_kernel(s, voxel_size, elevation_truncate)
                   for s in bin_sigmas]
        if verbose:
            print('  Adaptive elevation σ bins (mm): ' +
                  ', '.join(f'{s:.2f}' for s in bin_sigmas))
    else:
        offs, ker, half = _build_elev_kernel(
            elevation_sigma_mm, voxel_size, elevation_truncate)
        if verbose:
            print(f'  Elevation σ={elevation_sigma_mm:.2f}mm, '
                  f'half-width={half} voxels, {len(offs)} layers')

    T_tool0_us = T_tool0_probe @ T_probe_us
    t0 = time.time()
    n_pixels_total = 0

    for fi, (img, T_base_tool0) in enumerate(zip(frames, poses)):
        T_base_us = T_base_tool0 @ T_tool0_us
        elev_dir = T_base_us[:3, 2]

        mask = img >= intensity_min
        if not mask.any():
            continue

        v_idx, u_idx = np.where(mask)
        intensities = img[v_idx, u_idx].astype(np.float64)
        n_pixels_total += len(intensities)

        if pixel_weight_fn is not None:
            pw = pixel_weight_fn(img)
            extra_w = pw[v_idx, u_idx].astype(np.float64)
        else:
            extra_w = None

        u_phys = u_idx.astype(np.float64) * pixel_size_x
        v_phys = v_idx.astype(np.float64) * pixel_size_y
        N = len(u_phys)
        p_us = np.stack([u_phys, v_phys, np.zeros(N), np.ones(N)], axis=0)
        p_base_center = (T_base_us @ p_us)[:3].T
        ijk_center = (p_base_center - volume_bounds[0]) / voxel_size

        # Group by σ bin if adaptive (else single global kernel)
        if adaptive_elev:
            row_bin = bin_idx_per_row[v_idx]
        else:
            row_bin = np.zeros(N, dtype=np.int32)

        n_bins = len(kernels) if adaptive_elev else 1
        for b in range(n_bins):
            sel_b = row_bin == b
            if not sel_b.any():
                continue
            if adaptive_elev:
                offs_b, ker_b, _ = kernels[b]
            else:
                offs_b, ker_b = offs, ker
            ijk_b = ijk_center[sel_b]
            it_b = intensities[sel_b]
            ew_b = extra_w[sel_b] if extra_w is not None else None

            # Pre-build the 8 corner offsets (constant)
            CORNERS = np.array(
                [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                 (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)], dtype=np.int32)

            for k_off, w_e in zip(offs_b, ker_b):
                ijk_l = ijk_b + k_off * elev_dir
                M = ijk_l.shape[0]

                if trilinear:
                    floors = np.floor(ijk_l).astype(np.int32)  # (M, 3)
                    fr = ijk_l - floors                         # (M, 3)
                    om = 1.0 - fr
                    # Per-corner barycentric weights → (8, M)
                    all_w = np.empty((8, M), dtype=np.float64)
                    for c, (dx, dy, dz) in enumerate(CORNERS):
                        wx = fr[:, 0] if dx else om[:, 0]
                        wy = fr[:, 1] if dy else om[:, 1]
                        wz = fr[:, 2] if dz else om[:, 2]
                        all_w[c] = wx * wy * wz
                    all_w *= w_e
                    # Indices for each corner → (8, M, 3) → flat (8*M, 3)
                    all_idx = floors[None] + CORNERS[:, None, :]
                    flat_idx = all_idx.reshape(-1, 3)
                    flat_w = all_w.reshape(-1)
                    flat_int = np.tile(it_b, 8)
                    if ew_b is not None:
                        flat_w *= np.tile(ew_b, 8)

                    valid = (
                        (flat_idx[:, 0] >= 0) & (flat_idx[:, 0] < Nx) &
                        (flat_idx[:, 1] >= 0) & (flat_idx[:, 1] < Ny) &
                        (flat_idx[:, 2] >= 0) & (flat_idx[:, 2] < Nz) &
                        (flat_w > 0))
                    if not valid.any():
                        continue
                    iv = flat_idx[valid]
                    iw = flat_w[valid]
                    it_v = flat_int[valid]
                    flat = (iv[:, 0] * yz_stride +
                            iv[:, 1] * Nz + iv[:, 2])
                    # Two bincounts per (frame, σ-bin, elev-layer)
                    # (down from 8 in the naive loop).
                    volume_flat += np.bincount(
                        flat, weights=it_v * iw, minlength=N_total)
                    weight_flat += np.bincount(
                        flat, weights=iw, minlength=N_total)
                else:
                    ijk_int = np.round(ijk_l).astype(np.int32)
                    valid = ((ijk_int[:, 0] >= 0) & (ijk_int[:, 0] < Nx) &
                             (ijk_int[:, 1] >= 0) & (ijk_int[:, 1] < Ny) &
                             (ijk_int[:, 2] >= 0) & (ijk_int[:, 2] < Nz))
                    if not valid.any():
                        continue
                    iv = ijk_int[valid]
                    it_v = it_b[valid]
                    iw = np.full(len(iv), w_e, dtype=np.float64)
                    if ew_b is not None:
                        iw = iw * ew_b[valid]
                    flat = (iv[:, 0] * yz_stride + iv[:, 1] * Nz + iv[:, 2])
                    volume_flat += np.bincount(flat, weights=it_v * iw,
                                                minlength=N_total)
                    weight_flat += np.bincount(flat, weights=iw,
                                                minlength=N_total)

        if verbose and (fi + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            eta = (len(frames) - (fi + 1)) / rate
            print(f'  Splat {fi+1:4d}/{len(frames)}  '
                  f'({rate:.1f} fps, ETA {eta:.0f}s)')

    elapsed = time.time() - t0
    if verbose:
        print(f'  PBM v2 compound done in {elapsed:.1f}s '
              f'({n_pixels_total/1e6:.1f}M valid pixels)')

    volume = volume_flat.reshape(grid_shape)
    weight = weight_flat.reshape(grid_shape)
    mean_volume = np.zeros_like(volume, dtype=np.float32)
    mask_v = weight > 0
    mean_volume[mask_v] = (volume[mask_v] / weight[mask_v]).astype(np.float32)
    if verbose:
        fill = 100.0 * np.count_nonzero(mask_v) / mask_v.size
        print(f'  Pass 1 fill ratio: {fill:.1f}%')
    return mean_volume, weight.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Pass 2: Normalized Gaussian hole filling
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_hole_fill(volume, weight, voxel_size,
                       sigma_mm=1.5,
                       max_iter=2,
                       min_w_threshold=0.01,
                       restrict_to_bbox=True,
                       verbose=True):
    """Fill empty voxels via normalized Gaussian convolution.

    For each iteration:
        sm_v = Gauss(v * mask)
        sm_w = Gauss(mask)
        v[empty & sm_w>thr] = sm_v / sm_w
        mask[filled] = True (with reduced weight)

    Args:
        sigma_mm: spatial sigma of the Gaussian filler.
        max_iter: number of iterations (each iteration roughly bridges
            another sigma's worth of distance).
        min_w_threshold: only fill voxels whose smoothed weight exceeds
            this fraction (avoids extrapolating into truly distant
            empty regions).
        restrict_to_bbox: only operate inside the bounding box of the
            non-zero region (much faster for sparse volumes).
    """
    from scipy.ndimage import gaussian_filter

    sigma_vox = sigma_mm / 1000.0 / voxel_size

    if restrict_to_bbox:
        nz = np.argwhere(weight > 0)
        if len(nz) == 0:
            print("  WARNING: no non-zero voxels to fill from")
            return volume, weight
        pad = int(np.ceil(3 * sigma_vox))
        lo = np.maximum(nz.min(0) - pad, 0)
        hi = np.minimum(nz.max(0) + pad + 1, np.array(volume.shape))
        sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        if verbose:
            sub_shape = tuple(int(h - l) for l, h in zip(lo, hi))
            print(f"  Hole-fill bbox: {sub_shape} "
                  f"({np.prod(sub_shape)/1e6:.1f}M voxels)")
        vol_sub = volume[sl].copy()
        w_sub = weight[sl].astype(np.float64)
    else:
        vol_sub = volume.copy()
        w_sub = weight.astype(np.float64)

    if verbose:
        print(f"  Gaussian sigma={sigma_mm:.2f}mm ({sigma_vox:.2f} voxels), "
              f"max_iter={max_iter}")

    for it in range(max_iter):
        empty = w_sub <= 0
        n_empty = int(empty.sum())
        if n_empty == 0:
            if verbose:
                print(f"  Iter {it+1}: no empty voxels left")
            break

        mask = (~empty).astype(np.float64)
        masked_vol = vol_sub.astype(np.float64) * mask

        sm_vol = gaussian_filter(masked_vol, sigma=sigma_vox, truncate=2.5)
        sm_w = gaussian_filter(mask, sigma=sigma_vox, truncate=2.5)

        fillable = empty & (sm_w > min_w_threshold)
        n_fill = int(fillable.sum())

        if n_fill == 0:
            if verbose:
                print(f"  Iter {it+1}: no fillable empty voxels "
                      f"(sm_w threshold too high?)")
            break

        vol_sub = vol_sub.astype(np.float32)
        vol_sub[fillable] = (sm_vol[fillable] / sm_w[fillable]).astype(np.float32)
        # Mark as filled but with the smoothed weight (so subsequent
        # iterations weight them less than original measurements).
        w_sub[fillable] = sm_w[fillable].astype(np.float32)

        if verbose:
            print(f"  Iter {it+1}: filled {n_fill:,} of {n_empty:,} empty voxels")

    if restrict_to_bbox:
        out_vol = volume.copy()
        out_w = weight.copy()
        out_vol[sl] = vol_sub.astype(np.float32)
        out_w[sl] = w_sub.astype(np.float32)
        return out_vol, out_w
    else:
        return vol_sub.astype(np.float32), w_sub.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='PLUS-style PBM 3D ultrasound reconstruction')
    parser.add_argument('--scan_dir', required=True)
    parser.add_argument('--calib_dir', required=True)
    parser.add_argument('--output', required=True, help='Output .npy file')
    parser.add_argument('--voxel_size', type=float, default=0.0005,
                        help='Voxel edge length in meters (default 0.5mm)')
    parser.add_argument('--elevation_sigma_mm', type=float, default=2.5,
                        help='Out-of-plane Gaussian sigma in mm '
                             '(half the slice thickness; default 2.5)')
    parser.add_argument('--intensity_min', type=int, default=15,
                        help='Skip pixels darker than this raw value '
                             '(drops background; default 15)')
    parser.add_argument('--hole_fill_sigma_mm', type=float, default=1.5,
                        help='Hole-fill Gaussian sigma in mm '
                             '(0 to disable; default 1.5)')
    parser.add_argument('--hole_fill_iter', type=int, default=2,
                        help='Number of hole-fill iterations (default 2)')
    parser.add_argument('--force_threshold', type=float, default=1.0,
                        help='Min |Fz| in N to consider a frame '
                             'as scanning (default 1.0)')
    parser.add_argument('--bounds_margin', type=float, default=0.005,
                        help='Auto-bounds margin in m (default 0.005)')
    parser.add_argument('--pose_percentile', type=float, default=0.0,
                        help='Drop frames whose pose is outside the central '
                             '[p, 100-p] percentile window on all 3 axes. '
                             'Set 5-10 to remove transient outlier frames '
                             '(default 0 = disabled)')
    parser.add_argument('--smooth_window', type=int, default=0,
                        help='Savitzky-Golay window for pose smoothing '
                             '(odd, e.g. 11). 0 disables smoothing and '
                             'falls back to the raw per-row pose (default 0).')
    parser.add_argument('--smooth_order', type=int, default=3,
                        help='Polynomial order for SG smoothing (default 3).')
    parser.add_argument('--time_lag_ms', type=float, default=0.0,
                        help='Image-vs-pose timestamp lag in ms; smoothed '
                             'trajectory is sampled at ts_image - lag. '
                             'Use --auto_lag to estimate (default 0).')
    parser.add_argument('--auto_lag', action='store_true',
                        help='Estimate the image-pose lag from the '
                             'wrench/intensity cross-correlation and use it.')
    parser.add_argument('--algo', choices=['v1', 'v2'], default='v1',
                        help='v1 = NN splat (legacy), v2 = trilinear + '
                             'optional adaptive elevation σ (default v1)')
    parser.add_argument('--trilinear', action='store_true',
                        help='[v2] Use trilinear splat (8-corner)')
    parser.add_argument('--adaptive_elev', action='store_true',
                        help='[v2] Depth-adaptive elevation σ '
                             '(σ grows with image-row depth)')
    parser.add_argument('--adaptive_alpha', type=float, default=0.05,
                        help='[v2] Slope of σ-vs-depth (default 0.05 = '
                             '+0.05mm σ per mm of depth)')
    parser.add_argument('--n_sigma_bins', type=int, default=4,
                        help='[v2] Quantization levels for adaptive σ')
    parser.add_argument('--tgc_eq', action='store_true',
                        help='Apply per-row depth equalization to each frame')
    parser.add_argument('--tgc_gamma', type=float, default=0.5)
    parser.add_argument('--despeckle', choices=['none', 'median',
                                                  'bilateral', 'gauss'],
                        default='none')
    parser.add_argument('--despeckle_ksize', type=int, default=3)
    parser.add_argument('--shadow_w', action='store_true',
                        help='Apply acoustic-shadow per-pixel weighting '
                             '(suppresses pixels below tall reflectors)')
    parser.add_argument('--edge_w', action='store_true',
                        help='Apply edge-magnitude per-pixel weighting')
    args = parser.parse_args()

    print("Loading calibration...")
    T_tool0_probe, T_probe_us, px_x, px_y, roi = load_calibration(args.calib_dir)
    print(f"  Pixel size: {px_x*1000:.4f} × {px_y*1000:.4f} mm")
    if roi:
        print(f"  ROI: x={roi['x']} y={roi['y']} {roi['w']}×{roi['h']}")

    print("Loading scan data...")
    use_smooth = args.smooth_window > 0 or args.time_lag_ms != 0.0 \
        or args.auto_lag
    if use_smooth:
        lag_s = args.time_lag_ms / 1000.0
        if args.auto_lag:
            print("  Estimating image-pose lag...")
            lag_s = estimate_image_pose_lag(args.scan_dir, roi=roi,
                                             max_lag_s=0.2)
        win = args.smooth_window if args.smooth_window > 0 else 1
        frames, poses = load_scan_data_smooth(
            args.scan_dir, roi=roi,
            force_threshold=args.force_threshold,
            sg_window=win, sg_order=args.smooth_order,
            time_lag_s=lag_s)
    else:
        frames, poses = load_scan_data(args.scan_dir, roi=roi,
                                        force_threshold=args.force_threshold)
    print(f"  Loaded {len(frames)} frames")
    if len(frames) == 0:
        print("ERROR: no frames passed force-threshold filter.")
        return

    if args.pose_percentile > 0:
        print(f"\nFiltering pose outliers (percentile={args.pose_percentile})...")
        frames, poses = filter_pose_outliers(frames, poses,
                                              percentile=args.pose_percentile)
        if len(frames) == 0:
            print("ERROR: no frames left after pose filtering.")
            return

    print("Computing volume bounds...")
    bounds = auto_bounds(poses, T_tool0_probe, T_probe_us,
                         frames[0].shape, px_x, px_y,
                         margin=args.bounds_margin)
    extent = (bounds[1] - bounds[0]) * 1000
    print(f"  Bounds: {bounds.tolist()}")
    print(f"  Extent: {extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} mm")

    # Image-domain pre-processing (before splatting)
    xform_ops = []
    if args.tgc_eq:
        xform_ops.append(lambda im: apply_tgc_eq(
            im, intensity_min=args.intensity_min, gamma=args.tgc_gamma))
    if args.despeckle != 'none':
        xform_ops.append(lambda im: despeckle(
            im, args.despeckle, args.despeckle_ksize))
    if xform_ops:
        print(f"Pre-processing frames ({len(xform_ops)} ops)...")
        frames = transform_frames(frames, compose_xform(*xform_ops))

    # Per-pixel weight functions (multiplicative)
    weight_ops = []
    if args.shadow_w:
        weight_ops.append(shadow_weight)
    if args.edge_w:
        weight_ops.append(edge_weight)
    pixel_w_fn = compose_weight(*weight_ops) if weight_ops else None
    if pixel_w_fn:
        print(f"Per-pixel weight: "
              f"{'shadow ' if args.shadow_w else ''}"
              f"{'edge' if args.edge_w else ''}")

    print(f"\n=== Pass 1: PBM-{args.algo} compounding "
          f"(voxel={args.voxel_size*1000:.2f}mm, "
          f"elev_sigma={args.elevation_sigma_mm:.2f}mm) ===")
    if args.algo == 'v2':
        volume, weight = pbm_compound_v2(
            frames, poses, T_tool0_probe, T_probe_us,
            px_x, px_y, bounds, args.voxel_size,
            elevation_sigma_mm=args.elevation_sigma_mm,
            intensity_min=args.intensity_min,
            trilinear=args.trilinear or True,
            adaptive_elev=args.adaptive_elev,
            adaptive_alpha=args.adaptive_alpha,
            n_sigma_bins=args.n_sigma_bins,
            pixel_weight_fn=pixel_w_fn)
    else:
        if pixel_w_fn:
            print('  WARNING: --shadow_w/--edge_w only supported with '
                  '--algo v2; ignoring.')
        volume, weight = pbm_compound(
            frames, poses, T_tool0_probe, T_probe_us,
            px_x, px_y, bounds, args.voxel_size,
            elevation_sigma_mm=args.elevation_sigma_mm,
            intensity_min=args.intensity_min)

    if args.hole_fill_sigma_mm > 0 and args.hole_fill_iter > 0:
        print(f"\n=== Pass 2: Gaussian hole filling "
              f"(sigma={args.hole_fill_sigma_mm:.2f}mm × "
              f"{args.hole_fill_iter} iters) ===")
        volume, weight = gaussian_hole_fill(
            volume, weight, args.voxel_size,
            sigma_mm=args.hole_fill_sigma_mm,
            max_iter=args.hole_fill_iter)
        fill_ratio = 100.0 * np.count_nonzero(weight) / weight.size
        print(f"  Final fill ratio: {fill_ratio:.1f}% "
              f"({np.count_nonzero(weight):,} / {weight.size:,} voxels)")
    else:
        print("\n(Hole filling disabled)")

    print(f"\nVolume stats: shape={volume.shape}, "
          f"range=[{volume.min():.1f}, {volume.max():.1f}], "
          f"mean(non-zero)={volume[volume>0].mean():.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, volume)
    np.save(args.output.replace('.npy', '_weight.npy'), weight)
    np.save(args.output.replace('.npy', '_bounds.npy'), bounds)
    print(f"\nSaved → {args.output}")


if __name__ == '__main__':
    main()
