#!/usr/bin/env python3
"""Voxel embedding 3D reconstruction.

Reads scan data (US frames + poses) and maps each pixel into a 3D
voxel grid using the calibrated transforms. Vectorized for performance.

Usage:
    python3 voxel_reconstruct.py --scan_dir data/scans/scan_001 \
                                  --calib_dir data/calibration \
                                  --output data/reconstructions/vol_001.npy
"""

import argparse
import os
import csv
import numpy as np
import cv2
import yaml
import transforms3d.quaternions as tq


def load_calibration(calib_dir):
    """Load T_tool0_probe, T_probe_us, pixel sizes, and ROI from calibration files."""
    with open(os.path.join(calib_dir, 'probe_calibration.yaml'), 'r') as f:
        probe_cal = yaml.safe_load(f)

    T_tool0_probe = np.array(probe_cal['T_tool0_probe']).reshape(4, 4)
    T_probe_us = np.array(probe_cal['T_probe_us']).reshape(4, 4)
    pixel_size_x = probe_cal['pixel_size_x']
    pixel_size_y = probe_cal['pixel_size_y']

    roi = None
    if 'us_roi_x' in probe_cal:
        roi = {
            'x': probe_cal['us_roi_x'],
            'y': probe_cal['us_roi_y'],
            'w': probe_cal['us_roi_width'],
            'h': probe_cal['us_roi_height'],
        }

    return T_tool0_probe, T_probe_us, pixel_size_x, pixel_size_y, roi


def load_scan_data(scan_dir, roi=None, force_threshold=1.0):
    """Load frames and metadata, filtering to only keep scanning frames.

    Frames are considered valid scanning data when the contact force
    exceeds force_threshold (probe is pressing on the surface).

    Args:
        scan_dir: path to scan directory
        roi: dict with x, y, w, h to crop USB capture GUI elements
        force_threshold: minimum |Fz| in N to consider a frame as scanning
    """
    metadata_path = os.path.join(scan_dir, 'metadata.csv')
    frames = []
    poses = []
    total = 0
    skipped = 0

    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            fz = float(row['fz'])

            if abs(fz) < force_threshold:
                skipped += 1
                continue

            frame_id = int(row['frame_id'])
            img_path = os.path.join(scan_dir, 'frames', '%06d.png' % frame_id)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped += 1
                continue

            if roi is not None:
                img = img[roi['y']:roi['y']+roi['h'],
                          roi['x']:roi['x']+roi['w']]

            pos = np.array([float(row['px']), float(row['py']), float(row['pz'])])
            quat = np.array([float(row['qw']), float(row['qx']),
                             float(row['qy']), float(row['qz'])])

            T = np.eye(4)
            T[:3, :3] = tq.quat2mat(quat)
            T[:3, 3] = pos

            frames.append(img)
            poses.append(T)

    if roi is not None:
        print("  Applied ROI crop: x=%d y=%d %dx%d" % (roi['x'], roi['y'], roi['w'], roi['h']))
    print("  Filtered: %d/%d frames kept (|Fz|>=%.1fN), %d skipped" %
          (len(frames), total, force_threshold, skipped))

    return frames, poses


def voxel_embedding(frames, poses, T_tool0_probe, T_probe_us,
                    pixel_size_x, pixel_size_y, volume_bounds, voxel_size):
    """
    Map ultrasound pixels into a 3D voxel grid (vectorized).

    Args:
        frames: list of grayscale images
        poses: list of 4x4 T_base_tool0 matrices
        volume_bounds: (2,3) array [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        voxel_size: scalar voxel edge length in meters
    """
    grid_shape = np.ceil((volume_bounds[1] - volume_bounds[0]) / voxel_size).astype(int)
    volume = np.zeros(grid_shape, dtype=np.float64)
    weight = np.zeros(grid_shape, dtype=np.float64)

    T_tool0_us = T_tool0_probe @ T_probe_us

    for img, T_base_tool0 in zip(frames, poses):
        T_base_us = T_base_tool0 @ T_tool0_us

        h, w = img.shape
        u_coords = np.arange(w) * pixel_size_x
        v_coords = np.arange(h) * pixel_size_y

        uu, vv = np.meshgrid(u_coords, v_coords)
        ones = np.ones_like(uu)
        zeros = np.zeros_like(uu)

        # (4, H*W) homogeneous coordinates in US plane
        p_us = np.stack([uu.ravel(), vv.ravel(), zeros.ravel(), ones.ravel()], axis=0)

        # Transform to base frame
        p_base = T_base_us @ p_us  # (4, H*W)

        # Map to voxel indices
        voxel_coords = ((p_base[:3].T - volume_bounds[0]) / voxel_size).astype(int)

        # Filter valid indices
        valid = np.all((voxel_coords >= 0) & (voxel_coords < grid_shape), axis=1)
        valid_voxels = voxel_coords[valid]
        valid_intensities = img.ravel()[valid].astype(np.float64)

        # Accumulate
        np.add.at(volume, (valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]),
                  valid_intensities)
        np.add.at(weight, (valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]), 1.0)

    mask = weight > 0
    volume[mask] /= weight[mask]

    return volume.astype(np.float32), weight.astype(np.float32)


def auto_bounds(poses, T_tool0_probe, T_probe_us, img_shape, pixel_size_x, pixel_size_y, margin=0.02):
    """Automatically compute volume bounds from scan poses."""
    T_tool0_us = T_tool0_probe @ T_probe_us
    h, w = img_shape

    corners_us = np.array([
        [0, 0, 0, 1],
        [w * pixel_size_x, 0, 0, 1],
        [0, h * pixel_size_y, 0, 1],
        [w * pixel_size_x, h * pixel_size_y, 0, 1],
    ]).T  # (4, 4)

    all_points = []
    for T_base_tool0 in poses:
        T_base_us = T_base_tool0 @ T_tool0_us
        p_base = T_base_us @ corners_us
        all_points.append(p_base[:3].T)

    all_points = np.vstack(all_points)
    bounds = np.array([all_points.min(axis=0) - margin,
                       all_points.max(axis=0) + margin])
    return bounds


def main():
    parser = argparse.ArgumentParser(description='Voxel embedding 3D reconstruction')
    parser.add_argument('--scan_dir', required=True, help='Path to scan data directory')
    parser.add_argument('--calib_dir', required=True, help='Path to calibration directory')
    parser.add_argument('--output', required=True, help='Output .npy file path')
    parser.add_argument('--voxel_size', type=float, default=0.0005, help='Voxel size in meters')
    args = parser.parse_args()

    print("Loading calibration...")
    T_tool0_probe, T_probe_us, px_x, px_y, roi = load_calibration(args.calib_dir)
    print("  Pixel size: %.4f x %.4f mm" % (px_x * 1000, px_y * 1000))
    if roi:
        print("  ROI: x=%d y=%d %dx%d" % (roi['x'], roi['y'], roi['w'], roi['h']))

    print("Loading scan data...")
    frames, poses = load_scan_data(args.scan_dir, roi=roi)
    print("  Loaded %d frames" % len(frames))

    print("Computing volume bounds...")
    bounds = auto_bounds(poses, T_tool0_probe, T_probe_us,
                         frames[0].shape, px_x, px_y)
    print("  Bounds: %s" % bounds)

    print("Running voxel embedding (voxel_size=%.2fmm)..." % (args.voxel_size * 1000))
    volume, weight = voxel_embedding(
        frames, poses, T_tool0_probe, T_probe_us,
        px_x, px_y, bounds, args.voxel_size)

    print("  Volume shape: %s" % str(volume.shape))
    print("  Non-zero voxels: %d / %d (%.1f%%)" % (
        np.count_nonzero(weight), weight.size,
        100.0 * np.count_nonzero(weight) / weight.size))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, volume)
    np.save(args.output.replace('.npy', '_weight.npy'), weight)
    np.save(args.output.replace('.npy', '_bounds.npy'), bounds)
    print("Saved to %s" % args.output)


if __name__ == '__main__':
    main()
