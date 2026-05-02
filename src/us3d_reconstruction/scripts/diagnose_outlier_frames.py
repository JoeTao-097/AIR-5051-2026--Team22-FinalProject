#!/usr/bin/env python3
"""Identify frames whose pose/intensity distribution is anomalous.

Usage:
    python3 diagnose_outlier_frames.py \
        --scan_dir data/scans/scan_20260407_150839 \
        --calib_dir data/calibration \
        --force_threshold 5.0
"""

import argparse
import csv
import os
import numpy as np
import cv2
import yaml


def load_calib(calib_dir):
    with open(os.path.join(calib_dir, 'probe_calibration.yaml')) as f:
        c = yaml.safe_load(f)
    roi = None
    if 'us_roi_x' in c:
        roi = dict(x=c['us_roi_x'], y=c['us_roi_y'],
                   w=c['us_roi_width'], h=c['us_roi_height'])
    return roi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan_dir', required=True)
    ap.add_argument('--calib_dir', required=True)
    ap.add_argument('--force_threshold', type=float, default=1.0)
    ap.add_argument('--intensity_min', type=int, default=15)
    args = ap.parse_args()

    roi = load_calib(args.calib_dir)
    rows = []
    with open(os.path.join(args.scan_dir, 'metadata.csv')) as f:
        for row in csv.DictReader(f):
            if abs(float(row['fz'])) >= args.force_threshold:
                rows.append(row)

    print(f'Loaded {len(rows)} frames passing |Fz|>={args.force_threshold}N\n')

    # Pose statistics
    pos = np.array([[float(r['px']), float(r['py']), float(r['pz'])] for r in rows])
    fz  = np.array([float(r['fz']) for r in rows])
    fids = [int(r['frame_id']) for r in rows]

    pos_med = np.median(pos, axis=0)
    pos_mad = np.median(np.abs(pos - pos_med), axis=0)
    print(f'Pose median: x={pos_med[0]*1000:.1f} y={pos_med[1]*1000:.1f} '
          f'z={pos_med[2]*1000:.1f} mm')
    print(f'Pose MAD:    x={pos_mad[0]*1000:.1f} y={pos_mad[1]*1000:.1f} '
          f'z={pos_mad[2]*1000:.1f} mm')

    # Distance from median pose (in MAD units)
    dev = np.abs(pos - pos_med) / np.maximum(pos_mad, 1e-4)
    pose_score = dev.max(axis=1)

    # Image statistics: mean intensity inside ROI, fraction of bright pixels
    bright_frac = []
    mean_int = []
    for r in rows:
        fid = int(r['frame_id'])
        img = cv2.imread(os.path.join(args.scan_dir, 'frames',
                                       f'{fid:06d}.png'), cv2.IMREAD_GRAYSCALE)
        if img is None:
            bright_frac.append(0); mean_int.append(0); continue
        if roi:
            img = img[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
        bright_frac.append(np.mean(img >= args.intensity_min))
        mean_int.append(img.mean())
    bright_frac = np.array(bright_frac)
    mean_int = np.array(mean_int)

    # Top outliers
    print(f'\n=== TOP 15 POSE OUTLIERS (deviating most from median) ===')
    idx = np.argsort(-pose_score)[:15]
    print(f'{"frame":>6} {"Fz":>7} {"X":>7} {"Y":>7} {"Z":>7} '
          f'{"dev_X":>6} {"dev_Y":>6} {"dev_Z":>6} {"bright%":>7} {"mean_I":>6}')
    for i in idx:
        print(f'{fids[i]:>6} {fz[i]:>7.2f} '
              f'{pos[i,0]*1000:>7.1f} {pos[i,1]*1000:>7.1f} {pos[i,2]*1000:>7.1f} '
              f'{dev[i,0]:>6.1f} {dev[i,1]:>6.1f} {dev[i,2]:>6.1f} '
              f'{bright_frac[i]*100:>6.1f} {mean_int[i]:>6.1f}')

    print(f'\n=== TOP 10 BRIGHTEST FRAMES (likely air contact / artifacts) ===')
    idx = np.argsort(-mean_int)[:10]
    for i in idx:
        print(f'{fids[i]:>6} fz={fz[i]:>6.2f}N  '
              f'mean_I={mean_int[i]:>5.1f}  bright%={bright_frac[i]*100:>5.1f}  '
              f'pos=({pos[i,0]*1000:>6.1f},{pos[i,1]*1000:>6.1f},{pos[i,2]*1000:>6.1f})')

    print(f'\n=== TOP 10 DIMMEST FRAMES (possibly probe lifted) ===')
    idx = np.argsort(mean_int)[:10]
    for i in idx:
        print(f'{fids[i]:>6} fz={fz[i]:>6.2f}N  '
              f'mean_I={mean_int[i]:>5.1f}  bright%={bright_frac[i]*100:>5.1f}  '
              f'pos=({pos[i,0]*1000:>6.1f},{pos[i,1]*1000:>6.1f},{pos[i,2]*1000:>6.1f})')

    # Suggest a tighter window
    p25, p75 = np.percentile(pos, [10, 90], axis=0)
    print(f'\n=== SUGGESTED POSE WINDOW (10-90 percentile) ===')
    print(f'X: [{p25[0]*1000:.1f}, {p75[0]*1000:.1f}] mm  '
          f'(span {(p75[0]-p25[0])*1000:.1f} mm)')
    print(f'Y: [{p25[1]*1000:.1f}, {p75[1]*1000:.1f}] mm  '
          f'(span {(p75[1]-p25[1])*1000:.1f} mm)')
    print(f'Z: [{p25[2]*1000:.1f}, {p75[2]*1000:.1f}] mm  '
          f'(span {(p75[2]-p25[2])*1000:.1f} mm)')

    n_in_window = np.sum(np.all((pos >= p25) & (pos <= p75), axis=1))
    print(f'Frames inside this window: {n_in_window} / {len(rows)}')


if __name__ == '__main__':
    main()
