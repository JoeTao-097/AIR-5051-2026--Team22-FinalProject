#!/usr/bin/env python3
"""Pose trajectory utilities: smoothing, time interpolation, lag fix.

Why this module exists
─────────────────────
``voxel_reconstruct.load_scan_data`` currently reads the per-frame
metadata row verbatim and uses that pose as-is. This is suboptimal:

1. ``servoL`` + force-mode jitter shows up as ~0.2-0.5 mm high-frequency
   noise in the EE position, which is bigger than typical voxel sizes
   (0.5-0.7 mm) and smears the volume.
2. The pose is the value at the moment the metadata row was written, not
   at the actual ultrasound *exposure* timestamp. There's typically a
   30-60 ms lag between the USB frame's mid-exposure and the ROS
   message arrival, plus interleaved scheduling jitter.

Provided utilities
─────────────────
``load_metadata_full``       — load all rows (no force filter)
``smooth_trajectory``        — Savitzky-Golay on (t, xyz, quat)
``slerp_interpolate``        — sample SE(3) at arbitrary times
``load_scan_data_smooth``    — drop-in replacement for
                                ``voxel_reconstruct.load_scan_data`` that
                                applies smoothing + time interpolation +
                                fixed lag compensation
"""

import csv
import os
import numpy as np

try:
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation as R, Slerp
except ImportError as e:
    raise ImportError("pose_utils requires scipy>=1.4: " + str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Loading raw trajectory
# ─────────────────────────────────────────────────────────────────────────────

def load_metadata_full(scan_dir):
    """Load *every* row of metadata.csv (no filtering).

    Returns dict:
        frame_id : (N,) int
        ts       : (N,) float [seconds]
        pos      : (N, 3) float [meters]
        quat_xyzw: (N, 4) float
        wrench   : (N, 6) float [fx, fy, fz, tx, ty, tz]
    """
    frame_ids, ts, pos, quat, wrench = [], [], [], [], []
    with open(os.path.join(scan_dir, 'metadata.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_ids.append(int(row['frame_id']))
            ts.append(float(row['timestamp']))
            pos.append([float(row['px']), float(row['py']), float(row['pz'])])
            # CSV stores qx qy qz qw — keep xyzw order for scipy
            quat.append([float(row['qx']), float(row['qy']),
                         float(row['qz']), float(row['qw'])])
            wrench.append([float(row[k]) for k in
                           ('fx', 'fy', 'fz', 'tx', 'ty', 'tz')])
    return dict(
        frame_id=np.array(frame_ids, dtype=np.int64),
        ts=np.array(ts, dtype=np.float64),
        pos=np.array(pos, dtype=np.float64),
        quat_xyzw=np.array(quat, dtype=np.float64),
        wrench=np.array(wrench, dtype=np.float64),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion sign continuity (must run before any per-component smoothing
# or component-wise interpolation; flips quaternions to lie on the same
# hemisphere as their predecessor).
# ─────────────────────────────────────────────────────────────────────────────

def quat_make_continuous(quats):
    out = quats.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] = -out[i]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing
# ─────────────────────────────────────────────────────────────────────────────

def smooth_trajectory(pos, quat_xyzw, window=11, order=3):
    """Savitzky-Golay smooth translation + quaternion components.

    For translation we filter (x, y, z) independently. For quaternions
    we make them sign-continuous, filter the 4 components, then
    renormalize. This is a small-angle approximation that works well
    for short windows (≤21 samples) and the milliradian-level joint
    noise we see from ``servoL``.
    """
    if len(pos) < window:
        return pos.copy(), quat_xyzw.copy()

    pos_s = np.empty_like(pos)
    for k in range(3):
        pos_s[:, k] = savgol_filter(pos[:, k], window, order)

    qc = quat_make_continuous(quat_xyzw)
    qs = np.empty_like(qc)
    for k in range(4):
        qs[:, k] = savgol_filter(qc[:, k], window, order)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    return pos_s, qs


# ─────────────────────────────────────────────────────────────────────────────
# Time-resampling to arbitrary timestamps
# ─────────────────────────────────────────────────────────────────────────────

def slerp_interpolate(ts_src, pos_src, quat_xyzw_src, ts_query):
    """Cubic-interp position + Slerp orientation at ``ts_query``.

    Out-of-range timestamps are clamped to the nearest endpoint
    (otherwise ``Slerp`` raises). Returns:
        pos_q   (M, 3)
        quat_q  (M, 4) xyzw
    """
    ts_src = np.asarray(ts_src, dtype=np.float64)
    ts_query = np.clip(np.asarray(ts_query, dtype=np.float64),
                       ts_src[0], ts_src[-1])

    # Position: cubic spline (per-axis)
    from scipy.interpolate import CubicSpline
    pos_q = np.empty((len(ts_query), 3))
    for k in range(3):
        cs = CubicSpline(ts_src, pos_src[:, k], bc_type='natural',
                         extrapolate=False)
        pos_q[:, k] = cs(ts_query)

    # Quaternions: Slerp (handles sign continuity internally)
    rot_src = R.from_quat(quat_make_continuous(quat_xyzw_src))
    slerp = Slerp(ts_src, rot_src)
    quat_q = slerp(ts_query).as_quat()  # xyzw
    return pos_q, quat_q


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in replacement for voxel_reconstruct.load_scan_data
# ─────────────────────────────────────────────────────────────────────────────

def load_scan_data_smooth(scan_dir, roi=None, force_threshold=1.0,
                          sg_window=11, sg_order=3,
                          time_lag_s=0.0,
                          verbose=True):
    """Like ``load_scan_data`` but with smoothed and time-aligned poses.

    Pipeline:
        1. Read every metadata row → dense trajectory.
        2. Savitzky-Golay smooth translation + quaternion (sign-continuous).
        3. For each frame_id passing the force filter, interpolate the
           smoothed trajectory at ``ts_image - time_lag_s``.
        4. Load the image (with optional ROI crop).

    Args:
        time_lag_s: positive value means image samples lag the pose
            stream by this many seconds (typical 0.02-0.06 for USB
            capture). Estimated by cross-correlating wrench vs image
            mean if you don't know it (see ``estimate_image_pose_lag``).
    """
    import cv2
    import transforms3d.quaternions as tq

    md = load_metadata_full(scan_dir)
    if verbose:
        print(f'  Trajectory: {len(md["ts"])} samples, '
              f'duration {md["ts"][-1]-md["ts"][0]:.1f}s')

    # 1) Smooth the full trajectory
    if sg_window > 1 and len(md['pos']) >= sg_window:
        pos_s, quat_s = smooth_trajectory(md['pos'], md['quat_xyzw'],
                                          window=sg_window, order=sg_order)
        if verbose:
            disp = np.linalg.norm(pos_s - md['pos'], axis=1)
            print(f'  SG smoothing (win={sg_window}, order={sg_order}): '
                  f'pos correction RMS={disp.mean()*1000:.3f}mm, '
                  f'max={disp.max()*1000:.3f}mm')
    else:
        pos_s = md['pos']; quat_s = md['quat_xyzw']

    # 2) Force-threshold filter (on original wrench, not smoothed)
    fz = md['wrench'][:, 2]
    keep_mask = np.abs(fz) >= force_threshold
    keep_idx = np.where(keep_mask)[0]
    n_dropped_force = int((~keep_mask).sum())

    # 3) Time-corrected query timestamps
    ts_query = md['ts'][keep_idx] - time_lag_s

    # 4) Interpolate smoothed poses at frame timestamps
    if sg_window > 1 or time_lag_s != 0.0:
        pos_q, quat_q = slerp_interpolate(md['ts'], pos_s, quat_s, ts_query)
    else:
        pos_q = md['pos'][keep_idx]
        quat_q = md['quat_xyzw'][keep_idx]

    # 5) Load images
    frames, poses, used_idx = [], [], []
    skipped_img = 0
    for j, src_i in enumerate(keep_idx):
        fid = int(md['frame_id'][src_i])
        img_path = os.path.join(scan_dir, 'frames', f'{fid:06d}.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped_img += 1
            continue
        if roi is not None:
            img = img[roi['y']:roi['y'] + roi['h'],
                      roi['x']:roi['x'] + roi['w']]

        T = np.eye(4)
        # transforms3d uses [w, x, y, z]; we have [x, y, z, w]
        q = quat_q[j]
        T[:3, :3] = tq.quat2mat([q[3], q[0], q[1], q[2]])
        T[:3, 3] = pos_q[j]
        frames.append(img)
        poses.append(T)
        used_idx.append(src_i)

    if verbose:
        if roi is not None:
            print(f'  ROI crop: x={roi["x"]} y={roi["y"]} '
                  f'{roi["w"]}×{roi["h"]}')
        print(f'  Filtered: {len(frames)}/{len(md["ts"])} frames kept '
              f'(|Fz|>={force_threshold:.1f}N, dropped {n_dropped_force} '
              f'low-force, {skipped_img} missing images)')
        if time_lag_s != 0.0:
            print(f'  Time-lag compensation: {time_lag_s*1000:+.1f} ms')

    return frames, poses


# ─────────────────────────────────────────────────────────────────────────────
# Optional: estimate image-pose timing lag via mean-intensity vs |Fz|
# ─────────────────────────────────────────────────────────────────────────────

def estimate_image_pose_lag(scan_dir, roi=None, max_lag_s=0.2,
                            stride=2, verbose=True):
    """Cross-correlate per-frame mean intensity with |Fz| over time.

    The intuition: when the probe presses harder (|Fz| ↑), the contact
    coupling improves and the mean image intensity tends to rise. If
    image timestamps lag the wrench stream by Δt, the cross-correlation
    will peak at Δt. Returns the estimated lag in seconds.

    NOTE: the sign convention here is "positive lag = image samples are
    delayed wrt pose stream", i.e. you should query the trajectory at
    ``ts_image - lag``. Pass the returned value directly as
    ``time_lag_s`` to ``load_scan_data_smooth``.
    """
    import cv2
    md = load_metadata_full(scan_dir)
    fz = np.abs(md['wrench'][:, 2])

    # Subsample frames to keep this fast
    sel = np.arange(0, len(md['ts']), stride)
    means = np.zeros(len(sel))
    for k, i in enumerate(sel):
        fid = int(md['frame_id'][i])
        img = cv2.imread(os.path.join(scan_dir, 'frames',
                                       f'{fid:06d}.png'),
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if roi is not None:
            img = img[roi['y']:roi['y'] + roi['h'],
                      roi['x']:roi['x'] + roi['w']]
        means[k] = float(img.mean())

    # Resample both signals at uniform 30 Hz over [t0, t1]
    t0, t1 = md['ts'][sel[0]], md['ts'][sel[-1]]
    fs = 30.0
    t_uni = np.arange(t0, t1, 1.0 / fs)
    fz_uni = np.interp(t_uni, md['ts'][sel], fz[sel])
    im_uni = np.interp(t_uni, md['ts'][sel], means)

    fz_uni -= fz_uni.mean(); im_uni -= im_uni.mean()
    fz_uni /= (fz_uni.std() + 1e-9); im_uni /= (im_uni.std() + 1e-9)

    max_lag_n = int(max_lag_s * fs)
    lags = np.arange(-max_lag_n, max_lag_n + 1)
    xc = np.array([np.mean(fz_uni[max(0, -L):len(fz_uni) - max(0, L)] *
                            im_uni[max(0, L):len(im_uni) - max(0, -L)])
                   for L in lags])
    best = int(np.argmax(xc))
    lag_s = float(lags[best] / fs)
    if verbose:
        print(f'  Lag scan ±{max_lag_s*1000:.0f} ms: peak xc={xc[best]:.3f} '
              f'at Δt={lag_s*1000:+.1f} ms')
    return lag_s
