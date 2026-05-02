#!/usr/bin/env python3
"""Curved-surface scan path planner node.

Combines ArUco marker localization with depth point cloud to generate
scan paths that conform to the phantom's curved surface.

Pipeline:
  1. Read ArUco marker positions (/us3d/markers) to define ROI
  2. Capture and transform point cloud to base_link
  3. Crop, filter, and fit the surface using RBF interpolation
  4. Generate parallel scan lines projected onto the fitted surface
  5. Publish PoseArray on /us3d/scan_path
"""

import numpy as np
import open3d as o3d
import rospy
import tf2_ros
from scipy.interpolate import RBFInterpolator
from scipy.signal import savgol_filter
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from std_srvs.srv import Trigger, TriggerResponse
import transforms3d.quaternions as tq
import struct


# ─────────────────────────────────────────────────────────────────────────────
# Path smoothing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _smooth_z(z_array, window_mm=15.0, step_mm=1.0):
    """Savitzky-Golay smooth a sequence of Z values along the scan path.

    RBF surface fits over sparse depth-cloud points produce small
    high-frequency Z bumps that, in a freehand 3D US sweep, manifest as
    the probe alternately pressing and lifting off the surface.
    Smoothing the Z curve after generation removes these bumps without
    losing the overall surface shape.
    """
    n = len(z_array)
    if n < 5:
        return z_array
    win = max(5, int(window_mm / step_mm) | 1)   # odd, ≥5
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 5:
        return z_array
    poly = min(3, win - 1)
    return savgol_filter(z_array, window_length=win, polyorder=poly,
                         mode='nearest')


def _quat_slerp_smooth(quats, window=11):
    """Smooth a sequence of quaternions by SG-filtering each component
    after hemispheric alignment, then renormalising. Window auto-clamped
    to be odd and ≤ length of the sequence.
    """
    n = len(quats)
    if n < 5:
        return quats
    q = np.asarray(quats, dtype=np.float64)
    # Align hemispheres so that adjacent quaternions don't double-flip
    for i in range(1, n):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    win = max(5, window | 1)
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 5:
        return q
    poly = min(3, win - 1)
    smoothed = np.empty_like(q)
    for c in range(4):
        smoothed[:, c] = savgol_filter(q[:, c], window_length=win,
                                        polyorder=poly, mode='nearest')
    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    return smoothed / np.maximum(norms, 1e-9)


def _angle_between_quats_arr(q_arr):
    """Return inter-element angles (deg) for a sequence of quaternions."""
    if len(q_arr) < 2:
        return np.array([])
    angs = []
    for i in range(1, len(q_arr)):
        a = q_arr[i - 1]
        b = q_arr[i]
        if np.dot(a, b) < 0:
            b = -b
        cos_half = np.clip(abs(np.dot(a, b)), -1.0, 1.0)
        angs.append(np.degrees(2.0 * np.arccos(cos_half)))
    return np.array(angs)


class ScanPlannerNode:
    def __init__(self):
        rospy.init_node('scan_planner_node')

        self.line_spacing = rospy.get_param('/scan/line_spacing', 0.003)
        self.waypoint_step = rospy.get_param('/scan/waypoint_step', 0.001)
        self.scan_half_length = rospy.get_param('/scan/scan_half_length', 0.05)
        self.voxel_size = rospy.get_param('/surface_fitting/voxel_size', 0.002)
        self.crop_margin = rospy.get_param('/surface_fitting/crop_margin', 0.02)
        self.outlier_nb = rospy.get_param('/surface_fitting/outlier_nb_neighbors', 20)
        self.outlier_std = rospy.get_param('/surface_fitting/outlier_std_ratio', 2.0)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self._markers = None
        self._cloud_msg = None

        rospy.Subscriber('/us3d/markers', PoseArray, self._markers_cb)
        rospy.Subscriber('/camera/depth/points', PointCloud2,
                         self._cloud_cb, queue_size=1, buff_size=2**24)

        self.pub_path = rospy.Publisher('/us3d/scan_path', PoseArray,
                                       queue_size=1, latch=True)
        self.pub_surface = rospy.Publisher('/us3d/surface_cloud', PointCloud2,
                                          queue_size=1, latch=True)

        rospy.Service('/us3d/plan_scan', Trigger, self._plan_scan_srv)

        rospy.loginfo("Scan planner ready (voxel=%.1fmm, margin=%.0fmm)",
                      self.voxel_size * 1000, self.crop_margin * 1000)

    def _markers_cb(self, msg):
        self._markers = msg

    def _cloud_cb(self, msg):
        self._cloud_msg = msg

    def _plan_scan_srv(self, req):
        if self._markers is None or len(self._markers.poses) < 2:
            return TriggerResponse(success=False,
                                   message="Need at least 2 markers (ID 0 and 1). "
                                           "Call /us3d/detect_markers first.")

        # Re-read parameters every time the service is called so that
        # callers (instruction_planner, manual rosparam set, etc.)
        # can change /scan/scan_half_length / line_spacing / waypoint_step
        # at runtime and have the change take effect on the very next
        # plan_scan call. Without this, scan length is frozen to
        # whatever value was on the param server when the node started.
        prev_half = self.scan_half_length
        self.scan_half_length = float(rospy.get_param(
            '/scan/scan_half_length', self.scan_half_length))
        self.line_spacing = float(rospy.get_param(
            '/scan/line_spacing', self.line_spacing))
        self.waypoint_step = float(rospy.get_param(
            '/scan/waypoint_step', self.waypoint_step))
        if abs(self.scan_half_length - prev_half) > 1e-6:
            rospy.loginfo(
                "scan_half_length updated: %.0fmm -> %.0fmm (full "
                "scan length %.0fmm)",
                prev_half * 1000, self.scan_half_length * 1000,
                self.scan_half_length * 2 * 1000)

        try:
            if self._cloud_msg is not None:
                pose_array = self._plan_scan_with_surface()
                method = "surface-fitted"
            else:
                rospy.logwarn("No point cloud available, falling back to linear path")
                pose_array = self._plan_scan_two_markers()
                method = "linear"
            self.pub_path.publish(pose_array)
            msg = ("Generated %d waypoints (%s, marker 0 -> marker 1, "
                   "scan_length=%.0fmm)" % (
                len(pose_array.poses), method,
                self.scan_half_length * 2 * 1000))
            rospy.loginfo(msg)
            return TriggerResponse(success=True, message=msg)
        except Exception as e:
            rospy.logerr("Scan planning failed: %s", str(e))
            return TriggerResponse(success=False,
                                   message="Planning failed: %s" % str(e))

    def _compute_scan_region(self):
        """Compute scan center, direction, and start/end from two markers.

        Markers are on the phantom exterior; the scan region is centered
        at their midpoint and extends ±scan_half_length along their axis.
        """
        p0 = self._markers.poses[0].position
        p1 = self._markers.poses[1].position
        m0 = np.array([p0.x, p0.y, p0.z])
        m1 = np.array([p1.x, p1.y, p1.z])

        # Optionally reverse marker order (useful when the physical
        # ArUco IDs were placed in the opposite of the desired scan
        # direction, so user can fix without rearranging markers).
        if rospy.get_param('/scan/reverse_scan_direction', False):
            m0, m1 = m1, m0
            rospy.loginfo("Scan direction REVERSED via "
                          "/scan/reverse_scan_direction: marker1 → marker0")

        center = (m0 + m1) / 2.0
        direction = m1 - m0
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            raise RuntimeError("Markers are too close together (%.4f m)" % dist)
        direction /= dist

        half = self.scan_half_length
        scan_start = center - half * direction
        scan_end = center + half * direction

        rospy.loginfo("Scan region: center=(%.3f,%.3f,%.3f), dir=(%.3f,%.3f,%.3f), "
                      "length=%.0fmm (±%.0fmm from center)",
                      center[0], center[1], center[2],
                      direction[0], direction[1], direction[2],
                      2 * half * 1000, half * 1000)
        return scan_start, scan_end, center, direction

    def _plan_scan_with_surface(self):
        """Generate a curved path centered between two markers using depth point cloud."""
        scan_start, scan_end, center, scan_dir = self._compute_scan_region()
        marker_points = np.vstack([scan_start, scan_end])

        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                'base_link', self._cloud_msg.header.frame_id,
                rospy.Time(0), rospy.Duration(2.0))
            T = self._tf_to_matrix(tf_stamped.transform)
        except Exception as e:
            rospy.logwarn("TF lookup failed (%s), using cloud frame as-is", e)
            T = np.eye(4)

        pcd = self._pointcloud2_to_o3d(self._cloud_msg)
        if len(pcd.points) == 0:
            raise RuntimeError("Point cloud is empty after conversion")

        pts_h = np.hstack([np.asarray(pcd.points),
                           np.ones((len(pcd.points), 1))])
        pts_base = (T @ pts_h.T).T[:, :3]
        pcd.points = o3d.utility.Vector3dVector(pts_base)

        pcd = self._crop_pointcloud(pcd, marker_points)
        rospy.loginfo("After crop: %d points", len(pcd.points))
        if len(pcd.points) < 10:
            raise RuntimeError("Too few points (%d) after cropping" % len(pcd.points))

        pcd = self._filter_pointcloud(pcd)
        rospy.loginfo("After filter: %d points", len(pcd.points))
        if len(pcd.points) < 5:
            raise RuntimeError("Too few points (%d) after filtering — "
                               "need at least 5 for surface fit. The "
                               "depth cloud may not be capturing the "
                               "phantom surface; check camera angle."
                               % len(pcd.points))

        self._publish_surface_cloud(pcd)

        surface_pts = np.asarray(pcd.points)

        # Diagnostic: project points onto scan_dir, see if they trend
        # ascending or descending in Z. This is the "ground truth" of
        # what the depth cloud says (independent of RBF interpolation).
        if len(surface_pts) >= 3:
            scan_dir_2d = np.array([scan_dir[0], scan_dir[1]])
            scan_dir_2d /= max(np.linalg.norm(scan_dir_2d), 1e-9)
            projs = (surface_pts[:, :2] - scan_start[:2]) @ scan_dir_2d
            order = np.argsort(projs)
            n = len(order)
            head_avg_z = float(np.mean(surface_pts[order[:max(3, n//4)], 2]))
            tail_avg_z = float(np.mean(surface_pts[order[-max(3, n//4):], 2]))
            rospy.loginfo(
                "DEPTH-CLOUD trend along scan dir: %d filtered points, "
                "head (start side) avg Z=%.4f, tail (end side) avg Z=%.4f, "
                "trend=%s (delta=%+.1fmm)",
                n, head_avg_z, tail_avg_z,
                "ASCENDING" if tail_avg_z > head_avg_z else "DESCENDING",
                (tail_avg_z - head_avg_z) * 1000)

        interpolator = self._fit_surface(surface_pts)

        total_length = 2.0 * self.scan_half_length
        n_points = max(2, int(total_length / self.waypoint_step))
        eps = self.waypoint_step * 0.1

        lateral_dir = np.array([-scan_dir[1], scan_dir[0], 0.0])
        norm_lat = np.linalg.norm(lateral_dir)
        if norm_lat < 1e-6:
            lateral_dir = np.array([0.0, 1.0, 0.0])
        else:
            lateral_dir /= norm_lat

        # ── Pass 1: raw evaluation at each waypoint ───────────────
        # Use a wider eps so finite-difference normals are less noisy
        # (small eps amplifies any local RBF bump into a large normal
        # tilt → shaky probe orientation).
        eps_normal = max(self.waypoint_step * 5, 0.005)  # ≥5 mm

        positions_xy = []
        z_vals_raw = []
        normals_raw = []
        for j in range(n_points + 1):
            t = j / float(n_points)
            pos_xy = scan_start[:2] + t * (scan_end[:2] - scan_start[:2])
            z = float(interpolator(pos_xy.reshape(1, -1)))

            z_dx = float(interpolator(
                (pos_xy + eps_normal * scan_dir[:2]).reshape(1, -1)))
            z_dy = float(interpolator(
                (pos_xy + eps_normal * lateral_dir[:2]).reshape(1, -1)))
            dzdx = (z_dx - z) / eps_normal
            dzdy = (z_dy - z) / eps_normal
            normal = np.array([-dzdx, -dzdy, 1.0])
            normal /= np.linalg.norm(normal)

            positions_xy.append(pos_xy)
            z_vals_raw.append(z)
            normals_raw.append(normal)

        z_vals_raw = np.asarray(z_vals_raw)
        normals_raw = np.asarray(normals_raw)
        positions_xy = np.asarray(positions_xy)

        # ── ROOT-CAUSE DIAGNOSTIC ──────────────────────────────────
        # Print RAW Z values along the path BEFORE any smoothing.
        # This is what RBF gave us directly from the depth cloud.
        # Compare with the depth cloud you see in RViz (/us3d/surface_cloud)
        # — if the trend here matches the cloud, the planner is faithful.
        # If the trend here is REVERSED vs the cloud, there's a bug
        # upstream (RBF / point cloud transform / TF).
        rospy.loginfo(
            "RAW path Z (BEFORE smoothing): start=%.4f, mid=%.4f, end=%.4f, "
            "trend=%s (delta_end_minus_start=%+.1fmm)",
            float(z_vals_raw[0]),
            float(z_vals_raw[len(z_vals_raw) // 2]),
            float(z_vals_raw[-1]),
            "ASCENDING" if z_vals_raw[-1] > z_vals_raw[0] else "DESCENDING",
            float(z_vals_raw[-1] - z_vals_raw[0]) * 1000)

        # ── Pass 2: smooth Z and normals along the path ────────────
        # Z smoothing: SG filter with large window (≈ 30 mm) to remove
        # high-frequency RBF artifacts from sparse point clouds while
        # preserving the overall surface shape.
        z_smooth_window_mm = rospy.get_param(
            '/scan/z_smooth_window_mm', 30.0)
        z_vals = _smooth_z(z_vals_raw,
                           window_mm=z_smooth_window_mm,
                           step_mm=self.waypoint_step * 1000)

        # Diagnostic: did smoothing flip the trend?
        rospy.loginfo(
            "SMOOTHED path Z (AFTER SG filter): start=%.4f, mid=%.4f, end=%.4f, "
            "trend=%s (delta_end_minus_start=%+.1fmm)",
            float(z_vals[0]),
            float(z_vals[len(z_vals) // 2]),
            float(z_vals[-1]),
            "ASCENDING" if z_vals[-1] > z_vals[0] else "DESCENDING",
            float(z_vals[-1] - z_vals[0]) * 1000)

        # Sanity check: make sure smoothing doesn't flip the sign
        raw_trend = z_vals_raw[-1] - z_vals_raw[0]
        smooth_trend = z_vals[-1] - z_vals[0]
        if abs(raw_trend) > 0.001 and (raw_trend * smooth_trend < 0):
            rospy.logerr(
                "BUG: SG smoothing FLIPPED the Z trend sign! "
                "raw=%+.1fmm, smoothed=%+.1fmm. This should never happen.",
                raw_trend * 1000, smooth_trend * 1000)

        # Decide flat-surface vs curved.
        z_std_mm = float(np.std(z_vals) * 1000)
        z_range_mm = float((z_vals.max() - z_vals.min()) * 1000)
        # Default to a HUGE threshold so flat-mode kicks in unless
        # the user explicitly opts out by setting a small threshold
        # AND has dense, reliable depth-cloud data. Small thresholds
        # (e.g. 6mm) caused noisy RBF extrapolation to drive the path
        # 20-30mm up/down between adjacent waypoints, producing
        # protective stops when the probe collided with the surface.
        flat_threshold_mm = rospy.get_param(
            '/scan/flat_surface_threshold_mm', 1000.0)
        use_constant_orientation = z_range_mm < flat_threshold_mm

        if use_constant_orientation:
            force_vertical = rospy.get_param(
                '/scan/flat_force_vertical', True)
            flatten_z = rospy.get_param('/scan/flat_use_constant_z', False)
            if force_vertical:
                avg_normal = np.array([0.0, 0.0, 1.0])
                ori_msg = "FORCED VERTICAL"
            else:
                avg_normal = normals_raw.mean(axis=0)
                avg_normal /= np.linalg.norm(avg_normal)
                ori_msg = "averaged depth-cloud normal"
            normals = np.tile(avg_normal, (len(z_vals), 1))

            if flatten_z:
                # Replace the noisy RBF Z curve with a single
                # constant Z = median of the cloud surface. The
                # touchdown step will refine this with real force
                # feedback, so the small variation we lose here is
                # not informative anyway.
                flat_z = float(np.median(z_vals))
                z_vals = np.full_like(z_vals, flat_z)
                rospy.loginfo(
                    "Flat mode: orientation=%s, Z FLATTENED to constant "
                    "%.4f m (raw Z range was %.1fmm — likely RBF noise "
                    "on %d sparse points; touchdown will calibrate)",
                    ori_msg, flat_z, z_range_mm, len(surface_pts))
            else:
                rospy.loginfo(
                    "Flat mode: orientation=%s (Z range=%.1fmm)",
                    ori_msg, z_range_mm)
        else:
            # Smooth normals (SG on each component, then renormalise).
            ns = np.empty_like(normals_raw)
            n_pts = len(normals_raw)
            win = min(max(5, int(15.0 / (self.waypoint_step * 1000)) | 1),
                      n_pts if n_pts % 2 == 1 else n_pts - 1)
            if win >= 5:
                poly = min(3, win - 1)
                for c in range(3):
                    ns[:, c] = savgol_filter(normals_raw[:, c],
                                              window_length=win,
                                              polyorder=poly,
                                              mode='nearest')
                ns /= np.linalg.norm(ns, axis=1, keepdims=True)
            else:
                ns = normals_raw
            normals = ns
            rospy.loginfo(
                "Surface is curved (Z range=%.1fmm) → per-waypoint "
                "orientations (smoothed window=%d)", z_range_mm, win)

        # ── Pass 3: build Pose array with smoothed values ─────────
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = 'base_link'

        quat_list = []
        for j in range(len(z_vals)):
            normal = normals[j]
            probe_z = -normal
            probe_x = scan_dir.copy()
            probe_x -= probe_x.dot(probe_z) * probe_z
            px_norm = np.linalg.norm(probe_x)
            if px_norm < 1e-6:
                probe_x = lateral_dir.copy()
                probe_x -= probe_x.dot(probe_z) * probe_z
                px_norm = np.linalg.norm(probe_x)
            probe_x /= px_norm
            probe_y = np.cross(probe_z, probe_x)

            R = np.column_stack([probe_x, probe_y, probe_z])
            quat = tq.mat2quat(R)  # (w, x, y, z)
            quat_list.append(quat)

        # Final smoothing on the quaternion sequence (also handles the
        # rare hemisphere-flip pathology).
        quat_arr = _quat_slerp_smooth(quat_list, window=11)

        # Diagnostic: max inter-waypoint orientation jump (post-smooth).
        ang_between = _angle_between_quats_arr(quat_arr)

        for j in range(len(z_vals)):
            quat = quat_arr[j]
            pose = Pose()
            pose.position = Point(x=positions_xy[j][0],
                                  y=positions_xy[j][1],
                                  z=z_vals[j])
            pose.orientation = Quaternion(
                x=quat[1], y=quat[2], z=quat[3], w=quat[0])
            pose_array.poses.append(pose)

        rospy.loginfo(
            "Surface-fitted path: %d waypoints, scan=%.0fmm, "
            "Z range=[%.3f, %.3f] (raw std=%.1fmm, smoothed std=%.1fmm), "
            "max Δorient=%.2f°, mean Δorient=%.2f°",
            len(pose_array.poses), total_length * 1000,
            float(z_vals.min()), float(z_vals.max()),
            float(np.std(z_vals_raw) * 1000), z_std_mm,
            float(ang_between.max()) if len(ang_between) else 0.0,
            float(ang_between.mean()) if len(ang_between) else 0.0)
        return pose_array

    def _plan_scan_two_markers(self):
        """Fallback: linear path centered between markers, ±scan_half_length."""
        scan_start, scan_end, center, scan_dir = self._compute_scan_region()

        total_length = 2.0 * self.scan_half_length
        n_points = max(2, int(total_length / self.waypoint_step))

        rospy.loginfo("Linear scan path: center=(%.3f,%.3f,%.3f), "
                      "length=%.0fmm, %d waypoints",
                      center[0], center[1], center[2],
                      total_length * 1000, n_points)

        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = 'base_link'

        for j in range(n_points + 1):
            t = j / float(n_points)
            pos = scan_start + t * (scan_end - scan_start)

            pose = Pose()
            pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
            pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
            pose_array.poses.append(pose)

        return pose_array

    def _crop_pointcloud(self, pcd, marker_points):
        """Crop point cloud to a generous box around markers using numpy."""
        center = marker_points.mean(axis=0)
        margin = self.crop_margin
        # Separate Z extent (height range around markers). Phantoms can
        # be up to ~150mm tall sitting on a table, while markers are at
        # table level — need enough Z range to capture the full height.
        z_extent_min = rospy.get_param(
            '/surface_fitting/crop_z_extent_min', 0.30)

        extent = marker_points.ptp(axis=0)
        extent[0] = max(extent[0], 0.05) + 2 * margin
        extent[1] = max(extent[1], 0.05) + 2 * margin
        extent[2] = max(extent[2], z_extent_min)

        aabb_min = center - extent / 2
        aabb_max = center + extent / 2

        pts = np.asarray(pcd.points)
        rospy.loginfo("Crop: min=(%.3f,%.3f,%.3f) max=(%.3f,%.3f,%.3f), cloud=%d pts",
                      aabb_min[0], aabb_min[1], aabb_min[2],
                      aabb_max[0], aabb_max[1], aabb_max[2], len(pts))

        # Debug: print sample points and per-axis match counts
        if len(pts) > 0:
            for idx in [0, len(pts)//4, len(pts)//2, 3*len(pts)//4, len(pts)-1]:
                rospy.loginfo("  Sample pt[%d]: (%.3f, %.3f, %.3f)", idx,
                              pts[idx, 0], pts[idx, 1], pts[idx, 2])
            x_ok = np.sum((pts[:, 0] >= aabb_min[0]) & (pts[:, 0] <= aabb_max[0]))
            y_ok = np.sum((pts[:, 1] >= aabb_min[1]) & (pts[:, 1] <= aabb_max[1]))
            z_ok = np.sum((pts[:, 2] >= aabb_min[2]) & (pts[:, 2] <= aabb_max[2]))
            rospy.loginfo("  Per-axis match: X=%d, Y=%d, Z=%d", x_ok, y_ok, z_ok)

        mask = ((pts[:, 0] >= aabb_min[0]) & (pts[:, 0] <= aabb_max[0]) &
                (pts[:, 1] >= aabb_min[1]) & (pts[:, 1] <= aabb_max[1]) &
                (pts[:, 2] >= aabb_min[2]) & (pts[:, 2] <= aabb_max[2]))

        cropped_pts = pts[mask]

        # CRITICAL diagnostic: Z distribution of CROPPED points.
        # If max Z here is much LOWER than what you see physically as
        # the phantom top, the crop bbox is excluding the phantom.
        # Solution: increase /surface_fitting/crop_margin so the bbox
        # extends further from the marker line (markers may be placed
        # BESIDE the phantom on the table, not on the phantom itself).
        if len(cropped_pts) > 0:
            cz = cropped_pts[:, 2]
            # Z histogram (buckets 1cm)
            z_min_c, z_max_c = float(cz.min()), float(cz.max())
            if z_max_c - z_min_c > 0.001:
                buckets = np.arange(np.floor(z_min_c * 100) / 100,
                                    np.ceil(z_max_c * 100) / 100 + 0.011,
                                    0.01)
                hist, edges = np.histogram(cz, bins=buckets)
                hist_str = " ".join(
                    "[%+.2f:%d]" % (edges[i], hist[i]) for i in range(len(hist))
                    if hist[i] > 0)
                rospy.loginfo("  Cropped Z distribution (1cm bins): %s",
                              hist_str)
            rospy.loginfo("  Cropped Z range: [%.4f, %.4f] m. "
                          "If real phantom top is HIGHER than %.4f m, "
                          "the crop bbox is excluding the phantom — "
                          "increase /surface_fitting/crop_margin!",
                          z_min_c, z_max_c, z_max_c)

        cropped = o3d.geometry.PointCloud()
        cropped.points = o3d.utility.Vector3dVector(cropped_pts)
        return cropped

    def _filter_pointcloud(self, pcd):
        """Denoise and downsample the point cloud.

        IMPORTANT: This now does a TOP-LAYER extraction step BEFORE the
        statistical outlier filter. Without this, when the markers are
        placed BELOW the actual scan surface (e.g. on the table next to
        the phantom), the table points outnumber the phantom points and
        the statistical outlier filter mistakes the (sparse) phantom-top
        points for outliers and deletes them. The result is an RBF fit
        on the table, not on the phantom — and a path that's tens of
        millimeters below the true surface.

        Strategy:
          1. Voxel downsample (as before).
          2. Build a coarse Z histogram. Find the highest non-empty bin,
             call its Z value Z_top.
          3. Keep only points whose Z >= Z_top - keep_thickness.
             (keep_thickness ≈ 30 mm covers a typical phantom curvature.)
          4. Then run the statistical outlier filter on this top layer.

        Disable by setting /surface_fitting/extract_top_layer=false.
        """
        pcd_down = pcd.voxel_down_sample(self.voxel_size)

        # Step 2-3: Top-layer extraction
        if rospy.get_param('/surface_fitting/extract_top_layer', True) \
                and len(pcd_down.points) > 0:
            pts = np.asarray(pcd_down.points)
            keep_thickness = rospy.get_param(
                '/surface_fitting/top_layer_thickness_mm', 30.0) / 1000.0
            min_gap_mm = rospy.get_param(
                '/surface_fitting/top_layer_min_gap_mm', 3.0)
            min_gap = min_gap_mm / 1000.0
            keep_percent = rospy.get_param(
                '/surface_fitting/top_layer_keep_percent', 30.0)
            min_points = rospy.get_param(
                '/surface_fitting/top_layer_min_points', 8)

            z = pts[:, 2]
            n = len(z)
            z_min, z_max = float(z.min()), float(z.max())
            z_sorted = np.sort(z)

            method = None
            z_threshold = None

            # Strategy 1: GAP-DETECTION (works when phantom and
            # background are well-separated in Z).
            if n >= 4:
                gaps = np.diff(z_sorted)
                gi = int(np.argmax(gaps))
                largest_gap = float(gaps[gi])
                if largest_gap >= min_gap:
                    cand_threshold = float(z_sorted[gi + 1])
                    cand_count = int((z >= cand_threshold).sum())
                    if cand_count >= min_points:
                        z_threshold = cand_threshold
                        method = ("gap-detection (largest_gap=%.1fmm @ "
                                  "Z=%.4f, %d pts above)"
                                  % (largest_gap * 1000, cand_threshold,
                                     cand_count))

            # Strategy 2: PERCENTILE-based (always keeps top K% of Z values)
            if z_threshold is None:
                top_n = max(min_points, int(n * keep_percent / 100.0))
                top_n = min(top_n, n)
                # z_sorted[-top_n] = the (top_n)-th largest Z
                z_threshold = float(z_sorted[-top_n])
                method = ("top-%d%% (kept top %d of %d by Z, "
                          "z_threshold=%.4f)"
                          % (int(keep_percent), top_n, n, z_threshold))

            keep_mask = z >= z_threshold
            pts_kept = pts[keep_mask]

            rospy.loginfo(
                "Top-layer extraction: Z range [%.4f, %.4f]m (n=%d) → "
                "%s → %d points kept (%.0f%% removed)",
                z_min, z_max, n, method,
                len(pts_kept),
                100.0 * (1.0 - len(pts_kept) / n))
            pcd_top = o3d.geometry.PointCloud()
            pcd_top.points = o3d.utility.Vector3dVector(pts_kept)
            pcd_down = pcd_top

        if len(pcd_down.points) > self.outlier_nb:
            cl, ind = pcd_down.remove_statistical_outlier(
                nb_neighbors=self.outlier_nb, std_ratio=self.outlier_std)
            pcd_clean = pcd_down.select_by_index(ind)
        else:
            pcd_clean = pcd_down

        return pcd_clean

    def _fit_surface(self, points):
        """Fit Z = f(X, Y) using RBF interpolation."""
        xy = points[:, :2]
        z = points[:, 2]

        n_samples = min(len(xy), 2000)
        if len(xy) > n_samples:
            idx = np.random.choice(len(xy), n_samples, replace=False)
            xy = xy[idx]
            z = z[idx]

        interpolator = RBFInterpolator(
            xy, z, kernel='thin_plate_spline', smoothing=1e-4)
        return interpolator

    def _compute_scan_axes(self, marker_points):
        """Compute scan/lateral directions and bounds from marker positions."""
        centered = marker_points - marker_points.mean(axis=0)
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        scan_dir = eigenvectors[:, 2].copy()
        lateral_dir = eigenvectors[:, 1].copy()

        if scan_dir[0] < 0:
            scan_dir = -scan_dir
        lateral_dir = np.cross(np.array([0, 0, 1]), scan_dir)
        if np.linalg.norm(lateral_dir) < 1e-6:
            lateral_dir = np.cross(scan_dir, np.array([0, 1, 0]))
        lateral_dir /= np.linalg.norm(lateral_dir)
        scan_dir_2d = scan_dir.copy()
        scan_dir_2d[2] = 0
        scan_dir_2d /= np.linalg.norm(scan_dir_2d)

        proj_scan = centered @ scan_dir_2d
        proj_lat = centered @ lateral_dir

        margin = self.crop_margin / 2
        origin = marker_points.mean(axis=0)

        bounds = {
            'scan_min': proj_scan.min() - margin,
            'scan_max': proj_scan.max() + margin,
            'lat_min': proj_lat.min() - margin,
            'lat_max': proj_lat.max() + margin,
            'origin': origin,
        }

        return scan_dir_2d, lateral_dir, bounds

    def _generate_path(self, interpolator, points, scan_dir, lateral_dir, bounds):
        """Generate a single center-line scan path on the fitted curved surface."""
        origin = bounds['origin']
        scan_length = bounds['scan_max'] - bounds['scan_min']
        n_points = max(2, int(scan_length / self.waypoint_step))
        eps = self.waypoint_step * 0.1

        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = 'base_link'

        lat_center = (bounds['lat_min'] + bounds['lat_max']) / 2.0

        for j in range(n_points):
            scan_offset = bounds['scan_min'] + j * self.waypoint_step
            pos_xy = origin[:2] + scan_offset * scan_dir[:2] + lat_center * lateral_dir[:2]
            z = float(interpolator(pos_xy.reshape(1, -1)))

            z_dx = float(interpolator((pos_xy + eps * scan_dir[:2]).reshape(1, -1)))
            z_dy = float(interpolator((pos_xy + eps * lateral_dir[:2]).reshape(1, -1)))
            dzdx = (z_dx - z) / eps
            dzdy = (z_dy - z) / eps
            normal = np.array([-dzdx, -dzdy, 1.0])
            normal /= np.linalg.norm(normal)

            probe_z = -normal
            probe_x = scan_dir.copy()
            probe_x -= probe_x.dot(probe_z) * probe_z
            probe_x /= np.linalg.norm(probe_x)
            probe_y = np.cross(probe_z, probe_x)

            R = np.column_stack([probe_x, probe_y, probe_z])
            quat = tq.mat2quat(R)

            pose = Pose()
            pose.position = Point(x=pos_xy[0], y=pos_xy[1], z=z)
            pose.orientation = Quaternion(
                x=quat[1], y=quat[2], z=quat[3], w=quat[0])
            pose_array.poses.append(pose)

        rospy.loginfo("Generated center-line scan path: %d waypoints",
                      len(pose_array.poses))
        return pose_array

    def _publish_surface_cloud(self, pcd):
        """Publish the processed surface cloud for RViz visualization."""
        points = np.asarray(pcd.points).astype(np.float32)
        n = len(points)

        msg = PointCloud2()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'base_link'
        msg.height = 1
        msg.width = n
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * n
        msg.is_dense = True
        msg.data = points.tobytes()

        self.pub_surface.publish(msg)

    @staticmethod
    def _pointcloud2_to_o3d(cloud_msg):
        """Convert sensor_msgs/PointCloud2 to Open3D PointCloud."""
        field_names = [f.name for f in cloud_msg.fields]
        field_offsets = {f.name: f.offset for f in cloud_msg.fields}

        has_xyz = all(n in field_names for n in ('x', 'y', 'z'))
        if not has_xyz:
            raise RuntimeError("Point cloud missing x/y/z fields")

        ox, oy, oz = field_offsets['x'], field_offsets['y'], field_offsets['z']
        point_step = cloud_msg.point_step
        data = cloud_msg.data

        n_points = cloud_msg.width * cloud_msg.height
        points = []
        for i in range(n_points):
            base = i * point_step
            x = struct.unpack_from('f', data, base + ox)[0]
            y = struct.unpack_from('f', data, base + oy)[0]
            z = struct.unpack_from('f', data, base + oz)[0]
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                points.append([x, y, z])

        pcd = o3d.geometry.PointCloud()
        if points:
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
        return pcd

    @staticmethod
    def _tf_to_matrix(transform):
        t = transform.translation
        q = transform.rotation
        T = np.eye(4)
        T[:3, :3] = tq.quat2mat([q.w, q.x, q.y, q.z])
        T[:3, 3] = [t.x, t.y, t.z]
        return T


if __name__ == '__main__':
    try:
        node = ScanPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
