#!/usr/bin/env python3
"""VLM anchor node — turn a VLM bbox into virtual scan-region markers.

Replaces the physical ArUco-based scan-region anchoring. Pipeline:

    /us3d/phantom_info  (PhantomInfo)
            │            └── bbox_image_xyxy  (color image pixels,
            │                                  already de-scaled to
            │                                  the original camera
            │                                  resolution by the
            │                                  recognizer node)
            │
            ▼
    pick two endpoints along the LONG axis of the bbox
            │
            ▼
    look up depth at each endpoint pixel (median of a small patch
    to be robust to noise / holes)
            │
            ▼
    unproject (u, v, d)  →  3D point in camera_color_optical_frame
            │
            ▼
    TF transform to base_link
            │
            ▼
    publish PoseArray on /us3d/markers (latched, 2 poses ordered
    so that the LONGER axis of the bbox becomes the scan_dir)

This drops the requirement to physically place ArUco fiducials on
the phantom: the VLM bounding box plus the depth camera give us
enough to define a scan region. The downstream curved-surface
scan_planner is unchanged — it still reads /us3d/markers and
selects the first two poses as the scan endpoints.

ASSUMPTION: depth and color images are pixel-aligned. That is the
default for hardware-D2C-aligned RGBD streams. If your driver
publishes raw, unaligned depth (the default for OrbbecSDK_ROS1
gemini2 launch is `depth_registration:=false`), enable it via:

    roslaunch us3d_bringup camera.launch    # if you patch it to
                                            # pass depth_registration=true

…or accept ~1-3 cm error in the anchor placement; the touchdown
+ force-adaptive scan tolerate that much.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import numpy as np
import rospy
import tf2_ros
import transforms3d.quaternions as tq
from cv_bridge import CvBridge
from geometry_msgs.msg import (Point, Pose, PoseArray, Quaternion)
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger, TriggerResponse

from us3d_msgs.msg import PhantomInfo


def _patch_median_depth(depth: np.ndarray,
                        u: int, v: int,
                        patch: int = 5) -> Optional[float]:
    """Median depth (in metres) over a (patch x patch) window.

    `depth` is a 2D ndarray of either 16U (millimetres) or 32F
    (metres). Returns None if no finite, non-zero sample lands in
    the patch.
    """
    h, w = depth.shape[:2]
    half = max(1, patch // 2)
    u0 = max(0, u - half)
    v0 = max(0, v - half)
    u1 = min(w, u + half + 1)
    v1 = min(h, v + half + 1)
    if u1 <= u0 or v1 <= v0:
        return None

    win = depth[v0:v1, u0:u1].astype(np.float64)
    if depth.dtype == np.uint16 or depth.dtype == np.int16:
        win = win / 1000.0  # mm → m
    win = win[np.isfinite(win) & (win > 0.05) & (win < 5.0)]
    if win.size == 0:
        return None
    return float(np.median(win))


class VlmAnchorNode:
    def __init__(self):
        rospy.init_node('vlm_anchor_node')

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Latest cached state.
        self._lock = threading.Lock()
        self._phantom: Optional[PhantomInfo] = None
        self._depth: Optional[np.ndarray] = None
        self._depth_stamp = rospy.Time(0)
        self._depth_frame = ''
        self._cam_info: Optional[CameraInfo] = None

        # Topics / params. Read from /vlm/vlm_anchor (loaded by
        # vlm.yaml) with private (~) overrides on top.
        cfg = rospy.get_param('/vlm/vlm_anchor', {}) or {}
        self.color_frame = rospy.get_param(
            '~color_frame', cfg.get('color_frame',
                                    'camera_color_optical_frame'))
        self.base_frame = rospy.get_param(
            '~base_frame', cfg.get('base_frame', 'base_link'))
        self.shrink_inset = float(rospy.get_param(
            '~bbox_inset_fraction',
            cfg.get('bbox_inset_fraction', 0.10)))
        self.depth_patch = int(rospy.get_param(
            '~depth_patch_px', cfg.get('depth_patch_px', 7)))
        self.publish_topic = rospy.get_param(
            '~markers_topic',
            cfg.get('markers_topic', '/us3d/markers'))

        rospy.Subscriber('/us3d/phantom_info', PhantomInfo,
                         self._phantom_cb, queue_size=1)
        rospy.Subscriber('/camera/depth/image_raw', Image,
                         self._depth_cb, queue_size=1)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo,
                         self._cam_info_cb, queue_size=1)

        self.pub_markers = rospy.Publisher(
            self.publish_topic, PoseArray, queue_size=1, latch=True)
        self.pub_debug = rospy.Publisher(
            '/us3d/vlm_anchor_debug', Image, queue_size=1, latch=True)

        rospy.Service('/us3d/anchor_from_vlm', Trigger,
                      self._anchor_srv)

        rospy.loginfo(
            "vlm_anchor ready (markers→%s, color_frame=%s, "
            "base_frame=%s, inset=%.2f, depth_patch=%dpx)",
            self.publish_topic, self.color_frame,
            self.base_frame, self.shrink_inset, self.depth_patch)

    # ---- Subscribers ------------------------------------------------

    def _phantom_cb(self, msg: PhantomInfo) -> None:
        with self._lock:
            self._phantom = msg

    def _depth_cb(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')
        except Exception as e:
            rospy.logwarn_throttle(5.0,
                                   "depth conversion failed: %s", e)
            return
        with self._lock:
            self._depth = depth
            self._depth_stamp = msg.header.stamp or rospy.Time.now()
            self._depth_frame = msg.header.frame_id or ''

    def _cam_info_cb(self, msg: CameraInfo) -> None:
        with self._lock:
            self._cam_info = msg

    # ---- Service ----------------------------------------------------

    def _anchor_srv(self, _req):
        with self._lock:
            phantom = self._phantom
            depth = None if self._depth is None else self._depth.copy()
            cam_info = self._cam_info
            depth_frame = self._depth_frame

        if phantom is None:
            return TriggerResponse(
                success=False,
                message="No /us3d/phantom_info received yet; "
                        "call /us3d/recognize_phantom or "
                        "/us3d/plan_from_instruction first.")
        has_endpoints = (len(getattr(phantom, 'endpoints_xyxy_norm',
                                     []) or []) == 4)
        has_bbox = (len(phantom.bbox_image_xyxy or []) == 4)
        if not (has_endpoints or has_bbox):
            return TriggerResponse(
                success=False,
                message="PhantomInfo has neither endpoints_xyxy_norm "
                        "nor bbox_image_xyxy. VLM did not localise the "
                        "phantom; check the raw_json field for hints.")
        if cam_info is None:
            return TriggerResponse(
                success=False,
                message="No /camera/color/camera_info yet.")
        if depth is None:
            return TriggerResponse(
                success=False,
                message="No /camera/depth/image_raw yet.")

        try:
            ok, msg, pose_array = self._compute(
                phantom, depth, cam_info, depth_frame)
        except Exception as e:
            rospy.logerr("vlm_anchor failed: %s", e)
            return TriggerResponse(
                success=False, message="exception: %s" % e)

        if ok and pose_array is not None:
            self.pub_markers.publish(pose_array)
        return TriggerResponse(success=ok, message=msg)

    # ---- Core -------------------------------------------------------

    def _compute(self,
                 phantom: PhantomInfo,
                 depth: np.ndarray,
                 cam_info: CameraInfo,
                 depth_frame: str) -> Tuple[bool, str,
                                            Optional[PoseArray]]:
        # 1) Pick two color-image-pixel endpoints (u0, v0) and
        #    (u1, v1). Priority: explicit `endpoints_xyxy_norm`
        #    chosen by the instruction-aware VLM; fall back to the
        #    long edge of `bbox_image_xyxy` (legacy behaviour).
        cw = int(cam_info.width)
        ch = int(cam_info.height)
        eps = list(getattr(phantom, 'endpoints_xyxy_norm', []) or [])
        if len(eps) == 4:
            u0, v0, u1, v1, source = self._endpoints_from_norm(
                eps, cw, ch, self.shrink_inset)
        else:
            u0, v0, u1, v1, source = self._endpoints_from_bbox(
                phantom.bbox_image_xyxy, self.shrink_inset)

        # 2) Sanity check vs depth image size. Color intrinsics
        #    K are at color resolution (cw, ch). If depth resolution
        #    differs, we re-scale color pixels to depth pixels for
        #    the depth lookup, then back to color for unprojection.
        h, w = depth.shape[:2]
        if (cw, ch) != (w, h):
            sx = w / float(cw)
            sy = h / float(ch)
            rospy.logwarn(
                "depth (%dx%d) and color (%dx%d) sizes differ; "
                "scaling endpoints by (%.3f, %.3f). Enable "
                "depth_registration in the camera launch for "
                "exact alignment.", w, h, cw, ch, sx, sy)
            ud0, vd0 = u0 * sx, v0 * sy
            ud1, vd1 = u1 * sx, v1 * sy
        else:
            ud0, vd0 = u0, v0
            ud1, vd1 = u1, v1

        u0i = int(round(np.clip(ud0, 0, w - 1)))
        v0i = int(round(np.clip(vd0, 0, h - 1)))
        u1i = int(round(np.clip(ud1, 0, w - 1)))
        v1i = int(round(np.clip(vd1, 0, h - 1)))

        d0 = _patch_median_depth(depth, u0i, v0i, self.depth_patch)
        d1 = _patch_median_depth(depth, u1i, v1i, self.depth_patch)
        if d0 is None or d1 is None:
            return False, ("could not read depth at endpoints "
                           "(u,v)=(%d,%d) or (%d,%d) — depth holes "
                           "or endpoints outside FOV (source=%s)"
                           % (u0i, v0i, u1i, v1i, source)), None

        # 3) Unproject using the COLOR camera intrinsics.
        K = np.array(cam_info.K).reshape(3, 3)
        fx = float(K[0, 0]); fy = float(K[1, 1])
        cx = float(K[0, 2]); cy = float(K[1, 2])
        if fx < 1e-3 or fy < 1e-3:
            return False, "camera_info K has zero focal length", None

        def _unproject(u: float, v: float, d: float) -> np.ndarray:
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            Z = d
            return np.array([X, Y, Z], dtype=np.float64)

        p0_cam = _unproject(u0, v0, d0)
        p1_cam = _unproject(u1, v1, d1)

        # 4) TF camera_color_optical_frame → base_link.
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.base_frame, self.color_frame,
                rospy.Time(0), rospy.Duration(2.0))
        except (tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as e:
            return False, ("TF %s→%s lookup failed: %s"
                           % (self.color_frame, self.base_frame, e)), None

        T = self._tf_to_matrix(tf_stamped.transform)
        p0_base = (T @ np.r_[p0_cam, 1.0])[:3]
        p1_base = (T @ np.r_[p1_cam, 1.0])[:3]

        # 5) Build PoseArray on base_link. Two poses; orientation
        #    is identity — the scan_planner only uses positions.
        msg = PoseArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.base_frame
        for p in (p0_base, p1_base):
            pose = Pose()
            pose.position = Point(x=float(p[0]), y=float(p[1]),
                                  z=float(p[2]))
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            msg.poses.append(pose)

        dist_mm = float(np.linalg.norm(p1_base - p0_base) * 1000)
        text = ("anchor[%s]: pixel (%d,%d)->(%d,%d) → 3D distance "
                "%.1fmm | p0=(%.3f,%.3f,%.3f) p1=(%.3f,%.3f,%.3f) "
                "(depths %.3f / %.3f m)"
                % (source, u0i, v0i, u1i, v1i, dist_mm,
                   p0_base[0], p0_base[1], p0_base[2],
                   p1_base[0], p1_base[1], p1_base[2],
                   d0, d1))
        rospy.loginfo(text)

        # 6) Optional debug overlay.
        self._publish_debug(phantom, depth, u0i, v0i, u1i, v1i)
        return True, text, msg

    @staticmethod
    def _endpoints_from_norm(eps_norm: List[float],
                             color_w: int,
                             color_h: int,
                             inset_frac: float
                             ) -> Tuple[float, float,
                                        float, float, str]:
        """Convert 4 normalised values [u0_n, v0_n, u1_n, v1_n] to
        color-pixel coords; pull the two points slightly inwards
        toward each other to dodge depth-edge holes.
        """
        u0_n = float(np.clip(eps_norm[0], 0.0, 1.0))
        v0_n = float(np.clip(eps_norm[1], 0.0, 1.0))
        u1_n = float(np.clip(eps_norm[2], 0.0, 1.0))
        v1_n = float(np.clip(eps_norm[3], 0.0, 1.0))
        u0 = u0_n * (color_w - 1)
        v0 = v0_n * (color_h - 1)
        u1 = u1_n * (color_w - 1)
        v1 = v1_n * (color_h - 1)

        # Inset both points toward the segment midpoint by
        # `inset_frac` of the segment length. Same effect as the
        # bbox-edge inset used in the legacy path.
        mid_u = 0.5 * (u0 + u1)
        mid_v = 0.5 * (v0 + v1)
        u0 = u0 + inset_frac * (mid_u - u0)
        v0 = v0 + inset_frac * (mid_v - v0)
        u1 = u1 + inset_frac * (mid_u - u1)
        v1 = v1 + inset_frac * (mid_v - v1)
        return u0, v0, u1, v1, "endpoints_norm"

    @staticmethod
    def _endpoints_from_bbox(bbox_pix: List[float],
                             inset_frac: float
                             ) -> Tuple[float, float,
                                        float, float, str]:
        """Long-axis endpoints of a pixel-coord bbox (legacy)."""
        x_min, y_min, x_max, y_max = (float(bbox_pix[0]),
                                      float(bbox_pix[1]),
                                      float(bbox_pix[2]),
                                      float(bbox_pix[3]))
        dx = max(1.0, x_max - x_min)
        dy = max(1.0, y_max - y_min)
        if dx >= dy:
            inset_px = inset_frac * dx
            u0 = x_min + inset_px
            u1 = x_max - inset_px
            v0 = (y_min + y_max) / 2.0
            v1 = v0
            src = "bbox_long_edge"
        else:
            inset_px = inset_frac * dy
            u0 = (x_min + x_max) / 2.0
            u1 = u0
            v0 = y_min + inset_px
            v1 = y_max - inset_px
            src = "bbox_long_edge"
        return u0, v0, u1, v1, src

    def _publish_debug(self,
                       phantom: PhantomInfo,
                       depth: np.ndarray,
                       u0: int, v0: int,
                       u1: int, v1: int) -> None:
        try:
            import cv2
            d = depth.astype(np.float32)
            if depth.dtype in (np.uint16, np.int16):
                d = d / 1000.0
            d = np.clip(d, 0.05, 2.0)
            vis = ((d - d.min()) / max(1e-3, d.max() - d.min())
                   * 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            # Optional bbox overlay (debug only).
            if (phantom.bbox_image_xyxy
                    and len(phantom.bbox_image_xyxy) == 4):
                bx0 = int(round(phantom.bbox_image_xyxy[0]))
                by0 = int(round(phantom.bbox_image_xyxy[1]))
                bx1 = int(round(phantom.bbox_image_xyxy[2]))
                by1 = int(round(phantom.bbox_image_xyxy[3]))
                cv2.rectangle(vis, (bx0, by0), (bx1, by1),
                              (0, 255, 0), 2)
            cv2.circle(vis, (u0, v0), 6, (255, 255, 255), -1)
            cv2.circle(vis, (u1, v1), 6, (255, 255, 255), -1)
            cv2.line(vis, (u0, v0), (u1, v1), (0, 0, 255), 2)
            self.pub_debug.publish(
                self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
        except Exception as e:
            rospy.logdebug("debug overlay failed: %s", e)

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
        node = VlmAnchorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
