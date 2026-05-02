#!/usr/bin/env python3
"""Instruction planner node.

A SINGLE multimodal VLM call turns a natural-language scan
instruction (any language, but typically Chinese for this project)
plus the latest color frame into a complete scan plan, including
the two image endpoints that the user's instruction implies.

Service
-------
/us3d/plan_from_instruction (us3d_msgs/PlanFromInstruction):
    request:  string instruction, bool dry_run
    response: bool success, string message, string plan_json

The LLM is *not* allowed to emit robot waypoints. It produces a
JSON payload like:

    {
      "phantom_type":      "knee",
      "scan_axis":         "long",          // "long" | "short" | "auto" | "free"
      "scan_length_mm":    80.0,
      "scan_speed_mms":    3.0,
      "reverse_direction": false,
      "endpoints_norm":    [[x0, y0], [x1, y1]],   // ★ normalised [0,1]
      "bbox_norm":         [x_min, y_min, x_max, y_max],   // optional
      "use_marker_pair":   [0, 1],
      "notes":             "..."
    }

The endpoints are the two image points whose 3D un-projection
defines the scan's start and end (after vlm_anchor reprojects
them via the depth camera). The model is asked to pick endpoints
along the axis the operator's instruction asked for — long axis,
short axis, or any free direction such as "from the upper-left
corner of the phantom to the lower-right corner".

Dispatch flow
-------------
1) VLM single call (vision + instruction)            -> plan JSON
2) clamp + validate + write /scan/* parameters
3) publish PhantomInfo (with endpoints_xyxy_norm)    -> /us3d/phantom_info
4) call /us3d/anchor_from_vlm                        -> /us3d/markers
5) call /us3d/plan_scan                              -> /us3d/scan_path

Whatever the LLM says, every numeric field is clamped to the
configured safety range before any ROS parameter is touched. The
existing curved-surface scan_planner remains the sole path that
emits robot waypoints.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from us3d_msgs.msg import PhantomInfo
from us3d_msgs.srv import (PlanFromInstruction,
                           PlanFromInstructionResponse)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_client import VLMClient  # noqa: E402


PLANNER_SYSTEM_PROMPT = """\
You convert a single ultrasound-scan instruction PLUS a top-down
RGB image of the phantom on the table into a strict JSON plan.
The robot is a UR5e with a linear ultrasound probe; downstream
software turns your JSON into actual waypoints, so you NEVER
output poses, joints, or trajectories.

Output ONLY the JSON object below — no prose, no markdown fence.

{
  "phantom_type":      "<label or 'unknown'>",
  "scan_axis":         "long" | "short" | "auto" | "free",
  "scan_length_mm":    <float in [10, 200]>,
  "scan_speed_mms":    <float in [1, 10]>,
  "reverse_direction": <bool>,
  "endpoints_norm":    [[x0, y0], [x1, y1]],
  "bbox_norm":         [x_min, y_min, x_max, y_max],
  "notes":             "<one short sentence in Chinese>"
}

Rules for the endpoints (CRITICAL):
- "endpoints_norm" is REQUIRED. Two image points in NORMALISED
  coordinates [0, 1] of the image you see (top-left origin,
  x to the right, y downward). Use 4 decimals.
- The line connecting the two points MUST align with the
  direction the user's instruction implies:
    * 长轴 / long axis / 沿长边        -> longest axis of the phantom
    * 短轴 / short axis / 沿短边 / 横向 -> shortest axis of the phantom
    * 对角 / diagonal                  -> a meaningful corner-to-corner line
    * 从X到Y / 从左到右 / from start to end -> the explicit direction the
                                              operator described
    * (no direction given)            -> the long axis by default
- The two points should sit roughly 5-15% inside the phantom
  body from its outer edge (so the depth camera doesn't read a
  hole at the boundary). DO NOT place them on the table or
  outside the phantom.
- If the operator says "反向 / 倒着 / from end to start", swap
  the order of the two endpoints AND set "reverse_direction": true.

Rules for the rest:
- If length is not given, choose by phantom_type
  (knee ~80, neck ~60, abdomen ~120, default 60). Always [10, 200].
- If speed is not given, return 3.0.
- "scan_axis" is a hint of which axis you used; "free" means
  the operator described an explicit direction.
- "bbox_norm" tightly encloses the phantom body, used for
  debugging only.
- "phantom_type" must be lowercase and from a known set when
  possible; otherwise "unknown".

Be conservative: prefer reasonable defaults over guessing
fictional anatomical landmarks.
"""


def _build_user_prompt(instruction: str,
                       image_w: int,
                       image_h: int) -> str:
    return (
        "操作员指令: %s\n\n"
        "图像尺寸 %dx%d (输入图大小)。请按 schema 直接给出 JSON。"
        % ((instruction or "").strip(), image_w, image_h))


class InstructionPlannerNode:
    def __init__(self):
        rospy.init_node('instruction_planner_node')

        self.params = self._load_params()
        self.client = VLMClient.from_param_dict(self.params)
        # Heartbeat: log "VLM still streaming, X reasoning, Y content
        # chunks after Zs" every progress_interval_s seconds while
        # waiting for a thinking model. Disabled by setting <= 0.
        progress_s = float(self.params.get('progress_interval_s', 5.0))
        if progress_s > 0:
            self.client.set_progress_callback(
                self._vlm_progress_log, interval_s=progress_s)
        self.bridge = CvBridge()

        self._lock = threading.Lock()
        self._latest_img: Optional[np.ndarray] = None
        self._latest_stamp = rospy.Time(0)

        image_topic = str(self.params.get(
            'image_topic', '/camera/color/image_raw'))
        rospy.Subscriber(image_topic, Image,
                         self._image_cb, queue_size=1)

        self._plan_scan_name = str(self.params.get(
            'plan_scan_service', '/us3d/plan_scan'))
        self._service_name = str(self.params.get(
            'service_name', '/us3d/plan_from_instruction'))
        self._anchor_service = str(self.params.get(
            'anchor_service', '/us3d/anchor_from_vlm'))
        self._phantom_topic = str(self.params.get(
            'phantom_topic', '/us3d/phantom_info'))

        self.pub_phantom = rospy.Publisher(
            self._phantom_topic, PhantomInfo, queue_size=1, latch=True)

        self.srv = rospy.Service(
            self._service_name, PlanFromInstruction, self._on_request)

        rospy.loginfo(
            "instruction_planner ready: model=%s base=%s service=%s "
            "image_topic=%s downstream=%s anchor=%s",
            self.params.get('model'),
            self.params.get('base_url'),
            self._service_name,
            image_topic,
            self._plan_scan_name,
            self._anchor_service)

    # ---- Param loading ---------------------------------------------

    def _load_params(self) -> Dict[str, Any]:
        glob = rospy.get_param('/vlm', {}) or {}
        flat: Dict[str, Any] = {}
        for k in ('base_url', 'api_key', 'model', 'text_model',
                  'timeout_s', 'max_retries', 'temperature',
                  'top_p', 'max_tokens', 'use_response_format',
                  'stream'):
            if k in glob:
                flat[k] = glob[k]
        for k, v in (glob.get('instruction_planner') or {}).items():
            flat[k] = v
        priv = rospy.get_param('~', {}) or {}
        for k, v in priv.items():
            flat[k] = v
        return flat

    # ---- Heartbeat -------------------------------------------------

    @staticmethod
    def _vlm_progress_log(elapsed_s: float,
                          n_content_chunks: int,
                          n_reasoning_chunks: int) -> None:
        rospy.loginfo(
            "VLM streaming... %.1fs elapsed, reasoning=%d chunks, "
            "content=%d chunks (still working, do NOT Ctrl-C)",
            elapsed_s, n_reasoning_chunks, n_content_chunks)

    # ---- Subscribers -----------------------------------------------

    def _image_cb(self, msg: Image) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn_throttle(5.0,
                                   "image conversion failed: %s", e)
            return
        with self._lock:
            self._latest_img = img
            self._latest_stamp = msg.header.stamp or rospy.Time.now()

    # ---- Service ---------------------------------------------------

    def _on_request(self, req):
        instruction = (req.instruction or "").strip()
        if not instruction:
            return PlanFromInstructionResponse(
                success=False,
                message="instruction is empty",
                plan_json="")

        with self._lock:
            img = (None if self._latest_img is None
                   else self._latest_img.copy())
            stamp = self._latest_stamp
        if img is None:
            return PlanFromInstructionResponse(
                success=False,
                message=("No image yet on %s. Make sure the camera "
                         "is publishing." % self.params.get(
                             'image_topic', '/camera/color/image_raw')),
                plan_json="")

        h_orig, w_orig = img.shape[:2]
        max_side_px = int(self.params.get('image_max_side_px', 512))
        jpeg_q = int(self.params.get('image_jpeg_quality', 75))

        user_prompt = _build_user_prompt(instruction, w_orig, h_orig)

        t0 = time.time()
        result = self.client.chat_with_image(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_bgr=img,
            jpeg_quality=jpeg_q,
            max_side_px=max_side_px,
            want_json=True,
        )
        dt = time.time() - t0

        if not result.ok:
            rospy.logwarn(
                "instruction VLM call failed (%.2fs, HTTP %d, "
                "model=%s): %s | body: %s",
                dt, result.http_status, result.request_model,
                result.error,
                (result.raw_response or '')[:300] or '<empty>')
            return PlanFromInstructionResponse(
                success=False,
                message="VLM call failed: %s" % result.error,
                plan_json="")

        parsed = result.parsed or {}
        if not isinstance(parsed, dict) or not parsed:
            preview = (result.content or '').replace('\n', ' ')[:400]
            rospy.logwarn(
                "instruction LLM did not return parseable JSON "
                "(%.2fs, %d bytes). Raw text: %s",
                dt, len(result.content or ''), preview or '<empty>')
            return PlanFromInstructionResponse(
                success=False,
                message="LLM did not return parseable JSON. raw=%s"
                        % preview,
                plan_json=result.content)

        plan = self._validate_plan(parsed, w_orig, h_orig)
        plan_json = json.dumps(plan, ensure_ascii=True)
        rospy.loginfo(
            "planner: instruction=%r (%.2fs) -> plan=%s",
            instruction, dt, plan_json)

        # Publish PhantomInfo so vlm_anchor (and any other
        # subscriber) gets a fresh, instruction-aware bbox +
        # endpoints to work from.
        self._publish_phantom_info(plan, stamp, result, dt)

        if req.dry_run:
            return PlanFromInstructionResponse(
                success=True,
                message="dry_run: plan computed, not dispatched",
                plan_json=plan_json)

        ok, msg = self._dispatch(plan)
        return PlanFromInstructionResponse(
            success=ok, message=msg, plan_json=plan_json)

    # ---- Validation ------------------------------------------------

    def _validate_plan(self,
                       raw: Dict[str, Any],
                       image_w: int,
                       image_h: int) -> Dict[str, Any]:
        max_len = float(self.params.get('max_scan_length_mm', 200.0))
        min_len = float(self.params.get('min_scan_length_mm', 10.0))
        max_spd = float(self.params.get('max_scan_speed_mms', 10.0))
        min_spd = float(self.params.get('min_scan_speed_mms', 1.0))
        max_id = int(self.params.get('max_marker_id', 31))
        default_pair = list(self.params.get('default_marker_pair', [0, 1]))

        def _clip(v, lo, hi, default):
            try:
                f = float(v)
            except (TypeError, ValueError):
                return float(default)
            if not (f == f):  # NaN
                return float(default)
            return float(max(lo, min(hi, f)))

        plan: Dict[str, Any] = {}
        plan['phantom_type'] = str(raw.get('phantom_type') or 'unknown').lower()

        axis = str(raw.get('scan_axis') or 'auto').lower()
        if axis not in ('long', 'short', 'auto', 'free',
                        'transverse', 'sagittal'):
            axis = 'auto'
        plan['scan_axis'] = axis

        plan['scan_length_mm'] = _clip(
            raw.get('scan_length_mm'), min_len, max_len, 60.0)
        plan['scan_speed_mms'] = _clip(
            raw.get('scan_speed_mms'), min_spd, max_spd, 3.0)

        pair = raw.get('use_marker_pair') or default_pair
        if (isinstance(pair, (list, tuple)) and len(pair) == 2
                and all(isinstance(x, (int, float)) for x in pair)):
            a, b = int(pair[0]), int(pair[1])
            a = max(0, min(max_id, a))
            b = max(0, min(max_id, b))
            if a == b:
                a, b = default_pair[0], default_pair[1]
            plan['use_marker_pair'] = [a, b]
        else:
            plan['use_marker_pair'] = list(default_pair)

        plan['reverse_direction'] = bool(raw.get('reverse_direction', False))
        plan['notes'] = str(raw.get('notes') or '')[:512]

        # ---- Endpoints --------------------------------------------
        endpoints = self._parse_endpoints(raw)
        if endpoints is not None and plan['reverse_direction']:
            # Swap order so downstream sees user-requested direction.
            endpoints = [endpoints[2], endpoints[3],
                         endpoints[0], endpoints[1]]
        plan['endpoints_norm'] = endpoints if endpoints else []

        # ---- Bbox (optional, for debug overlay) -------------------
        bbox = self._parse_bbox_norm(raw)
        plan['bbox_norm'] = bbox if bbox else []

        return plan

    @staticmethod
    def _parse_endpoints(raw: Dict[str, Any]) -> Optional[List[float]]:
        """Accept either:
          endpoints_norm: [[x0, y0], [x1, y1]]
          endpoints_norm: [x0, y0, x1, y1]
          endpoints:      same as above
        Returns flat [x0, y0, x1, y1] in [0, 1] or None.
        """
        for key in ('endpoints_norm', 'endpoints',
                    'endpoints_xyxy_norm', 'endpoints_xyxy'):
            v = raw.get(key)
            if v is None:
                continue
            flat: List[float] = []
            if (isinstance(v, (list, tuple)) and len(v) == 2
                    and all(isinstance(p, (list, tuple))
                            and len(p) == 2 for p in v)):
                flat = [float(v[0][0]), float(v[0][1]),
                        float(v[1][0]), float(v[1][1])]
            elif (isinstance(v, (list, tuple)) and len(v) == 4
                    and all(isinstance(x, (int, float)) for x in v)):
                flat = [float(v[0]), float(v[1]),
                        float(v[2]), float(v[3])]
            else:
                continue

            # Accept percentages (0-100) too.
            if max(abs(x) for x in flat) > 1.5:
                flat = [x / 100.0 for x in flat]
            # Clamp to [0, 1].
            flat = [max(0.0, min(1.0, x)) for x in flat]
            # Reject degenerate (same point both ends).
            dx = flat[2] - flat[0]
            dy = flat[3] - flat[1]
            if (dx * dx + dy * dy) < 1e-4:
                continue
            return flat
        return None

    @staticmethod
    def _parse_bbox_norm(raw: Dict[str, Any]) -> Optional[List[float]]:
        for key in ('bbox_norm', 'bbox', 'bbox_image_xyxy'):
            v = raw.get(key)
            if (isinstance(v, (list, tuple)) and len(v) == 4
                    and all(isinstance(x, (int, float)) for x in v)):
                vals = [float(x) for x in v]
                if max(abs(x) for x in vals) > 1.5:
                    # Pixel coords in some imaginary size; skip
                    # (not useful without that size).
                    continue
                xs = sorted([vals[0], vals[2]])
                ys = sorted([vals[1], vals[3]])
                clamped = [max(0.0, min(1.0, xs[0])),
                           max(0.0, min(1.0, ys[0])),
                           max(0.0, min(1.0, xs[1])),
                           max(0.0, min(1.0, ys[1]))]
                if (clamped[2] - clamped[0]) < 1e-3 \
                        or (clamped[3] - clamped[1]) < 1e-3:
                    continue
                return clamped
        return None

    # ---- PhantomInfo + dispatch ------------------------------------

    def _publish_phantom_info(self,
                              plan: Dict[str, Any],
                              stamp: rospy.Time,
                              result,
                              elapsed_s: float) -> None:
        info = PhantomInfo()
        info.header.stamp = (stamp if stamp != rospy.Time(0)
                             else rospy.Time.now())
        info.header.frame_id = 'camera_color_optical_frame'
        info.phantom_type = str(plan.get('phantom_type') or 'unknown')
        info.confidence = 1.0   # planner doesn't surface a numeric score
        info.description = str(plan.get('notes') or '')
        axis_h = str(plan.get('scan_axis') or '')
        if axis_h == 'auto':
            axis_h = ''
        info.scan_axis_hint = axis_h
        info.scan_length_hint_mm = float(plan.get('scan_length_mm', 0.0))
        info.endpoints_xyxy_norm = list(plan.get('endpoints_norm') or [])

        # Convert bbox_norm (if any) to legacy pixel-space field
        # by reading current camera image dimensions on the side
        # holding the lock.
        bbox_norm = plan.get('bbox_norm') or []
        with self._lock:
            img = self._latest_img
        if (len(bbox_norm) == 4 and img is not None):
            h, w = img.shape[:2]
            info.bbox_image_xyxy = [
                float(bbox_norm[0] * w),
                float(bbox_norm[1] * h),
                float(bbox_norm[2] * w),
                float(bbox_norm[3] * h),
            ]
        else:
            info.bbox_image_xyxy = []

        info.raw_json = json.dumps({
            "plan": plan,
            "raw_text": (result.content or '')[:1500],
            "elapsed_s": round(elapsed_s, 3),
            "usage": result.usage,
        }, ensure_ascii=True)

        self.pub_phantom.publish(info)

    def _dispatch(self, plan: Dict[str, Any]) -> Tuple[bool, str]:
        if bool(self.params.get('apply_to_param_server', True)):
            half_len_m = (plan['scan_length_mm'] / 1000.0) / 2.0
            speed_m = plan['scan_speed_mms'] / 1000.0
            try:
                rospy.set_param('/scan/scan_half_length', float(half_len_m))
                rospy.set_param('/scan/scan_speed', float(speed_m))
                rospy.set_param('/scan/reverse_scan_direction',
                                bool(plan['reverse_direction']))
            except Exception as e:
                return False, "set_param failed: %s" % e

        # Call /us3d/anchor_from_vlm (default ON). It will pick up
        # the latched PhantomInfo we just published.
        anchor_msg = ""
        if bool(self.params.get('use_vlm_anchor', True)):
            try:
                rospy.wait_for_service(self._anchor_service, timeout=2.0)
                anchor = rospy.ServiceProxy(self._anchor_service, Trigger)
                aresp = anchor()
                anchor_msg = " | anchor: %s" % (aresp.message[:300])
                if not aresp.success:
                    rospy.logwarn(
                        "vlm anchor call returned failure: %s; falling "
                        "back to whatever /us3d/markers already has",
                        aresp.message)
            except rospy.ROSException:
                rospy.logdebug(
                    "%s not available; assuming /us3d/markers is "
                    "supplied elsewhere (e.g. ArUco detector)",
                    self._anchor_service)
            except rospy.ServiceException as e:
                rospy.logwarn("anchor service call raised: %s", e)

        try:
            rospy.wait_for_service(self._plan_scan_name, timeout=5.0)
        except rospy.ROSException:
            return False, ("downstream %s not available (is scan_planner "
                           "running?)" % self._plan_scan_name)

        try:
            call = rospy.ServiceProxy(self._plan_scan_name, Trigger)
            resp = call()
        except rospy.ServiceException as e:
            return False, "%s call raised: %s" % (self._plan_scan_name, e)

        if not resp.success:
            return False, "%s reported: %s%s" % (
                self._plan_scan_name, resp.message, anchor_msg)
        return True, "dispatched: %s | downstream: %s%s" % (
            self._summarize(plan), resp.message, anchor_msg)

    @staticmethod
    def _summarize(plan: Dict[str, Any]) -> str:
        ep = plan.get('endpoints_norm') or []
        ep_str = ("ep=[%.2f,%.2f]->[%.2f,%.2f]" % tuple(ep)
                  if len(ep) == 4 else "ep=none")
        return ("type=%s axis=%s len=%.0fmm speed=%.1fmm/s rev=%s %s"
                % (plan.get('phantom_type'),
                   plan.get('scan_axis'),
                   plan.get('scan_length_mm', 0.0),
                   plan.get('scan_speed_mms', 0.0),
                   plan.get('reverse_direction'),
                   ep_str))


if __name__ == '__main__':
    try:
        node = InstructionPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
