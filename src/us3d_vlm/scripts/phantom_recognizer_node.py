#!/usr/bin/env python3
"""Phantom recognition node powered by a Vision-Language Model.

Subscribes to the color camera topic, periodically sends the latest
frame to a vLLM server, and publishes a us3d_msgs/PhantomInfo with
the model's structured judgment.

Trigger modes
-------------
1. Periodic (default): every `rate_hz` seconds, send the freshest
   image. Useful while the operator is positioning the phantom.
2. Service: call /us3d/recognize_phantom (std_srvs/Trigger) to
   force a one-shot recognition NOW. Returns success once the
   resulting PhantomInfo has been published.

The node is intentionally tolerant: if the VLM server is down it
logs a throttled warning and keeps trying. PhantomInfo is published
with `confidence=0` and a descriptive `raw_json` field on failure.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse

from us3d_msgs.msg import PhantomInfo

# Allow `import vlm_client` whether the script runs from source
# (catkin_ws/src/us3d_vlm/scripts) or from install.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_client import VLMClient  # noqa: E402


PHANTOM_SYSTEM_PROMPT = """\
Identify the ultrasound phantom on the table from a top-down RGB
camera. Ignore any ArUco markers; focus on the phantom body.

Output ONLY this JSON (no prose, no markdown fence):

{"phantom_type":"<label>","confidence":<0-1>,"description":"<≤25 字中文>","scan_axis_hint":"<long|short|''>","scan_length_hint_mm":<mm>,"bbox":[x0,y0,x1,y1]}

Rules:
- phantom_type ∈ known labels list, else "unknown".
- "bbox" is REQUIRED when phantom_type != "unknown".
  ★ Coordinates are NORMALISED to [0, 1] of the image you see:
    x0/x1 along the horizontal axis (0=left, 1=right),
    y0/y1 along the vertical axis (0=top, 1=bottom).
    Use 4 decimals, e.g. [0.2350, 0.1804, 0.7234, 0.6125].
    DO NOT output pixel coordinates.
  Tightly enclose the phantom body. Long axis aligned with the
  phantom's principal anatomical / scan axis.
- Prefer "unknown" over guessing.
"""


def _build_user_prompt(known_labels: list) -> str:
    labels = ", ".join("'%s'" % s for s in known_labels)
    return (
        "Identify the ultrasound phantom in this image and propose a "
        "scan strategy. Known label set: " + labels + ". "
        "Reply with the JSON object only."
    )


class PhantomRecognizerNode:
    def __init__(self):
        rospy.init_node('phantom_recognizer_node')

        # Resolve config (private params override global /vlm/* tree)
        self.params = self._load_params()

        if not bool(self.params.get('enabled', True)):
            rospy.logwarn("phantom_recognizer disabled via params; "
                          "node will spin doing nothing.")

        self.client = VLMClient.from_param_dict(self.params)
        self.bridge = CvBridge()

        self._latest_img: Optional[np.ndarray] = None
        self._latest_stamp = rospy.Time(0)
        self._lock = threading.Lock()
        self._query_lock = threading.Lock()  # serialise VLM calls so
                                             #   periodic timer never
                                             #   stacks on top of an
                                             #   already-running query
        self._known_labels = list(self.params.get('known_labels', []))

        rate_hz = float(self.params.get('rate_hz', 0.0))
        # rate_hz <= 0 disables the periodic timer entirely; VLM is
        # then only invoked by the /us3d/recognize_phantom service.
        # This is the recommended default for slow / expensive
        # cloud thinking models.
        self._period = (1.0 / rate_hz) if rate_hz > 0 else 0.0

        image_topic = str(self.params.get(
            'image_topic', '/camera/color/image_raw'))
        publish_topic = str(self.params.get(
            'publish_topic', '/us3d/phantom_info'))

        self.pub_info = rospy.Publisher(
            publish_topic, PhantomInfo, queue_size=1, latch=True)
        rospy.Subscriber(image_topic, Image, self._image_cb, queue_size=1)

        self.srv_recognize = rospy.Service(
            '/us3d/recognize_phantom', Trigger, self._recognize_srv)

        rospy.loginfo(
            "phantom_recognizer ready: model=%s base=%s rate=%s "
            "image_topic=%s service=/us3d/recognize_phantom",
            self.params.get('model'),
            self.params.get('base_url'),
            ("%.2fHz" % rate_hz) if rate_hz > 0 else "OFF (service-only)",
            image_topic)

        self._last_query_t = 0.0
        if bool(self.params.get('enabled', True)) and self._period > 0:
            self._timer = rospy.Timer(
                rospy.Duration(self._period), self._timer_cb)

    # ---- Param loading ---------------------------------------------

    def _load_params(self) -> Dict[str, Any]:
        # Global config (loaded by vlm.yaml under /vlm/...).
        glob = rospy.get_param('/vlm', {}) or {}
        flat: Dict[str, Any] = {}
        # promote top-level connection params
        for k in ('base_url', 'api_key', 'model', 'text_model',
                  'timeout_s', 'max_retries', 'temperature',
                  'top_p', 'max_tokens'):
            if k in glob:
                flat[k] = glob[k]
        # node-specific
        for k, v in (glob.get('phantom_recognizer') or {}).items():
            flat[k] = v
        # private overrides
        priv = rospy.get_param('~', {}) or {}
        for k, v in priv.items():
            flat[k] = v
        return flat

    # ---- Subscribers / timers --------------------------------------

    def _image_cb(self, msg: Image) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn_throttle(5.0, "image conversion failed: %s", e)
            return
        with self._lock:
            self._latest_img = img
            self._latest_stamp = msg.header.stamp or rospy.Time.now()

    def _timer_cb(self, _evt) -> None:
        # If a previous query is still in flight (VLMs can take 10-60s
        # per image on local servers), drop this tick instead of
        # piling up parallel HTTP requests. Service calls have priority,
        # so we use a non-blocking acquire here.
        if not self._query_lock.acquire(blocking=False):
            rospy.logdebug("VLM query still in progress; skipping tick")
            return
        try:
            now = time.time()
            if now - self._last_query_t < self._period * 0.95:
                return
            self._last_query_t = now
            self._run_once(blocking=False)
        finally:
            self._query_lock.release()

    # ---- Service ---------------------------------------------------

    def _recognize_srv(self, _req):
        # Try to grab the query lock with a short wait. If the
        # background timer is already in the middle of a VLM call
        # (which can take 30-60 s on cloud thinking models), report
        # busy immediately rather than making the user wait blindly.
        wait_s = float(self.params.get('service_lock_wait_s', 2.0))
        if not self._query_lock.acquire(timeout=wait_s):
            return TriggerResponse(
                success=False,
                message=("VLM is busy with a periodic recognition. "
                         "Either wait and retry, or set "
                         "/vlm/phantom_recognizer/rate_hz: 0 to "
                         "disable the timer."))
        try:
            info = self._run_once(blocking=True)
        finally:
            self._query_lock.release()
        if info is None:
            return TriggerResponse(
                success=False,
                message="No image available yet on the configured topic.")
        if info.confidence <= 0.0 and info.phantom_type in ("", "unknown"):
            # Strip non-ASCII so the CLI doesn't print byte escapes.
            raw_ascii = (info.raw_json or '').encode(
                'ascii', 'replace').decode('ascii')[:200]
            return TriggerResponse(
                success=False,
                message="VLM call failed or returned no useful output. "
                        "See /us3d/phantom_info for raw_json. "
                        "ascii_preview=%s" % raw_ascii)
        # The service `message` is shown verbatim by `rosservice
        # call`, which yaml-dumps non-ASCII bytes as \xHH escapes
        # and looks ugly in the terminal. Keep this field strictly
        # ASCII (numeric + label + bbox) and route the real Chinese
        # description through the latched /us3d/phantom_info topic
        # (use `rostopic echo /us3d/phantom_info` to read it).
        bbox_str = (
            "[%.0f,%.0f,%.0f,%.0f]" % tuple(info.bbox_image_xyxy)
            if len(info.bbox_image_xyxy) == 4 else "[]")
        return TriggerResponse(
            success=True,
            message=(
                "type=%s conf=%.2f axis=%s len_hint=%.0fmm bbox=%s "
                "(see /us3d/phantom_info for description)"
                % (info.phantom_type, info.confidence,
                   info.scan_axis_hint or '-',
                   info.scan_length_hint_mm,
                   bbox_str)))

    # ---- Core recognition ------------------------------------------

    def _run_once(self, blocking: bool) -> Optional[PhantomInfo]:
        with self._lock:
            img = None if self._latest_img is None else self._latest_img.copy()
            stamp = self._latest_stamp
        if img is None:
            if blocking:
                rospy.logwarn("recognize requested but no image arrived yet")
            return None

        h_orig, w_orig = img.shape[:2]
        max_side_px = int(self.params.get('image_max_side_px', 768))
        long_side = max(h_orig, w_orig)
        # Same scale logic encode_image_jpeg_b64 uses internally.
        scale = (long_side / float(max_side_px)
                 if long_side > max_side_px else 1.0)

        prompt = _build_user_prompt(self._known_labels)
        t0 = time.time()
        result = self.client.chat_with_image(
            system_prompt=PHANTOM_SYSTEM_PROMPT,
            user_prompt=prompt,
            image_bgr=img,
            jpeg_quality=int(self.params.get('image_jpeg_quality', 80)),
            max_side_px=max_side_px,
            want_json=True,
        )
        dt = time.time() - t0

        info = PhantomInfo()
        info.header.stamp = stamp if stamp != rospy.Time(0) \
            else rospy.Time.now()
        info.header.frame_id = 'camera_color_optical_frame'

        if not result.ok:
            info.phantom_type = "unknown"
            info.confidence = 0.0
            info.description = "VLM call failed"
            info.raw_json = json.dumps({
                "error": result.error,
                "elapsed_s": dt,
                "http_status": result.http_status,
                "request_url": result.request_url,
                "request_model": result.request_model,
                "raw_response": result.raw_response[:512],
            })
            self.pub_info.publish(info)
            rospy.logwarn_throttle(
                10.0,
                "VLM call failed (%.2fs, HTTP %d, model=%s url=%s): %s "
                "| body: %s",
                dt, result.http_status, result.request_model,
                result.request_url, result.error,
                (result.raw_response or '')[:300] or '<empty>')
            return info

        parsed = result.parsed or {}
        # Diagnostics: when the LLM responded but JSON parsing failed,
        # OR when the model returned phantom_type=unknown, dump the
        # raw text to the console so the operator can see WHY. Without
        # this, a "type=unknown conf=0.00" line is impossible to debug.
        if not parsed:
            preview = (result.content or '').replace('\n', ' ')[:400]
            body_preview = (result.raw_response or '').replace(
                '\n', ' ')[:400]
            rospy.logwarn(
                "VLM returned non-JSON (%.2fs, HTTP %d, content=%d bytes, "
                "model=%s). content=%s | raw_body=%s",
                dt, result.http_status, len(result.content or ''),
                result.request_model,
                preview or '<empty>',
                body_preview or '<empty>')
        info.phantom_type = str(
            parsed.get('phantom_type') or 'unknown').lower()
        try:
            info.confidence = float(parsed.get('confidence', 0.0))
        except (TypeError, ValueError):
            info.confidence = 0.0
        info.confidence = max(0.0, min(1.0, info.confidence))
        info.description = str(parsed.get('description', '') or '')
        info.scan_axis_hint = str(parsed.get('scan_axis_hint', '') or '')
        try:
            info.scan_length_hint_mm = float(
                parsed.get('scan_length_hint_mm', 0.0))
        except (TypeError, ValueError):
            info.scan_length_hint_mm = 0.0

        # Try the new normalised-coordinate schema first; fall back
        # to the legacy pixel-coordinate field for older prompts /
        # models that ignored the new instructions.
        info.bbox_image_xyxy = self._parse_bbox(
            parsed, w_orig, h_orig, scale)
        # Diagnostic: if the model declared a non-unknown phantom
        # type but the bbox is missing / degenerate / out of frame,
        # the downstream anchor will fail. Surface this clearly.
        if (info.phantom_type not in ('', 'unknown')
                and not info.bbox_image_xyxy):
            raw_bbox = (parsed.get('bbox')
                        or parsed.get('bbox_norm')
                        or parsed.get('bbox_image_xyxy'))
            rospy.logwarn(
                "VLM identified type=%s but produced no usable bbox "
                "(raw=%s, image=%dx%d). Downstream /us3d/anchor_from_vlm "
                "will fail. Likely cause: model output bbox in pixels "
                "of an imagined size instead of normalised [0,1].",
                info.phantom_type, raw_bbox, w_orig, h_orig)

        info.raw_json = json.dumps(
            {
                "parsed": parsed,
                "raw_text": result.content,
                "elapsed_s": round(dt, 3),
                "usage": result.usage,
            }, ensure_ascii=False)

        self.pub_info.publish(info)
        # Always log the parsed result (or the raw text when type is
        # 'unknown' or confidence is 0 — those are the cases where
        # silent unknowns are most confusing).
        if (info.phantom_type in ('', 'unknown')
                or info.confidence <= 0.0):
            preview = (result.content or '').replace('\n', ' ')[:400]
            rospy.loginfo(
                "VLM phantom: type=%s conf=%.2f axis=%s len_hint=%.0fmm "
                "(%.2fs) raw=%s",
                info.phantom_type, info.confidence,
                info.scan_axis_hint or '-',
                info.scan_length_hint_mm, dt, preview or '<empty>')
        else:
            rospy.loginfo(
                "VLM phantom: type=%s conf=%.2f axis=%s len_hint=%.0fmm "
                "(%.2fs) desc=%s",
                info.phantom_type, info.confidence,
                info.scan_axis_hint or '-',
                info.scan_length_hint_mm, dt,
                info.description[:120].replace('\n', ' '))
        return info

    # ---- Helpers ---------------------------------------------------

    @staticmethod
    def _parse_bbox(parsed: Dict[str, Any],
                    w_orig: int, h_orig: int,
                    legacy_scale: float) -> list:
        """Convert the LLM's bbox into pixel coords on the original
        camera image.

        Tries, in order:
          1. New schema "bbox" with normalised values in [0, 1]
             (preferred — robust to whatever the model thinks the
             input image size is).
          2. Legacy "bbox_image_xyxy" treated as pixel coords on
             the down-scaled image (long side = max_side_px), then
             rescaled by `legacy_scale`.
          3. If anything looks bogus (e.g. all zeros, or values way
             out of range that can't be rescued), return [].

        Always sorts and clamps so x_min < x_max and 0 <= ... <
        original-image extents.
        """
        candidates = []
        for key in ('bbox', 'bbox_norm', 'bbox_image_xyxy'):
            v = parsed.get(key)
            if (isinstance(v, (list, tuple)) and len(v) == 4
                    and all(isinstance(x, (int, float)) for x in v)):
                candidates.append((key, [float(x) for x in v]))

        if not candidates:
            return []

        # If we have a 'bbox' that looks normalised (all values
        # below ~1.5), trust it. Otherwise treat it as legacy
        # pixel coords. This handles models that ignore the
        # normalised-coordinate instruction and emit pixels.
        for key, raw in candidates:
            looks_norm = max(abs(x) for x in raw) <= 1.5
            if looks_norm:
                xs = [raw[0] * w_orig, raw[2] * w_orig]
                ys = [raw[1] * h_orig, raw[3] * h_orig]
            else:
                xs = [raw[0] * legacy_scale, raw[2] * legacy_scale]
                ys = [raw[1] * legacy_scale, raw[3] * legacy_scale]

            x_min, x_max = sorted(xs)
            y_min, y_max = sorted(ys)
            x_min = max(0.0, min(x_min, w_orig - 1.0))
            x_max = max(0.0, min(x_max, w_orig - 1.0))
            y_min = max(0.0, min(y_min, h_orig - 1.0))
            y_max = max(0.0, min(y_max, h_orig - 1.0))

            # Drop degenerate boxes (< 5 px on either side) — these
            # almost always mean the LLM clipped or hallucinated.
            if (x_max - x_min) < 5.0 or (y_max - y_min) < 5.0:
                continue
            return [x_min, y_min, x_max, y_max]

        return []


if __name__ == '__main__':
    try:
        node = PhantomRecognizerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
