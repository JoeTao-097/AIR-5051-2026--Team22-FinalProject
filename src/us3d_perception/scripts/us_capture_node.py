#!/usr/bin/env python3
"""Ultrasound image capture node.

Reads frames from a USB video capture card via OpenCV and publishes
them as sensor_msgs/Image on /us3d/image_raw.
"""

import os
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class USCaptureNode:
    def __init__(self):
        rospy.init_node('us_capture_node')

        # device_id can be an integer (V4L2 index) or a string path
        # (e.g. "/dev/us_capture" — preferred, since the integer index
        # changes after USB re-enumeration).
        self.device_id = rospy.get_param('~device_id', 0)
        self.frame_width = rospy.get_param('~frame_width', 640)
        self.frame_height = rospy.get_param('~frame_height', 480)
        self.fps = rospy.get_param('~fps', 30)
        self.frame_id = rospy.get_param('~frame_id', 'us_image_plane')
        # Warn (but keep publishing) if frames are entirely black for
        # this many seconds — usually means the US machine is off or
        # the HDMI/VGA signal is missing.
        self.blank_warn_secs = rospy.get_param('~blank_warn_secs', 3.0)

        self.bridge = CvBridge()
        self.pub = rospy.Publisher('/us3d/image_raw', Image, queue_size=1)

        self.cap = self._open_capture(self.device_id)
        if self.cap is None or not self.cap.isOpened():
            rospy.logfatal("Cannot open video device %r", self.device_id)
            rospy.signal_shutdown("Camera open failed")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        rospy.loginfo("US capture opened: %dx%d @ %.1f fps (device %r)",
                      actual_w, actual_h, actual_fps, self.device_id)

        self._last_signal_time = rospy.Time.now()
        self._blank_warned = False

    def _open_capture(self, device_id):
        """Open the V4L2 device.

        Accepts either an integer index or a device path string. When
        given a path, resolves any symlink (e.g. /dev/us_capture →
        /dev/video0) and opens that path with the V4L2 backend. This
        is more robust than opening by integer index, which is unstable
        across USB re-enumeration.

        If the requested path is missing (typical when the udev rule
        in tools/99-us-capture.rules has not been installed yet),
        auto-discover the MACROSILICON USB3 Video grabber by USB id
        and fall back to its /dev/videoN, so the node works out of
        the box.
        """
        if isinstance(device_id, str):
            path = device_id
            if not os.path.exists(path):
                fallback = self._autodiscover_grabber()
                if fallback is None:
                    rospy.logfatal(
                        "Device path %s does not exist and no "
                        "MACROSILICON USB3 Video grabber found via "
                        "/sys/class/video4linux. Plug it in or pass "
                        "_device_id:=N (integer index) explicitly.",
                        path)
                    return None
                rospy.logwarn(
                    "%s missing — auto-discovered grabber at %s. "
                    "Install tools/99-us-capture.rules for a stable "
                    "symlink.", path, fallback)
                path = fallback
            real = os.path.realpath(path)
            if real != path:
                rospy.loginfo("Resolved %s -> %s", path, real)
            return cv2.VideoCapture(real, cv2.CAP_V4L2)
        return cv2.VideoCapture(int(device_id))

    @staticmethod
    def _autodiscover_grabber():
        """Find /dev/videoN for the MACROSILICON USB3 Video card.

        Walks /sys/class/video4linux/videoN/device/.. up the USB
        hierarchy and matches idVendor==345f && idProduct==2131.
        Among multiple matches (the card exposes USB2+USB3 endpoints
        as two separate /dev/videoN nodes), prefer the lowest index
        (typically the higher-resolution one).
        """
        VENDOR = '345f'
        PRODUCT = '2131'
        candidates = []
        try:
            for entry in sorted(os.listdir('/sys/class/video4linux')):
                if not entry.startswith('video'):
                    continue
                base = '/sys/class/video4linux/' + entry + '/device'
                # Walk parents to find idVendor/idProduct in USB tree.
                cur = base
                for _ in range(8):
                    try:
                        cur = os.path.realpath(cur)
                        v_path = os.path.join(cur, 'idVendor')
                        p_path = os.path.join(cur, 'idProduct')
                        if (os.path.exists(v_path)
                                and os.path.exists(p_path)):
                            with open(v_path) as f:
                                vid = f.read().strip()
                            with open(p_path) as f:
                                pid = f.read().strip()
                            if vid.lower() == VENDOR \
                                    and pid.lower() == PRODUCT:
                                candidates.append(
                                    '/dev/' + entry)
                            break
                        cur = os.path.dirname(cur)
                    except (OSError, IOError):
                        break
        except (OSError, IOError):
            return None
        if not candidates:
            return None
        # Lowest videoN among matches (MACROSILICON exposes two
        # endpoints; index 0 is typically USB3 / higher quality).
        candidates.sort(
            key=lambda p: int(p[len('/dev/video'):]) if
                          p[len('/dev/video'):].isdigit() else 999)
        return candidates[0]

    def _check_blank(self, frame):
        """Warn periodically if the captured frame is entirely black.

        Mean intensity below 1 is essentially "no signal" — typically
        the US machine is off or the video cable is unplugged.
        """
        is_blank = bool(frame.max() < 5)
        now = rospy.Time.now()
        if is_blank:
            elapsed = (now - self._last_signal_time).to_sec()
            if elapsed > self.blank_warn_secs and not self._blank_warned:
                rospy.logwarn(
                    "US capture frames have been completely BLANK for %.1fs. "
                    "Check: (1) US machine is powered on, (2) HDMI/VGA cable "
                    "to capture card is seated, (3) US machine is in B-mode "
                    "(not standby).", elapsed)
                self._blank_warned = True
        else:
            self._last_signal_time = now
            if self._blank_warned:
                rospy.loginfo("US signal recovered.")
                self._blank_warned = False

    def run(self):
        rate = rospy.Rate(self.fps)
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn_throttle(5.0, "Failed to read US frame")
                rate.sleep()
                continue

            self._check_blank(frame)

            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.frame_id
            self.pub.publish(msg)
            rate.sleep()

        self.cap.release()


if __name__ == '__main__':
    try:
        node = USCaptureNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
