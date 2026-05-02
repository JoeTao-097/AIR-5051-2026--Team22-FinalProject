#!/usr/bin/env python3
"""Publish ArUco marker TF for hand-eye calibration.

Detects a single ArUco marker and publishes its pose as a TF frame
(aruco_marker_frame) relative to camera_color_optical_frame.
Replaces aruco_ros/single for compatibility with easy_handeye.
"""

import numpy as np
import rospy
import cv2
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge


class ArucoTFPublisher:
    ARUCO_DICTS = {
        0: cv2.aruco.DICT_4X4_50,
        1: cv2.aruco.DICT_4X4_100,
        2: cv2.aruco.DICT_5X5_50,
        3: cv2.aruco.DICT_5X5_100,
    }

    def __init__(self):
        rospy.init_node('aruco_tf_publisher')

        dict_id = rospy.get_param('~dictionary', 0)
        self.marker_id = rospy.get_param('~marker_id', 0)
        self.marker_size = rospy.get_param('~marker_size', 0.075)
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_optical_frame')
        self.marker_frame = rospy.get_param('~marker_frame', 'aruco_marker_frame')

        aruco_dict_id = self.ARUCO_DICTS.get(dict_id, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.pub_result = rospy.Publisher('~result', Image, queue_size=1)

        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self._info_cb)
        rospy.Subscriber('/camera/color/image_raw', Image, self._image_cb)

        rospy.loginfo("ArUco TF publisher ready (dict=%d, marker_id=%d, size=%.3f m)",
                      dict_id, self.marker_id, self.marker_size)

    def _info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            rospy.loginfo("Camera intrinsics received")

    def _image_cb(self, msg):
        if self.camera_matrix is None:
            return

        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        corners, ids, _ = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            return

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

        found = False
        for i, mid in enumerate(ids.flatten()):
            if mid == self.marker_id:
                self._publish_tf(rvecs[i], tvecs[i], msg.header.stamp)
                found = True
                break

        debug_img = img.copy()
        cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)
        if found:
            cv2.drawFrameAxes(debug_img, self.camera_matrix, self.dist_coeffs,
                              rvecs[i], tvecs[i], self.marker_size * 0.5)
        self.pub_result.publish(self.bridge.cv2_to_imgmsg(debug_img, 'bgr8'))

    def _publish_tf(self, rvec, tvec, stamp):
        R, _ = cv2.Rodrigues(rvec)
        from transforms3d.quaternions import mat2quat
        quat = mat2quat(R)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.camera_frame
        t.child_frame_id = self.marker_frame
        t.transform.translation.x = tvec[0][0]
        t.transform.translation.y = tvec[0][1]
        t.transform.translation.z = tvec[0][2]
        t.transform.rotation.w = quat[0]
        t.transform.rotation.x = quat[1]
        t.transform.rotation.y = quat[2]
        t.transform.rotation.z = quat[3]

        self.tf_broadcaster.sendTransform(t)


if __name__ == '__main__':
    try:
        node = ArucoTFPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
