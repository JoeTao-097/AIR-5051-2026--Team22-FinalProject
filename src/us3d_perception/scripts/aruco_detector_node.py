#!/usr/bin/env python3
"""ArUco marker detector for phantom localization.

Subscribes to camera RGB+depth, detects ArUco markers on the phantom
surface, and publishes their 3D positions in the robot base frame.
Supports multi-pose observation and averaging for improved accuracy.
"""

import numpy as np
import rospy
import cv2
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from cv_bridge import CvBridge
from std_srvs.srv import Trigger, TriggerResponse
import message_filters
import transforms3d.quaternions as tq
import transforms3d.affines as ta


# ArUco dictionary name -> OpenCV constant
ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class ArucoDetectorNode:
    def __init__(self):
        rospy.init_node('aruco_detector_node')

        dict_name = rospy.get_param('/aruco/dictionary', 'DICT_4X4_50')
        self.marker_size = rospy.get_param('/aruco/marker_size', 0.075)
        self.marker_ids = rospy.get_param('/aruco/marker_ids', [0, 1, 2, 3])

        aruco_dict_id = ARUCO_DICTS.get(dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Accumulated marker observations: {marker_id: [poses_in_base]}
        self.observations = {}
        self.detecting = False

        self.pub_markers = rospy.Publisher('/us3d/markers', PoseArray, queue_size=1, latch=True)
        self.pub_debug = rospy.Publisher('/us3d/aruco_debug', Image, queue_size=1)

        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self._camera_info_cb)

        color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)
        ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=5, slop=0.05)
        ts.registerCallback(self._image_cb)

        self.srv_detect = rospy.Service('/us3d/detect_markers', Trigger, self._detect_srv)
        self.srv_clear = rospy.Service('/us3d/clear_markers', Trigger, self._clear_srv)

        rospy.loginfo("ArUco detector ready (dict=%s, marker_size=%.3f m)", dict_name, self.marker_size)

    def _camera_info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            rospy.loginfo("Camera intrinsics received")

    def _detect_srv(self, req):
        """Service to trigger one round of detection from current view."""
        self.detecting = True
        rospy.sleep(2.0)
        self.detecting = False
        n = sum(len(v) for v in self.observations.values())
        detected_ids = sorted(self.observations.keys())
        rospy.loginfo("Detected marker IDs: %s, total observations: %d",
                      detected_ids, n)
        return TriggerResponse(
            success=True,
            message="Detected IDs: %s, total observations: %d" % (detected_ids, n))

    def _clear_srv(self, req):
        self.observations.clear()
        return TriggerResponse(success=True, message="Cleared all observations")

    def _image_cb(self, color_msg, depth_msg):
        if not self.detecting or self.camera_matrix is None:
            return

        color_img = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        corners, ids, _ = cv2.aruco.detectMarkers(
            color_img, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            return

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                'base_link', 'camera_color_optical_frame',
                color_msg.header.stamp, rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, "TF lookup failed: %s", str(e))
            return

        T_base_cam = self._tf_to_matrix(tf_stamped.transform)

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id not in self.marker_ids:
                continue

            R_cam, _ = cv2.Rodrigues(rvecs[i])
            t_cam = tvecs[i].flatten()
            T_cam_marker = np.eye(4)
            T_cam_marker[:3, :3] = R_cam
            T_cam_marker[:3, 3] = t_cam

            T_base_marker = T_base_cam @ T_cam_marker

            if marker_id not in self.observations:
                self.observations[marker_id] = []
            self.observations[marker_id].append(T_base_marker)

        debug_img = color_img.copy()
        cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug_img, 'bgr8'))

        self._publish_averaged_markers()

    def _publish_averaged_markers(self):
        """Publish averaged marker poses from all observations."""
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = 'base_link'

        for marker_id in sorted(self.observations.keys()):
            matrices = self.observations[marker_id]
            avg_t = np.mean([m[:3, 3] for m in matrices], axis=0)
            avg_R = matrices[-1][:3, :3]

            quat = tq.mat2quat(avg_R)  # w, x, y, z
            pose = Pose()
            pose.position = Point(x=avg_t[0], y=avg_t[1], z=avg_t[2])
            pose.orientation = Quaternion(x=quat[1], y=quat[2], z=quat[3], w=quat[0])
            pose_array.poses.append(pose)

        self.pub_markers.publish(pose_array)
        rospy.loginfo("Published %d averaged marker poses", len(pose_array.poses))

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
        node = ArucoDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
