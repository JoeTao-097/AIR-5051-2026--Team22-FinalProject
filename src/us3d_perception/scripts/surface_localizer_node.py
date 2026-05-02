#!/usr/bin/env python3
"""Surface localizer node.

Subscribes to detected ArUco marker poses and computes the scan region:
origin, scan direction, lateral direction, normal, length, and width.
Publishes us3d_msgs/ScanRegion and broadcasts phantom_frame TF.
"""

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseArray, TransformStamped, Vector3, Point
from us3d_msgs.msg import ScanRegion
import transforms3d.quaternions as tq


class SurfaceLocalizerNode:
    def __init__(self):
        rospy.init_node('surface_localizer_node')

        self.min_markers = rospy.get_param('~min_markers', 3)

        self.pub_region = rospy.Publisher('/us3d/scan_region', ScanRegion, queue_size=1, latch=True)
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        rospy.Subscriber('/us3d/markers', PoseArray, self._markers_cb)

        rospy.loginfo("Surface localizer ready (min_markers=%d)", self.min_markers)

    def _markers_cb(self, msg):
        if len(msg.poses) < self.min_markers:
            rospy.logwarn("Need at least %d markers, got %d", self.min_markers, len(msg.poses))
            return

        points = np.array([[p.position.x, p.position.y, p.position.z]
                           for p in msg.poses])

        centroid = points.mean(axis=0)

        # PCA to find principal axes on the marker plane
        centered = points - centroid
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Eigenvectors sorted by ascending eigenvalue:
        # smallest eigenvalue -> normal direction
        # largest two -> scan plane axes
        normal = eigenvectors[:, 0]
        scan_dir = eigenvectors[:, 2]     # primary scan direction (longest extent)
        lateral_dir = eigenvectors[:, 1]  # lateral direction

        # Ensure normal points upward (positive Z in base_link)
        if normal[2] < 0:
            normal = -normal
        # Ensure scan_dir has a consistent orientation
        if scan_dir[0] < 0:
            scan_dir = -scan_dir
        # Make lateral_dir consistent via right-hand rule
        lateral_dir = np.cross(normal, scan_dir)
        lateral_dir /= np.linalg.norm(lateral_dir)
        scan_dir = np.cross(lateral_dir, normal)
        scan_dir /= np.linalg.norm(scan_dir)

        # Project points onto scan_dir and lateral_dir to get bounding extent
        proj_scan = centered @ scan_dir
        proj_lat = centered @ lateral_dir

        margin = 0.01  # 10mm margin
        scan_length = proj_scan.max() - proj_scan.min() + 2 * margin
        scan_width = proj_lat.max() - proj_lat.min() + 2 * margin

        # Origin = corner of the scan region (min scan, min lateral)
        origin = centroid + (proj_scan.min() - margin) * scan_dir + \
                 (proj_lat.min() - margin) * lateral_dir

        region = ScanRegion()
        region.header.stamp = rospy.Time.now()
        region.header.frame_id = 'base_link'
        region.origin = Point(x=origin[0], y=origin[1], z=origin[2])
        region.scan_dir = Vector3(x=scan_dir[0], y=scan_dir[1], z=scan_dir[2])
        region.lateral_dir = Vector3(x=lateral_dir[0], y=lateral_dir[1], z=lateral_dir[2])
        region.normal = Vector3(x=normal[0], y=normal[1], z=normal[2])
        region.length = scan_length
        region.width = scan_width

        self.pub_region.publish(region)
        rospy.loginfo("Scan region: origin=(%.3f,%.3f,%.3f) L=%.3f W=%.3f",
                      origin[0], origin[1], origin[2], scan_length, scan_width)

        # Broadcast phantom_frame TF
        R = np.column_stack([scan_dir, lateral_dir, normal])
        quat = tq.mat2quat(R)  # w, x, y, z

        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = 'base_link'
        tf_msg.child_frame_id = 'phantom_frame'
        tf_msg.transform.translation.x = centroid[0]
        tf_msg.transform.translation.y = centroid[1]
        tf_msg.transform.translation.z = centroid[2]
        tf_msg.transform.rotation.x = quat[1]
        tf_msg.transform.rotation.y = quat[2]
        tf_msg.transform.rotation.z = quat[3]
        tf_msg.transform.rotation.w = quat[0]
        self.tf_broadcaster.sendTransform(tf_msg)


if __name__ == '__main__':
    try:
        node = SurfaceLocalizerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
