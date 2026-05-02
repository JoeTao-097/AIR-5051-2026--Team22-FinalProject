#!/usr/bin/env python3
"""Scan path generator node.

Subscribes to /us3d/scan_region and generates parallel scan line
waypoints for force-controlled ultrasound scanning.
Publishes PoseArray on /us3d/scan_path.
"""

import numpy as np
import rospy
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from us3d_msgs.msg import ScanRegion
import transforms3d.quaternions as tq


class PathGeneratorNode:
    def __init__(self):
        rospy.init_node('path_generator_node')

        self.line_spacing = rospy.get_param('/scan/line_spacing', 0.003)
        self.waypoint_step = rospy.get_param('/scan/waypoint_step', 0.001)
        self.approach_height = rospy.get_param('/scan/approach_height', 0.05)

        self.pub_path = rospy.Publisher('/us3d/scan_path', PoseArray, queue_size=1, latch=True)

        rospy.Subscriber('/us3d/scan_region', ScanRegion, self._region_cb)

        rospy.loginfo("Path generator ready (spacing=%.1fmm, step=%.1fmm)",
                      self.line_spacing * 1000, self.waypoint_step * 1000)

    def _region_cb(self, region):
        origin = np.array([region.origin.x, region.origin.y, region.origin.z])
        scan_dir = np.array([region.scan_dir.x, region.scan_dir.y, region.scan_dir.z])
        lateral_dir = np.array([region.lateral_dir.x, region.lateral_dir.y, region.lateral_dir.z])
        normal = np.array([region.normal.x, region.normal.y, region.normal.z])

        n_lines = max(1, int(region.width / self.line_spacing))
        n_points = max(2, int(region.length / self.waypoint_step))

        # Build orientation: probe Z-axis points into surface (-normal)
        # probe X-axis along scan_dir
        z_axis = -normal / np.linalg.norm(normal)
        x_axis = scan_dir / np.linalg.norm(scan_dir)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])
        quat = tq.mat2quat(R)  # w, x, y, z

        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = 'base_link'

        for i in range(n_lines):
            lateral_offset = i * self.line_spacing
            line_origin = origin + lateral_offset * lateral_dir

            for j in range(n_points):
                scan_offset = j * self.waypoint_step
                pos = line_origin + scan_offset * scan_dir

                pose = Pose()
                pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
                pose.orientation = Quaternion(x=quat[1], y=quat[2], z=quat[3], w=quat[0])
                pose_array.poses.append(pose)

        self.pub_path.publish(pose_array)
        rospy.loginfo("Generated %d scan lines, %d total waypoints",
                      n_lines, len(pose_array.poses))

    @property
    def points_per_line(self):
        return None  # computed dynamically


if __name__ == '__main__':
    try:
        node = PathGeneratorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
