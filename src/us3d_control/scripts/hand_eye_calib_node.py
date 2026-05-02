#!/usr/bin/env python3
"""Hand-eye calibration helper node.

Automates the data collection process for easy_handeye:
moves the robot to a set of predefined poses while the camera
observes a fixed ArUco calibration board.
"""

import rospy
import rtde_control
import rtde_receive
from std_srvs.srv import Trigger, TriggerResponse


class HandEyeCalibNode:
    def __init__(self):
        rospy.init_node('hand_eye_calib_node')

        self.robot_ip = rospy.get_param('/robot/ip', '192.168.1.10')
        self.rtde_c = None
        self.rtde_r = None

        # Predefined calibration poses (joint angles in radians)
        # These should provide diverse viewpoints of the calibration board
        self.calib_poses = [
            [-1.5708, -1.3090, 1.3090, -1.5708, -1.5708, 0.0000],
            [-1.5708, -1.4835, 1.4835, -1.5708, -1.5708, 0.0000],
            [-1.5708, -1.1345, 1.1345, -1.5708, -1.5708, 0.0000],
            [-1.3963, -1.3090, 1.3090, -1.5708, -1.5708, 0.0000],
            [-1.7453, -1.3090, 1.3090, -1.5708, -1.5708, 0.0000],
            [-1.5708, -1.3090, 1.3090, -1.5708, -1.5708, 0.3491],
            [-1.5708, -1.3090, 1.3090, -1.5708, -1.5708, -0.3491],
            [-1.3963, -1.4835, 1.4835, -1.5708, -1.5708, 0.1745],
            [-1.7453, -1.1345, 1.1345, -1.5708, -1.5708, -0.1745],
            [-1.5708, -1.2217, 1.2217, -1.5708, -1.5708, 0.2618],
            [-1.5708, -1.3963, 1.3963, -1.5708, -1.5708, -0.2618],
            [-1.3090, -1.3090, 1.3090, -1.5708, -1.5708, 0.0000],
            [-1.8326, -1.3090, 1.3090, -1.5708, -1.5708, 0.0000],
            [-1.5708, -1.3090, 1.4835, -1.7453, -1.5708, 0.0000],
            [-1.5708, -1.3090, 1.1345, -1.3963, -1.5708, 0.0000],
        ]

        rospy.Service('/us3d/calib_connect', Trigger, self._connect_srv)
        rospy.Service('/us3d/calib_collect', Trigger, self._collect_srv)

        rospy.loginfo("Hand-eye calibration helper ready (%d poses)", len(self.calib_poses))

    def _connect_srv(self, req):
        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            return TriggerResponse(success=True, message="Connected to robot")
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))

    def _collect_srv(self, req):
        """Move through calibration poses, pausing at each for data capture."""
        if self.rtde_c is None:
            return TriggerResponse(success=False, message="Robot not connected")

        try:
            # Wait for easy_handeye take_sample service
            take_sample = None
            try:
                rospy.wait_for_service('/us3d_handeye/take_sample', timeout=5.0)
                take_sample = rospy.ServiceProxy('/us3d_handeye/take_sample', Trigger)
            except rospy.ROSException:
                rospy.logwarn("easy_handeye take_sample service not found, manual capture needed")

            for i, joints in enumerate(self.calib_poses):
                if rospy.is_shutdown():
                    break

                rospy.loginfo("Moving to calibration pose %d/%d", i + 1, len(self.calib_poses))
                self.rtde_c.moveJ(joints, 0.5, 0.5)
                rospy.sleep(2.0)  # settle time

                if take_sample:
                    resp = take_sample()
                    rospy.loginfo("  Sample %d: %s", i + 1, resp.message)
                else:
                    rospy.loginfo("  At pose %d, waiting 3s for manual capture...", i + 1)
                    rospy.sleep(3.0)

            return TriggerResponse(success=True,
                                   message="Collected %d calibration samples" % len(self.calib_poses))
        except Exception as e:
            return TriggerResponse(success=False, message="Collection failed: %s" % str(e))


if __name__ == '__main__':
    try:
        node = HandEyeCalibNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
