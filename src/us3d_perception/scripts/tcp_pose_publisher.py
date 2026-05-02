#!/usr/bin/env python3
"""Publish TCP pose from TF tree as PoseStamped.

Looks up the transform from base_link to tool0 at a fixed rate
and publishes it on /us3d/current_pose for downstream consumers
(e.g. sync_recorder_node).
"""

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped


def main():
    rospy.init_node('tcp_pose_publisher')

    source_frame = rospy.get_param('~source_frame', 'base_link')
    target_frame = rospy.get_param('~target_frame', 'tool0')
    rate_hz = rospy.get_param('~rate', 125.0)

    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer)

    pub = rospy.Publisher('/us3d/current_pose', PoseStamped, queue_size=10)
    rate = rospy.Rate(rate_hz)

    rospy.loginfo("TCP pose publisher: %s -> %s at %.0f Hz",
                  source_frame, target_frame, rate_hz)

    while not rospy.is_shutdown():
        try:
            t = tf_buffer.lookup_transform(source_frame, target_frame,
                                           rospy.Time(0),
                                           rospy.Duration(0.1))
            msg = PoseStamped()
            msg.header = t.header
            msg.pose.position.x = t.transform.translation.x
            msg.pose.position.y = t.transform.translation.y
            msg.pose.position.z = t.transform.translation.z
            msg.pose.orientation = t.transform.rotation
            pub.publish(msg)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
