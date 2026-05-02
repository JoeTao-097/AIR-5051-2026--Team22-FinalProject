#!/usr/bin/env python3
"""Synchronized data recorder node.

Uses message_filters.ApproximateTimeSynchronizer to align:
  - Ultrasound image (/us3d/image_raw)
  - TCP pose (/us3d/current_pose)
  - Wrench (/us3d/wrench)

Saves structured dataset: PNG frames + metadata CSV + rosbag backup.
"""

import os
import csv
import datetime
import numpy as np
import rospy
import cv2
import rosbag
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, WrenchStamped
from cv_bridge import CvBridge
from std_srvs.srv import Trigger, TriggerResponse
import message_filters


class SyncRecorderNode:
    def __init__(self):
        rospy.init_node('sync_recorder_node')

        base_dir = rospy.get_param('/recording/output_dir',
                                   os.path.expanduser('~/joe/us3dscan/data/scans'))
        self.sync_slop = rospy.get_param('/recording/sync_tolerance', 0.01)
        self.save_bag = rospy.get_param('/recording/save_rosbag', True)
        # Number of leading frames to discard after start_recording to avoid
        # capturing the first sync_cb that may carry a stale image still
        # buffered in the USB capture pipeline (~50-150 ms).
        self.warmup_frames = rospy.get_param('/recording/warmup_frames', 5)

        self.base_dir = os.path.expandvars(base_dir)
        self.bridge = CvBridge()
        self.recording = False
        self.frame_count = 0
        self.received_count = 0
        self.scan_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.bag = None

        # Synchronized subscribers
        us_sub = message_filters.Subscriber('/us3d/image_raw', Image)
        pose_sub = message_filters.Subscriber('/us3d/current_pose', PoseStamped)
        wrench_sub = message_filters.Subscriber('/us3d/wrench', WrenchStamped)

        ts = message_filters.ApproximateTimeSynchronizer(
            [us_sub, pose_sub, wrench_sub],
            queue_size=10, slop=self.sync_slop)
        ts.registerCallback(self._sync_cb)

        rospy.Service('/us3d/start_recording', Trigger, self._start_srv)
        rospy.Service('/us3d/stop_recording', Trigger, self._stop_srv)

        rospy.loginfo("Sync recorder ready (slop=%.3fs, bag=%s)", self.sync_slop, self.save_bag)

    def _start_srv(self, req):
        if self.recording:
            return TriggerResponse(success=False, message="Already recording")

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.scan_dir = os.path.join(self.base_dir, 'scan_' + timestamp)
        frames_dir = os.path.join(self.scan_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        csv_path = os.path.join(self.scan_dir, 'metadata.csv')
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'frame_id', 'timestamp',
            'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw',
            'fx', 'fy', 'fz', 'tx', 'ty', 'tz'
        ])

        if self.save_bag:
            bag_path = os.path.join(self.scan_dir, 'scan.bag')
            self.bag = rosbag.Bag(bag_path, 'w')

        self.frame_count = 0
        self.received_count = 0
        self.recording = True
        rospy.loginfo("Recording started: %s (warmup_frames=%d)",
                      self.scan_dir, self.warmup_frames)
        return TriggerResponse(success=True, message="Recording to %s" % self.scan_dir)

    def _stop_srv(self, req):
        if not self.recording:
            return TriggerResponse(success=False, message="Not recording")

        self.recording = False

        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

        if self.bag:
            self.bag.close()
            self.bag = None

        msg = "Recorded %d frames to %s" % (self.frame_count, self.scan_dir)
        rospy.loginfo(msg)
        return TriggerResponse(success=True, message=msg)

    def _sync_cb(self, us_msg, pose_msg, wrench_msg):
        if not self.recording:
            return

        # Drop the first `warmup_frames` to avoid capturing a stale frame
        # still buffered in the USB capture pipeline at the moment we
        # called /us3d/start_recording.
        self.received_count += 1
        if self.received_count <= self.warmup_frames:
            return

        # Save ultrasound frame as PNG
        frame = self.bridge.imgmsg_to_cv2(us_msg, 'bgr8')
        frame_path = os.path.join(self.scan_dir, 'frames', '%06d.png' % self.frame_count)
        cv2.imwrite(frame_path, frame)

        # Write metadata row
        p = pose_msg.pose.position
        q = pose_msg.pose.orientation
        f = wrench_msg.wrench.force
        t = wrench_msg.wrench.torque
        stamp = us_msg.header.stamp.to_sec()

        self.csv_writer.writerow([
            self.frame_count, '%.6f' % stamp,
            '%.6f' % p.x, '%.6f' % p.y, '%.6f' % p.z,
            '%.6f' % q.x, '%.6f' % q.y, '%.6f' % q.z, '%.6f' % q.w,
            '%.4f' % f.x, '%.4f' % f.y, '%.4f' % f.z,
            '%.4f' % t.x, '%.4f' % t.y, '%.4f' % t.z
        ])

        # Write to rosbag
        if self.bag:
            self.bag.write('/us3d/image_raw', us_msg, us_msg.header.stamp)
            self.bag.write('/us3d/current_pose', pose_msg, pose_msg.header.stamp)
            self.bag.write('/us3d/wrench', wrench_msg, wrench_msg.header.stamp)

        self.frame_count += 1

        if self.frame_count % 100 == 0:
            rospy.loginfo("Recorded %d frames", self.frame_count)

    def __del__(self):
        if self.csv_file:
            self.csv_file.close()
        if self.bag:
            self.bag.close()


if __name__ == '__main__':
    try:
        node = SyncRecorderNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
