#!/usr/bin/env python3
"""
Add known collision objects (table, walls, etc.) to the MoveIt PlanningScene.
Run after MoveIt move_group is up.

Usage:
    rosrun us3d_control add_scene_objects.py
    rosrun us3d_control add_scene_objects.py _table_height:=-0.01 _table_size_x:=1.5 _table_size_y:=1.5
"""
import sys
import rospy
from moveit_commander import PlanningSceneInterface, roscpp_initialize
import geometry_msgs.msg


def add_table(scene, height, size_x, size_y):
    pose = geometry_msgs.msg.PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.position.x = 0.0
    pose.pose.position.y = 0.0
    pose.pose.position.z = height - 0.01  # box center is half-thickness below surface
    pose.pose.orientation.w = 1.0
    scene.add_box("table", pose, size=(size_x, size_y, 0.02))
    rospy.loginfo("Added table: %.1f x %.1f m at z=%.3f (base_link)", size_x, size_y, height)


def add_back_wall(scene, distance):
    pose = geometry_msgs.msg.PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.position.x = -distance
    pose.pose.position.y = 0.0
    pose.pose.position.z = 0.5
    pose.pose.orientation.w = 1.0
    scene.add_box("back_wall", pose, size=(0.02, 2.0, 1.5))
    rospy.loginfo("Added back wall at x=%.2f (base_link)", -distance)


def main():
    roscpp_initialize(sys.argv)
    rospy.init_node("add_scene_objects")

    scene = PlanningSceneInterface(synchronous=True)
    rospy.sleep(1.0)

    table_height = rospy.get_param("~table_height", -0.01)
    table_size_x = rospy.get_param("~table_size_x", 2.0)
    table_size_y = rospy.get_param("~table_size_y", 2.0)
    back_wall_dist = rospy.get_param("~back_wall_distance", 0.0)

    add_table(scene, table_height, table_size_x, table_size_y)

    if back_wall_dist > 0:
        add_back_wall(scene, back_wall_dist)

    rospy.loginfo("Scene objects added. Known objects: %s", scene.get_known_object_names())
    rospy.loginfo("Node will stay alive to keep objects in the scene. Ctrl+C to exit.")
    rospy.spin()


if __name__ == "__main__":
    main()
