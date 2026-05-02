cmake_minimum_required(VERSION 3.0.2)
project(us3d_acquisition)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  message_filters
  cv_bridge
  rosbag
  us3d_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/sync_recorder_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
