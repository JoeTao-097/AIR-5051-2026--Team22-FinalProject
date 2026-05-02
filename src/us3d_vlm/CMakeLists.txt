cmake_minimum_required(VERSION 3.0.2)
project(us3d_vlm)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  std_srvs
  sensor_msgs
  geometry_msgs
  cv_bridge
  us3d_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs std_srvs sensor_msgs geometry_msgs cv_bridge us3d_msgs
)

catkin_install_python(PROGRAMS
  scripts/phantom_recognizer_node.py
  scripts/instruction_planner_node.py
  scripts/vlm_anchor_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
