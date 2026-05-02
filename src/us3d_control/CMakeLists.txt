cmake_minimum_required(VERSION 3.0.2)
project(us3d_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  std_srvs
  tf2_ros
  us3d_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/path_generator_node.py
  scripts/force_scan_node.py
  scripts/hand_eye_calib_node.py
  scripts/add_scene_objects.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
