cmake_minimum_required(VERSION 3.0.2)
project(us3d_perception)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  cv_bridge
  tf2_ros
  us3d_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/us_capture_node.py
  scripts/aruco_detector_node.py
  scripts/surface_localizer_node.py
  scripts/tcp_pose_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
