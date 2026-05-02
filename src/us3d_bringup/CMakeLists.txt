cmake_minimum_required(VERSION 3.0.2)
project(us3d_bringup)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY launch config urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
