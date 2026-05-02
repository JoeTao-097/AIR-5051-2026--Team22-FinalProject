cmake_minimum_required(VERSION 3.0.2)
project(us3d_reconstruction)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/voxel_reconstruct.py
  scripts/tsdf_reconstruct.py
  scripts/visualize_volume.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
