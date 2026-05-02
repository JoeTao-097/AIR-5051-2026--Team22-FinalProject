cmake_minimum_required(VERSION 3.0.2)
project(us3d_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  sensor_msgs
  geometry_msgs
)

add_message_files(FILES
  USFrame.msg
  ScanDataPoint.msg
  ScanRegion.msg
  PhantomInfo.msg
)

add_service_files(FILES
  PlanFromInstruction.srv
)

generate_messages(DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs geometry_msgs
)
