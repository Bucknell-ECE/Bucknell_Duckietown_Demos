cmake_minimum_required(VERSION 2.8.3)
project(augmented_reality_basics)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  duckietown_msgs
  sensor_msgs
)

catkin_python_setup()

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})
