cmake_minimum_required(VERSION 3.8)
project(lane_det_demo)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR /home/kosuke/opencv-3.2.0/build)
find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cpp polyfit.cpp polyfit.h lane_detection.cpp lane_detection.h)
add_executable(lane_det_demo ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( lane_det_demo ${OpenCV_LIBS} )


