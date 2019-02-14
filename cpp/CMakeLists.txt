
cmake_minimum_required(VERSION 3.1)
# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(OpenCV_DIR /usr/local/lib/cmake/opencv4)

project(cpp)

find_package(OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable( yolo yolo.cpp )
target_link_libraries(yolo ${OpenCV_LIBS})


