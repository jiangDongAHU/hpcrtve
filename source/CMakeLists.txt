cmake_minimum_required(VERSION 2.8.0)
project(demo)

set(CMAKE_BUILD_TYPE Debug)
# OPENCV
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)

CUDA_ADD_EXECUTABLE(demo main.cpp kernel.cu)

target_link_libraries(demo ${OpenCV_LIBS})
