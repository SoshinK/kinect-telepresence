cmake_minimum_required(VERSION 3.8)

project(TryBuild)

set(default_build_type "Release")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(src ./src/undistort.cpp)
add_library(undistort STATIC ${src})
include_directories(./headers)
 


find_package(k4a REQUIRED)
include_directories(${k4a_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_subdirectory(extern)

# include_directories(sample_helper_includes)

# add_subdirectory(sample_helper_libs)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}  k4a k4abt k4arecord ${OpenCV_LIBS} undistort)


