cmake_minimum_required(VERSION 3.0.0)
project(CrosswalkDetection VERSION 0.1.0)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
set (CMAKE_CXX_STANDARD 17)
include(CTest)
enable_testing()

add_executable(CrosswalkDetection main.cpp)
target_link_libraries( CrosswalkDetection ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
