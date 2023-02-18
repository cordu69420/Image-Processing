cmake_minimum_required(VERSION 3.0.0)
project(ImageThresholding VERSION 0.1.0)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include(CTest)
enable_testing()

add_executable(ImageThresholding main.cpp)
target_link_libraries(ImageThresholding ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
