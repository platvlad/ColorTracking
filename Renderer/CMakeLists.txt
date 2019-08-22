cmake_minimum_required(VERSION 3.14)
project(Renderer)

set(CMAKE_CXX_STANDARD 98)

FIND_PACKAGE(OpenCV 3 REQUIRED)

set(LIBRARIES
        ${OpenCV_LIBS}
)

INCLUDE_DIRECTORIES(
        ${OpenCV_INCLUDE_DIRS}
        ../deps/include
        include
)

add_library(Renderer include/histogram.h src/histogram.cpp include/CircleWindow.h src/CircleWindow.cpp src/mesh.cpp include/mesh.h src/renderer.cpp include/renderer.h src/maps.cpp include/maps.h)

TARGET_LINK_LIBRARIES(Renderer ${LIBRARIES})
