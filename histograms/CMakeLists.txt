cmake_minimum_required(VERSION 3.14)
project(histograms)

set(CMAKE_CXX_STANDARD 98)

FIND_PACKAGE(OpenCV 3 REQUIRED)

set(DEBUG_LIBRARIES
        ${OpenCV_LIBS}
#        ${PROJECT_SOURCE_DIR}/../Renderer/cmake-build-debug/libRenderer.a
)

set(OPTIMIZED_LIBRARIES
        ${OpenCV_LIBS}
#        ${PROJECT_SOURCE_DIR}/../Renderer/cmake-build-release/libRenderer.a
        )

INCLUDE_DIRECTORIES(
        ${OpenCV_INCLUDE_DIRS}
        ../deps/include
#        ../Renderer/include
        include
)

add_library(histograms src/Object3d.cpp src/PoseEstimator.cpp include/CircleWindow.h include/histogram.h include/projection.h include/mesh.h include/renderer.h src/CircleWindow.cpp src/histogram.cpp src/projection.cpp src/mesh.cpp src/renderer.cpp include/MeshRenderer.h src/MeshRenderer.cpp include/PoseEstimator.h)

TARGET_LINK_LIBRARIES(histograms debug ${DEBUG_LIBRARIES})
TARGET_LINK_LIBRARIES(histograms optimized ${OPTIMIZED_LIBRARIES})
