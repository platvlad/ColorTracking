cmake_minimum_required(VERSION 3.12)
project(histograms)

set(CMAKE_CXX_STANDARD 98)

FIND_PACKAGE(glm REQUIRED)

FIND_PACKAGE(OpenCV REQUIRED)


set(DEBUG_LIBRARIES
        ${GLM_LIBS}
        ${OpenCV_LIBS}
#        ${PROJECT_SOURCE_DIR}\\..\\deps\\opencv2\\OPENCV_STATICLIB_WIN_VS2015\\*.lib
)

set(OPTIMIZED_LIBRARIES
        ${GLM_LIBS}
        ${OpenCV_LIBS}
#        ${PROJECT_SOURCE_DIR}\\..\\deps\\opencv2\\OPENCV_STATICLIB_WIN_VS2015\\*.lib
        )

INCLUDE_DIRECTORIES(
        ${GLM_INCLUDE_DIRS}
#        ${PROJECT_SOURCE_DIR}\\..\\deps\\opencv2\\OPENCV_INCLUDE
        ${OpenCV_INCLUDE_DIRS}
        include
)

add_library(histograms src/Object3d.cpp src/PoseEstimator.cpp include/CircleWindow.h include/histogram.h include/projection.h include/mesh.h include/renderer.h src/CircleWindow.cpp src/histogram.cpp src/projection.cpp src/mesh.cpp src/renderer.cpp include/MeshRenderer.h src/MeshRenderer.cpp include/PoseEstimator.h)

TARGET_LINK_LIBRARIES(histograms debug ${DEBUG_LIBRARIES})
TARGET_LINK_LIBRARIES(histograms optimized ${OPTIMIZED_LIBRARIES})
