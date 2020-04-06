cmake_minimum_required(VERSION 3.12)
project(histograms)

set(CMAKE_CXX_STANDARD 17)

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

add_library(histograms 
            include/Object3d.h src/Object3d.cpp 
            include/PoseEstimator.h src/PoseEstimator.cpp 
            include/CircleWindow.h src/CircleWindow.cpp 
            include/histogram.h src/histogram.cpp 
            include/projection.h src/projection.cpp 
            include/mesh.h src/mesh.cpp 
            include/renderer.h src/renderer.cpp 
            include/MeshRenderer.h src/MeshRenderer.cpp)

TARGET_LINK_LIBRARIES(histograms debug ${DEBUG_LIBRARIES})
TARGET_LINK_LIBRARIES(histograms optimized ${OPTIMIZED_LIBRARIES})
