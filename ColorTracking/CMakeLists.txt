cmake_minimum_required(VERSION 3.14)
project(ColorTracking)

set(CMAKE_CXX_STANDARD 98)

FIND_PACKAGE(OpenCV 3 REQUIRED)
FIND_PACKAGE( Boost 1.66 COMPONENTS filesystem program_options system REQUIRED )

set(DEBUG_LIBRARIES
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        /usr/local/lib/libnlopt.dylib
        ${PROJECT_SOURCE_DIR}/deps/lib/libyaml-cpp.a
        ${PROJECT_SOURCE_DIR}/../histograms/cmake-build-debug/libhistograms.a
#        ${PROJECT_SOURCE_DIR}/../Renderer/cmake-build-debug/libRenderer.a
        )

set(OPTIMIZED_LIBRARIES
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        /usr/local/lib/libnlopt.dylib
        ${PROJECT_SOURCE_DIR}/deps/lib/libyaml-cpp.a
        ${PROJECT_SOURCE_DIR}/../histograms/cmake-build-release/libhistograms.a
#        ${PROJECT_SOURCE_DIR}/../Renderer/cmake-build-release/libRenderer.a
        )

INCLUDE_DIRECTORIES(
        ${Boost_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/deps/include
        ../deps/include
        ${PROJECT_SOURCE_DIR}/../histograms/include
#        ${PROJECT_SOURCE_DIR}/../Renderer/include
        include
        tests
)

add_executable(ColorTracking main.cpp include/DataIO.h src/DataIO.cpp include/GroundTruthPoseGetter.h src/GroundTruthPoseGetter.cpp tests/tests.cpp tests/tests.h include/test_runner/argparsing.hpp include/test_runner/data.hpp include/test_runner/io.hpp include/utils/OptProcessor.hpp src/test_runner/argparsing.cpp src/test_runner/data.cpp src/test_runner/io.cpp src/utils/OptProcessor.cpp include/SLSQPPoseGetter.h src/SLSQPPoseGetter.cpp include/PoseGetter.h src/GradientEstimator.cpp include/GradientEstimator.h src/GaussNewtonPoseGetter.cpp include/GaussNewtonPoseGetter.h src/GradientHessianEstimator.cpp include/GradientHessianEstimator.h src/NewtonPoseGetter.cpp include/NewtonPoseGetter.h src/GradientHonestHessianEstimator.cpp include/GradientHonestHessianEstimator.h)

TARGET_LINK_LIBRARIES(ColorTracking debug ${DEBUG_LIBRARIES})
TARGET_LINK_LIBRARIES(ColorTracking optimized ${OPTIMIZED_LIBRARIES})
