cmake_minimum_required(VERSION 3.12)
project(ColorTracking)

set(CMAKE_CXX_STANDARD 98)

set(Boost_USE_STATIC_LIBS ON)
set(BOOST_ROOT ${PROJECT_SOURCE_DIR}/../deps/boost_1_72_0)

FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(glm REQUIRED)
FIND_PACKAGE( Boost 1.72 COMPONENTS filesystem program_options system REQUIRED )
FIND_PACKAGE(NLopt REQUIRED)


set(DEBUG_LIBRARIES
        ${OpenCV_LIBS}
        ${OPENGL_LIBRARIES}
        ${glew_LIBRARIES}
        ${GLM_LIBS}
        ${Boost_LIBRARIES}
        ${NLOPT_LIBRARIES}
        debug ${PROJECT_SOURCE_DIR}/deps/yaml-cpp-release-0.3.0/build/Debug/libyaml-cppmdd.lib
        debug ${PROJECT_SOURCE_DIR}/../histograms/build/Debug/histograms.lib
        debug ${PROJECT_SOURCE_DIR}/../lkt-tracker-library/build/Debug/lkttracker.lib
        debug ${PROJECT_SOURCE_DIR}/../deps/glew-2.1.0/build/vc15/lib/Debug/glew32d.lib
        )

set(OPTIMIZED_LIBRARIES
        ${OpenCV_LIBS}
        ${OPENGL_LIBRARIES}
        ${glew_LIBRARIES}
        ${GLM_LIBS}
        ${Boost_LIBRARIES}
        ${NLOPT_LIBRARIES}
        optimized ${PROJECT_SOURCE_DIR}/deps/yaml-cpp-release-0.3.0/build/Release/libyaml-cppmd.lib
        optimized ${PROJECT_SOURCE_DIR}/../histograms/build/Release/histograms.lib
        optimized ${PROJECT_SOURCE_DIR}/../lkt-tracker-library/build/Release/lkttracker.lib
        optimized ${PROJECT_SOURCE_DIR}/../deps/glew-2.1.0/build/vc15/lib/Release/glew32.lib
        )

INCLUDE_DIRECTORIES(
        ${OPENGL_INCLUDE_DIR}
        ${glew_INCLUDE_DIR}
        ${Boost_INCLUDE_DIR}
        ${GLM_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/deps/include
        ${NLOPT_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/deps/yaml-cpp-release-0.3.0/include
        ../deps/include
        ${PROJECT_SOURCE_DIR}/../histograms/include
        ${PROJECT_SOURCE_DIR}/../lkt-tracker-library/include
        include
        tests
)

add_executable(ColorTracking 
               main.cpp 
               include/DataIO.h src/DataIO.cpp 
               include/GroundTruthPoseGetter.h src/GroundTruthPoseGetter.cpp 
               tests/tests.h tests/tests.cpp 
               include/test_runner/argparsing.hpp src/test_runner/argparsing.cpp
               include/test_runner/data.hpp src/test_runner/data.cpp
               include/test_runner/io.hpp src/test_runner/io.cpp
               include/utils/OptProcessor.hpp src/utils/OptProcessor.cpp 
               include/SLSQPPoseGetter.h src/SLSQPPoseGetter.cpp 
               include/SlsqpLktPoseGetter.h src/SlsqpLktPoseGetter.cpp 
               include/PoseGetter.h 
               include/GradientEstimator.h src/GradientEstimator.cpp 
               include/GaussNewtonPoseGetter.h src/GaussNewtonPoseGetter.cpp 
               include/GradientHessianEstimator.h src/GradientHessianEstimator.cpp 
               include/NewtonPoseGetter.h src/NewtonPoseGetter.cpp 
               include/GradientHonestHessianEstimator.h src/GradientHonestHessianEstimator.cpp 
               include/LevenbergPoseGetter.h src/LevenbergPoseGetter.cpp 
               include/LkPoseGetter.h src/LkPoseGetter.cpp 
               include/Feature3DInfoList.h src/Feature3DInfoList.cpp 
               include/GeomUtils.h src/GeomUtils.cpp 
               include/Tracker.h src/Tracker.cpp
               include/PyramideTracker.h
               include/LktInitTracker.h src/LktInitTracker.cpp
               include/GroundTruthTracker.h src/GroundTruthTracker.cpp
               include/SlsqpLktTracker.h src/SlsqpLktTracker.cpp
               include/LktTracker.h src/LktTracker.cpp
              )
              
set(LINK_LIBRARIES optimized ${OPTIMIZED_LIBRARIES} debug ${DEBUG_LIBRARIES})

TARGET_LINK_LIBRARIES(ColorTracking ${LINK_LIBRARIES})
#TARGET_LINK_LIBRARIES(ColorTracking debug ${DEBUG_LIBRARIES})
#TARGET_LINK_LIBRARIES(ColorTracking optimized ${OPTIMIZED_LIBRARIES})
