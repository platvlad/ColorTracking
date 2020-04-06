cmake_minimum_required(VERSION 3.12)
project(ColorTracking)

set(CMAKE_CXX_STANDARD 17)

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
        debug ${PROJECT_SOURCE_DIR}/deps/yaml-cpp-master/build/Debug/yaml-cppd.lib
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
        optimized ${PROJECT_SOURCE_DIR}/deps/yaml-cpp-master/build/Release/yaml-cpp.lib
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
        ${PROJECT_SOURCE_DIR}/deps/yaml-cpp-master/include
        ../deps/include
        ${PROJECT_SOURCE_DIR}/../histograms/include
        ${PROJECT_SOURCE_DIR}/../lkt-tracker-library/include
        include
        tests
)

add_executable(ColorTracking 
               main.cpp 
               tests/tests.h tests/tests.cpp 
               include/DataIO.h src/DataIO.cpp 
               include/GeomUtils.h src/GeomUtils.cpp 
               include/Feature3DInfoList.h src/Feature3DInfoList.cpp 
               include/estimator/GradientEstimator.h src/estimator/GradientEstimator.cpp 
               include/estimator/GradientHonestHessianEstimator.h src/estimator/GradientHonestHessianEstimator.cpp 
               include/estimator/GradientHessianEstimator.h src/estimator/GradientHessianEstimator.cpp 
               include/test_runner/argparsing.hpp src/test_runner/argparsing.cpp
               include/test_runner/data.hpp src/test_runner/data.cpp
               include/test_runner/io.hpp src/test_runner/io.cpp
               include/pose_getter/GroundTruthPoseGetter.h src/pose_getter/GroundTruthPoseGetter.cpp 
               include/pose_getter/SLSQPPoseGetter.h src/pose_getter/SLSQPPoseGetter.cpp 
               include/pose_getter/SlsqpLktPoseGetter.h src/pose_getter/SlsqpLktPoseGetter.cpp 
               include/pose_getter/PoseGetter.h 
               include/pose_getter/GaussNewtonPoseGetter.h src/pose_getter/GaussNewtonPoseGetter.cpp 
               include/pose_getter/NewtonPoseGetter.h src/pose_getter/NewtonPoseGetter.cpp 
               include/pose_getter/LevenbergPoseGetter.h src/pose_getter/LevenbergPoseGetter.cpp 
               include/pose_getter/LkPoseGetter.h src/pose_getter/LkPoseGetter.cpp 
               include/tracker/Tracker.h src/tracker/Tracker.cpp
               include/tracker/PyramideTracker.h
               include/tracker/LktInitTracker.h src/tracker/LktInitTracker.cpp
               include/tracker/GroundTruthTracker.h src/tracker/GroundTruthTracker.cpp
               include/tracker/SlsqpLktTracker.h src/tracker/SlsqpLktTracker.cpp
               include/tracker/LktTracker.h src/tracker/LktTracker.cpp
               include/utils/OptProcessor.hpp src/utils/OptProcessor.cpp 
              )
              
set(LINK_LIBRARIES optimized ${OPTIMIZED_LIBRARIES} debug ${DEBUG_LIBRARIES})

TARGET_LINK_LIBRARIES(ColorTracking ${LINK_LIBRARIES})
#TARGET_LINK_LIBRARIES(ColorTracking debug ${DEBUG_LIBRARIES})
#TARGET_LINK_LIBRARIES(ColorTracking optimized ${OPTIMIZED_LIBRARIES})
