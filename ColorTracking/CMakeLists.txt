cmake_minimum_required(VERSION 3.12)
project(ColorTracking)

set(CMAKE_CXX_STANDARD 98)

set(Boost_USE_STATIC_LIBS ON)
set(BOOST_ROOT ${PROJECT_SOURCE_DIR}/../deps/boost_1_72_0)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(glm REQUIRED)
FIND_PACKAGE( Boost 1.72 COMPONENTS filesystem program_options system REQUIRED )
FIND_PACKAGE(NLopt REQUIRED)


set(DEBUG_LIBRARIES
        ${OpenCV_LIBS}
        ${GLM_LIBS}
        ${Boost_LIBRARIES}
        ${NLOPT_LIBRARIES}
        debug ${PROJECT_SOURCE_DIR}/deps/yaml-cpp-release-0.3.0/build/Debug/libyaml-cppmdd.lib
        debug ${PROJECT_SOURCE_DIR}/../histograms/build/Debug/histograms.lib
        debug ${PROJECT_SOURCE_DIR}/../lkt-tracker-library/build/Debug/lkttracker.lib
        )

set(OPTIMIZED_LIBRARIES
        ${OpenCV_LIBS}
        ${GLM_LIBS}
        ${Boost_LIBRARIES}
        ${NLOPT_LIBRARIES}
        optimized ${PROJECT_SOURCE_DIR}/deps/yaml-cpp-release-0.3.0/build/Release/libyaml-cppmd.lib
        optimized ${PROJECT_SOURCE_DIR}/../histograms/build/Release/histograms.lib
        optimized ${PROJECT_SOURCE_DIR}/../lkt-tracker-library/build/Release/lkttracker.lib
        )

INCLUDE_DIRECTORIES(
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
               tests/tests.cpp tests/tests.h 
               include/test_runner/argparsing.hpp src/test_runner/argparsing.cpp
               include/test_runner/data.hpp src/test_runner/data.cpp
               include/test_runner/io.hpp src/test_runner/io.cpp
               include/utils/OptProcessor.hpp src/utils/OptProcessor.cpp 
               include/SLSQPPoseGetter.h src/SLSQPPoseGetter.cpp 
               include/PoseGetter.h 
               src/GradientEstimator.cpp include/GradientEstimator.h 
               src/GaussNewtonPoseGetter.cpp include/GaussNewtonPoseGetter.h 
               src/GradientHessianEstimator.cpp include/GradientHessianEstimator.h 
               src/NewtonPoseGetter.cpp include/NewtonPoseGetter.h 
               src/GradientHonestHessianEstimator.cpp include/GradientHonestHessianEstimator.h 
               src/LevenbergPoseGetter.cpp include/LevenbergPoseGetter.h 
               src/FeatureTracker.cpp include/FeatureTracker.h 
               include/GeomUtils.h src/GeomUtils.cpp
              )
              
set(LINK_LIBRARIES optimized ${OPTIMIZED_LIBRARIES} debug ${DEBUG_LIBRARIES})

TARGET_LINK_LIBRARIES(ColorTracking ${LINK_LIBRARIES})
#TARGET_LINK_LIBRARIES(ColorTracking debug ${DEBUG_LIBRARIES})
#TARGET_LINK_LIBRARIES(ColorTracking optimized ${OPTIMIZED_LIBRARIES})
