#ifndef TESTRUNNER_ARGPARSING_HPP
#define TESTRUNNER_ARGPARSING_HPP

#include <string>

#include <boost/optional.hpp>
#include <boost/filesystem/path.hpp>

namespace testrunner
{

struct TrackingConfig
{
    boost::filesystem::path results;
    boost::filesystem::path camera;
    boost::filesystem::path groundTruth;
    boost::filesystem::path mesh;
    boost::filesystem::path precalc;
    bool experimental;
    bool backTracking;
    bool toFrame;
    boost::optional<boost::filesystem::path> fixations;
    boost::optional<boost::filesystem::path> rgb;
};


boost::optional<TrackingConfig> parseTrackingArguments(
        int argc, char *argv[]);


struct PoseEstimationConfig
{
    boost::filesystem::path results;
    boost::filesystem::path camera;
    boost::filesystem::path mesh;
    boost::filesystem::path rgbDir;
    boost::filesystem::path initialPoses;
    boost::filesystem::path trainRgbDir;
    boost::filesystem::path trainGroundTruth;
};


boost::optional<PoseEstimationConfig> parsePoseEstimationArguments(
        int argc, char *argv[]);


struct ErrorCalculationConfig
{
    boost::filesystem::path errors;
    boost::filesystem::path mesh;
    boost::filesystem::path diameter;
    boost::filesystem::path groundTruth;
    boost::filesystem::path estimate;
};


boost::optional<ErrorCalculationConfig> parseErrorCalculationArguments(
        int argc, char *argv[]);


struct DiameterCalculationConfig
{
    boost::filesystem::path mesh;
    boost::filesystem::path diameter;
};


boost::optional<DiameterCalculationConfig> parseDiameterCalculationArguments(
        int argc, char *argv[]);

}
// namespace testrunner

#endif // TESTRUNNER_ARGPARSING_HPP
