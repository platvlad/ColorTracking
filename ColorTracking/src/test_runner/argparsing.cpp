#include <test_runner/argparsing.hpp>

#include <utils/OptProcessor.hpp>

namespace testrunner
{

boost::optional<TrackingConfig> parseTrackingArguments(
        int argc, char *argv[])
{
    using boost::program_options::value;
    using boost::program_options::bool_switch;
    using boost::filesystem::path;
    using boost::optional;

    TrackingConfig config;
    utils::OptProcessor optProcessor;
    optProcessor.addOptions()
            ("results,o", value<path>(&config.results)->required(), "Results destination path.")
            ("camera,c", value<path>(&config.camera)->required(), "Camera description path.")
            ("ground-truth,g", value<path>(&config.groundTruth)->required(), "Ground truth poses path.")
            ("mesh,m", value<path>(&config.mesh)->required(), "Object model path.")
            ("precalc,p", value<path>(&config.precalc)->required(), "Analysis file path.")
            ("experimental,e", bool_switch(&config.experimental), "Experimental tracking.")
            ("back-tracking,b", bool_switch(&config.backTracking), "Back tracking (ignored in refine).")
            ("to-frame,t", bool_switch(&config.toFrame), "Tracking to frame (ignored in refine).")
            ("fixations,f", value<optional<path> >(&config.fixations), "DoF fixations description path (optional).")
            ("rgb,v", value<optional<path> >(&config.rgb), "Input video path (optional).");

    utils::OptProcessor::Result const res = optProcessor.process(argc, argv);
    if (res.shutdown) {
        return boost::none;
    }
    return config;
}


boost::optional<PoseEstimationConfig> parsePoseEstimationArguments(
        int argc, char *argv[])
{
    using boost::program_options::value;
    using boost::filesystem::path;

    PoseEstimationConfig config;
    utils::OptProcessor optProcessor;
    optProcessor.addOptions()
            ("results,o", value<path>(&config.results)->required(), "Results destination path.")
            ("camera,c", value<path>(&config.camera)->required(), "Camera description path.")
            ("mesh,m", value<path>(&config.mesh)->required(), "Object model path.")
            ("rgb-dir,i", value<path>(&config.rgbDir)->required(), "Input frame sequence dir.")
            ("initial-poses,p", value<path>(&config.initialPoses)->required(), "Initial poses path.")
            ("train-rgb-dir,k", value<path>(&config.trainRgbDir)->required(), "Train frame sequence dir.")
            ("train-ground-truth,t", value<path>(&config.trainGroundTruth)->required(), "Train ground truth poses path.");

    utils::OptProcessor::Result const res = optProcessor.process(argc, argv);
    if (res.shutdown) {
        return boost::none;
    }
    return config;

}


boost::optional<ErrorCalculationConfig> parseErrorCalculationArguments(
        int argc, char *argv[])
{
    using boost::program_options::value;
    using boost::filesystem::path;
    using boost::optional;

    ErrorCalculationConfig config;
    utils::OptProcessor optProcessor;
    optProcessor.addOptions()
            ("errors,o", value<path>(&config.errors)->required(), "Errors destination path.")
            ("mesh,m", value<path>(&config.mesh)->required(), "Object model path.")
            ("diameter,d", value<path>(&config.diameter)->required(), "Object model diameter path.")
            ("ground-truth,g", value<path>(&config.groundTruth)->required(), "Ground truth poses path.")
            ("estimate,e", value<path>(&config.estimate)->required(), "Estimated poses path.");

    utils::OptProcessor::Result const res = optProcessor.process(argc, argv);
    if (res.shutdown) {
        return boost::none;
    }
    return config;
}


boost::optional<DiameterCalculationConfig> parseDiameterCalculationArguments(
        int argc, char *argv[])
{
    using boost::program_options::value;
    using boost::filesystem::path;

    DiameterCalculationConfig config;
    utils::OptProcessor optProcessor;
    optProcessor.addOptions()
            ("mesh,m", value<path>(&config.mesh)->required(), "Object model path.")
            ("diameter,d", value<path>(&config.diameter)->required(), "Calculated diameter path.");

    utils::OptProcessor::Result const res = optProcessor.process(argc, argv);
    if (res.shutdown) {
        return boost::none;
    }
    return config;
}

}
