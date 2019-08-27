#include <GroundTruthPoseGetter.h>
#include <DataIO.h>

GroundTruthPoseGetter::GroundTruthPoseGetter(const boost::filesystem::path& ground_truth_path) : current_frame(1),
                                                                                                 path(ground_truth_path)
{
    frame_number = DataIO::getNumFrames(ground_truth_path);
}

glm::mat4 GroundTruthPoseGetter::getPose(const cv::Mat& frame)
{
    if (current_frame <= frame_number)
    {
        glm::mat4 pose = DataIO::getPose(path, current_frame);
        ++current_frame;
        return pose;
    }
    return glm::mat4();
}
