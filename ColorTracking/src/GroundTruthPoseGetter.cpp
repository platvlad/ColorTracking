#include <GroundTruthPoseGetter.h>
#include <InputReader.h>

GroundTruthPoseGetter::GroundTruthPoseGetter(const boost::filesystem::path& ground_truth_path) : current_frame(1),
                                                                                                 path(ground_truth_path)
{
    frame_number = InputReader::getNumFrames(ground_truth_path);
}

glm::mat4 GroundTruthPoseGetter::getPose()
{
    if (current_frame <= frame_number)
    {
        glm::mat4 pose = InputReader::getPose(path);
        ++current_frame;
        return pose;
    }
    return glm::mat4();
}
