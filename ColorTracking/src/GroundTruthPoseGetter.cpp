#include <GroundTruthPoseGetter.h>
#include <DataIO.h>

GroundTruthPoseGetter::GroundTruthPoseGetter(const DataIO &dataIO) : current_frame(1),
                                                                     dataIO(&dataIO)
{
    frame_number = dataIO.getNumFrames();
}

glm::mat4 GroundTruthPoseGetter::getPose(const cv::Mat& frame)
{
    if (current_frame <= frame_number)
    {
        glm::mat4 pose = dataIO->getPose(current_frame);
        ++current_frame;
        return pose;
    }
    return glm::mat4();
}

glm::mat4 GroundTruthPoseGetter::getPose(int frame_index)
{
    if (frame_index > 0) {
        return dataIO->getPose(frame_index);
    }
    return glm::mat4();
}

glm::mat4 GroundTruthPoseGetter::getPose(const cv::Mat &frame, int)
{
    return getPose(frame);
}
