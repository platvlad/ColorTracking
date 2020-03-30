#include "Tracker.h"

Tracker::Tracker(const std::string &directory_name) : data(directory_name)
{
}

void Tracker::processFrame(const cv::Mat3b & input, cv::Mat3b & output)
{
    if (!input.empty())
    {
        cv::flip(input, output, 0);
    }
}

cv::Mat3b Tracker::getFrame()
{
    cv::VideoCapture& videoCapture = data.videoCapture;
    cv::Mat3b frame;
    videoCapture >> frame;
    cv::Mat3b processed_frame;
    processFrame(frame, processed_frame);
    return processed_frame;
}

glm::mat4 Tracker::getPoseOnPyramide(const cv::Mat3b & frame, PoseGetter & pose_getter, int num_levels)
{
    std::vector<cv::Mat3b> pyramide(num_levels);
    if (num_levels > 0)
    {
        cv::pyrDown(frame, pyramide[0]);
    }
    for (int i = 1; i < pyramide.size(); ++i)
    {
        cv::pyrDown(pyramide[i - 1], pyramide[i]);
    }
    for (int i = pyramide.size() - 1; i >= 0; --i)
    {
        pose_getter.getPose(pyramide[i], i + 1);
    }
    return pose_getter.getPose(frame, 0);
}

glm::mat4 Tracker::extrapolate(const glm::mat4 & prev_pose, const glm::mat4 & curr_pose)
{
    glm::mat4 prev_inv = glm::inverse(prev_pose);
    glm::mat4 diff = curr_pose * prev_inv;
    return diff * curr_pose;
}
