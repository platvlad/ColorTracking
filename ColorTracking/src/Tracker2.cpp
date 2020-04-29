#include "Tracker2.h"

#include <opencv2/calib3d/calib3d.hpp>

Tracker2::Tracker2(const std::string &directory_name) : data(directory_name)
{
}

void Tracker2::equalizeHSV(const cv::Mat3b &input, cv::Mat3b &output)
{
    cv::cvtColor(input, output, CV_BGR2HSV);
    std::vector<cv::Mat1b> hsv_channels;
    cv::split(output, hsv_channels);
    //cv::equalizeHist(hsv_channels[0], hsv_channels[0]);
    //cv::equalizeHist(hsv_channels[1], hsv_channels[1]);
    cv::equalizeHist(hsv_channels[2], hsv_channels[2]);
    cv::merge(hsv_channels, output);
}

void Tracker2::processFrame(const cv::Mat3b & input, cv::Mat3b & output, bool hsv)
{
    if (!input.empty())
    {
        if (!hsv)
        {
            cv::flip(input, output, 0);
            return;
        }
        cv::Mat3b flipped_frame;
        cv::flip(input, flipped_frame, 0);
        cv::cvtColor(flipped_frame, output, CV_BGR2HSV);
        std::vector<cv::Mat1b> hsv_channels;
        cv::split(output, hsv_channels);
        //cv::equalizeHist(hsv_channels[0], hsv_channels[0]);
        //cv::equalizeHist(hsv_channels[1], hsv_channels[1]);
        cv::equalizeHist(hsv_channels[2], hsv_channels[2]);
        cv::merge(hsv_channels, output);
    }
}

void Tracker2::processFrame(const cv::Mat3b & input, cv::Mat3b & output)
{
    if (!input.empty())
    {
        cv::flip(input, output, 0);
    }
}

cv::Mat3b Tracker2::getFrame(bool hsv)
{
    cv::VideoCapture& videoCapture = data.videoCapture;
    cv::Mat3b frame;
    videoCapture >> frame;
    cv::Mat3b processed_frame;
    processFrame(frame, processed_frame, hsv);
    return processed_frame;
}

glm::mat4 Tracker2::getPoseOnPyramide(const cv::Mat3b & frame, PoseGetter & pose_getter, size_t num_levels)
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

void Tracker2::transformToParams(const glm::mat4 &pose, double* params)
{
    cv::Matx33f rot_matr(pose[0][0], pose[1][0], pose[2][0],
        pose[0][1], pose[1][1], pose[2][1],
        pose[0][2], pose[1][2], pose[2][2]);
    cv::Matx31f rot_vec;
    cv::Rodrigues(rot_matr, rot_vec);
    params[0] = rot_vec(0, 0);
    params[1] = rot_vec(1, 0);
    params[2] = rot_vec(2, 0);
    params[3] = pose[3][0];
    params[4] = pose[3][1];
    params[5] = pose[3][2];
}

glm::mat4 Tracker2::paramsToTransform(double* params)
{
    cv::Matx31f rot_vec(params[0], params[1], params[2]);
    cv::Matx33f rot_matr;
    cv::Rodrigues(rot_vec, rot_matr);
    return glm::mat4(rot_matr(0, 0), rot_matr(1, 0), rot_matr(2, 0), 0,
        rot_matr(0, 1), rot_matr(1, 1), rot_matr(2, 1), 0,
        rot_matr(0, 2), rot_matr(1, 2), rot_matr(2, 2), 0,
        params[3], params[4], params[5], 1);
}

glm::mat4 Tracker2::extrapolate(const glm::mat4 & prev_pose, const glm::mat4 & curr_pose)
{
    glm::mat4 prev_inv = glm::inverse(prev_pose);
    glm::mat4 diff = curr_pose * prev_inv;
    return diff * curr_pose;
}

glm::mat4 Tracker2::extrapolate(const glm::mat4 &prev_prev_pose, const glm::mat4 &prev_pose, const glm::mat4 &curr_pose)
{
    double prev_prev_params[6];
    double prev_params[6];
    double curr_params[6];
    transformToParams(prev_prev_pose, prev_prev_params);
    transformToParams(prev_pose, prev_params);
    transformToParams(curr_pose, curr_params);
    double next_params[6];
    for (int i = 0; i < 6; ++i)
    {
        next_params[i] = 5 * curr_params[i] / 2 + prev_prev_params[i] / 2 - 2 * prev_params[i];
    }
    return paramsToTransform(next_params);
}
