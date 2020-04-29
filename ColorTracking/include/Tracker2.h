#pragma once

#include "DataIO2.h"
#include "PoseGetter.h"
#include "Object3d.h"

void plotEnergy(const histograms::Object3d& object3d, const cv::Mat3b& frame, 
    const glm::mat4& pose, int frame_number, std::string directory_name);

class Tracker2
{
protected:
    DataIO2 data;

    void equalizeHSV(const cv::Mat3b &input, cv::Mat3b &output);

    void processFrame(const cv::Mat3b & input, cv::Mat3b & output, bool hsv);

    virtual void processFrame(const cv::Mat3b &input, cv::Mat3b &output);

    void transformToParams(const glm::mat4 &pose, double* params);

    glm::mat4 paramsToTransform(double* params);

    cv::Mat3b getFrame(bool hsv = false);
    glm::mat4 getPoseOnPyramide(const cv::Mat3b &frame, PoseGetter &pose_getter, size_t num_levels = 3);
    glm::mat4 extrapolate(const glm::mat4 &prev_pose, const glm::mat4 &curr_pose);
    glm::mat4 extrapolate(const glm::mat4 &prev_prev_pose, const glm::mat4 &prev_pose, const glm::mat4 &curr_pose);

public:
    Tracker2(const std::string &directory_name);

    virtual void run() = 0;
};