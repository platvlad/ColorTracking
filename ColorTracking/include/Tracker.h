#pragma once

#include "DataIO.h"
#include "PoseGetter.h"
#include "Object3d.h"

void plotEnergy(const histograms::Object3d& object3d, const cv::Mat3b& frame, 
    const glm::mat4& pose, int frame_number, std::string directory_name);

class Tracker
{
protected:
    DataIO data;

    virtual void processFrame(const cv::Mat3b &input, cv::Mat3b &output);

    cv::Mat3b getFrame();
    glm::mat4 getPoseOnPyramide(const cv::Mat3b &frame, PoseGetter &pose_getter, int num_levels = 3);
    glm::mat4 extrapolate(const glm::mat4 &prev_pose, const glm::mat4 &curr_pose);

public:
    Tracker(const std::string &directory_name);

    virtual void run() = 0;
};