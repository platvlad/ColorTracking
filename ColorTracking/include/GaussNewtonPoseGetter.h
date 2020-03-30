#ifndef COLORTRACKING_GAUSSNEWTONPOSEGETTER_H
#define COLORTRACKING_GAUSSNEWTONPOSEGETTER_H


#include <Object3d.h>
#include "PoseGetter.h"

class GaussNewtonPoseGetter : public PoseGetter
{
    histograms::Object3d* object3d;
    glm::mat4 initial_pose;

    void plot2Points(const cv::Mat &frame, const glm::mat4 &init, double* params, const std::string &file_name);

public:
    GaussNewtonPoseGetter(histograms::Object3d* object3d, const glm::mat4& initial_pose);

    glm::mat4 getPose(const cv::Mat& frame, int mode);

    glm::mat4 getPose(const cv::Mat& frame);

    virtual void setInitialPose(const glm::mat4 &pose);

    ~GaussNewtonPoseGetter() {}
};


#endif //COLORTRACKING_GAUSSNEWTONPOSEGETTER_H
