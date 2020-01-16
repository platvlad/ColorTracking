#ifndef COLORTRACKING_LEVENBERGPOSEGETTER_H
#define COLORTRACKING_LEVENBERGPOSEGETTER_H


#include <Object3d.h>
#include <PoseEstimator.h>
#include "PoseGetter.h"

class LevenbergPoseGetter: public PoseGetter
{
    histograms::Object3d* object3d;
    glm::mat4 initial_pose;
public:
    LevenbergPoseGetter(histograms::Object3d* object3d, const glm::mat4& initial_pose);

    glm::mat4 getPose(const cv::Mat& frame, int mode);

    glm::mat4 getPose(const cv::Mat& frame);

    void getGradientHessian(const histograms::PoseEstimator& estimator);

    virtual void setInitialPose(const glm::mat4 &pose);

    ~LevenbergPoseGetter() {}
};


#endif //COLORTRACKING_LEVENBERGPOSEGETTER_H
