#ifndef COLORTRACKING_NEWTONPOSEGETTER_H
#define COLORTRACKING_NEWTONPOSEGETTER_H

#include <Object3d.h>
#include "PoseGetter.h"
#include <PoseEstimator.h>


class NewtonPoseGetter : public PoseGetter
{
    histograms::Object3d* object3d;
    glm::mat4 initial_pose;
public:
    NewtonPoseGetter(histograms::Object3d* object3d, const glm::mat4& initial_pose);

    glm::mat4 getPose(const cv::Mat& frame, int mode);

    glm::mat4 getPose(const cv::Mat& frame);

    void getGradientHessian(const histograms::PoseEstimator& estimator);
};


#endif //COLORTRACKING_NEWTONPOSEGETTER_H
