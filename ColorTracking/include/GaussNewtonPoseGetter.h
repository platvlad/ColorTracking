#ifndef COLORTRACKING_GAUSSNEWTONPOSEGETTER_H
#define COLORTRACKING_GAUSSNEWTONPOSEGETTER_H


#include <Object3d.h>
#include "PoseGetter.h"

class GaussNewtonPoseGetter : public PoseGetter
{
    histograms::Object3d* object3d;
    glm::mat4 initial_pose;
public:
    GaussNewtonPoseGetter(histograms::Object3d* object3d, const glm::mat4& initial_pose);

    glm::mat4 getPose(const cv::Mat& frame, int mode);

    glm::mat4 getPose(const cv::Mat& frame);

    ~GaussNewtonPoseGetter() {}
};


#endif //COLORTRACKING_GAUSSNEWTONPOSEGETTER_H
