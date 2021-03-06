#ifndef COLORTRACKING_NEWTONPOSEGETTER_H
#define COLORTRACKING_NEWTONPOSEGETTER_H

#include <Object3d.h>
#include "pose_getter/PoseGetter.h"
#include <PoseEstimator.h>


class NewtonPoseGetter : public PoseGetter
{
    histograms::Object3d* object3d;
    glm::mat4 initial_pose;
public:
    NewtonPoseGetter(histograms::Object3d* object3d, const glm::mat4& initial_pose);

    glm::mat4 getPose(const cv::Mat& frame, int mode);

    glm::mat4 getPose(const cv::Mat& frame);

    virtual void setInitialPose(const glm::mat4 &pose);

    ~NewtonPoseGetter() {}
};


#endif //COLORTRACKING_NEWTONPOSEGETTER_H
