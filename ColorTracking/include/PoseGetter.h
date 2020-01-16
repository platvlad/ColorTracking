#ifndef COLORTRACKING_POSEGETTER_H
#define COLORTRACKING_POSEGETTER_H

#include <glm/mat4x4.hpp>
#include <opencv2/core/mat.hpp>

class PoseGetter
{
public:
    PoseGetter() {};
    virtual glm::mat4 getPose(const cv::Mat& frame) = 0;
    virtual glm::mat4 getPose(const cv::Mat& frame, int mode) = 0;
    virtual void setInitialPose(const glm::mat4 &pose) = 0;
    virtual ~PoseGetter() {}
};

#endif //COLORTRACKING_POSEGETTER_H
