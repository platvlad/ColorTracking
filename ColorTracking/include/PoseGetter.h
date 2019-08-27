#ifndef COLORTRACKING_POSEGETTER_H
#define COLORTRACKING_POSEGETTER_H

#include <glm/mat4x4.hpp>
#include <opencv2/core/mat.hpp>

class PoseGetter
{
public:
    PoseGetter() {};
    virtual glm::mat4 getPose(const cv::Mat& frame) = 0;
};

#endif //COLORTRACKING_POSEGETTER_H
