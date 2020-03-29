#ifndef COLORTRACKING_GRADIENTHESSIANESTIMATOR_H
#define COLORTRACKING_GRADIENTHESSIANESTIMATOR_H


#include <glm/ext/matrix_float4x4.hpp>
#include <PoseEstimator.h>

class GradientHessianEstimator
{
public:
    static void getGradient(const glm::mat4 &initial_pose,
                            histograms::PoseEstimator &estimator,
                            double* grad,
                            double* step);
};


#endif //COLORTRACKING_GRADIENTHESSIANESTIMATOR_H
