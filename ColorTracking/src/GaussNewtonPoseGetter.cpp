//
// Created by Vladislav Platonov on 18/11/2019.
//

#include <PoseEstimator.h>
#include <GradientHessianEstimator.h>
#include "GaussNewtonPoseGetter.h"

glm::mat4 applyResultToPose(const glm::mat4& matr, const double* params);

GaussNewtonPoseGetter::GaussNewtonPoseGetter(histograms::Object3d *object3d, const glm::mat4& initial_pose) :
                                                object3d(object3d),
                                                initial_pose(initial_pose)
{

}

glm::mat4 GaussNewtonPoseGetter::getPose(const cv::Mat &frame, int mode)
{
    histograms::PoseEstimator estimator;
    double x[6] = { 0 };
    double grad[6] = { 0 };
    for (int i = 0; i < 6; ++i)
    {
        glm::mat4 transform_matrix;
        transform_matrix = applyResultToPose(initial_pose, x);
        int histo_part = 10;
        float current_value = estimator.estimateEnergy(*object3d, frame, transform_matrix, histo_part, false);
        GradientHessianEstimator::getGradient(initial_pose, estimator, grad);
        for (int j = 0; j < 6; ++j)
        {
            x[j] += 0.01 * grad[j];
        }
    }
    initial_pose = applyResultToPose(initial_pose, x);
    return initial_pose;
}

glm::mat4 GaussNewtonPoseGetter::getPose(const cv::Mat &frame)
{
    return getPose(frame, 0);
}
