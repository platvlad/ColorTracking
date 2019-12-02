#include <GradientHonestHessianEstimator.h>
#include "NewtonPoseGetter.h"



glm::mat4 applyResultToPose(const glm::mat4& matr, const double* params);

NewtonPoseGetter::NewtonPoseGetter(histograms::Object3d *object3d, const glm::mat4 &initial_pose):
                                    object3d(object3d),
                                    initial_pose(initial_pose)
{
}

glm::mat4 NewtonPoseGetter::getPose(const cv::Mat &frame, int mode)
{
    double params[6] = { 0 };
    int iter_number = 20;
    int histo_part = 10;
    float best_value = std::numeric_limits<float>::max();
    glm::mat4 best_pose = initial_pose;
    for (int i = 0; i < iter_number; ++i)
    {
        glm::mat4 transform_matrix = applyResultToPose(initial_pose, params);
        initial_pose = transform_matrix;
        for (int j = 0; j < 6; ++j)
        {
            params[j] = 0;
        }
        histograms::PoseEstimator estimator;
        float current_value = estimator.estimateEnergy(*object3d, frame, transform_matrix, histo_part, false);
        if (current_value < best_value)
        {
            best_value = current_value;
            best_pose = initial_pose;

        }
        GradientHonestHessianEstimator hessianEstimator;
        double grad[6] = { 0 };
        double step[6] = { 0 };
        hessianEstimator.getGradient(initial_pose, estimator, grad, step, object3d);
        double param_length = sqrt(step[0] * step[0] + step[1] * step[1] + step[2] * step[2] + step[3] * step[3] + step[4] * step[4] + step[5] * step[5]);
        for (int j = 0; j < 6; ++j)
        {
            params[j] += step[j];
        }
    }
    initial_pose = best_pose;
    return best_pose;
}

glm::mat4 NewtonPoseGetter::getPose(const cv::Mat &frame)
{
    return getPose(frame, 0);
}

void NewtonPoseGetter::getGradientHessian(const histograms::PoseEstimator &estimator)
{

}
