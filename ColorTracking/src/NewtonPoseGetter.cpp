#include <GradientHonestHessianEstimator.h>
#include "NewtonPoseGetter.h"



glm::mat4 applyResultToPose(const glm::mat4& matr, const double* params);

void plotEnergy(const histograms::Object3d& object3d, const cv::Mat3b& frame, const glm::mat4& pose, int frame_number);

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
    double least_params = std::numeric_limits<float>::max();
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
        bool correct_hessian = hessianEstimator.getGradient(initial_pose, estimator, grad, step, object3d);
        double param_length = sqrt(step[0] * step[0] + step[1] * step[1] + step[2] * step[2] + step[3] * step[3] + step[4] * step[4] + step[5] * step[5]);
        double grad_length = sqrt(grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2] + grad[3] * grad[3] + grad[4] * grad[4] + grad[5] * grad[5]);
        if (param_length > 1e-6 && param_length < least_params)
        {
            least_params = param_length;
        }
        else if (param_length >= least_params)
        {
            i = std::max(i, iter_number - 3);
        }
        if (param_length > 100)
        {
            continue;
        }
        double quotient[6];
        if (correct_hessian)
        {
            for (int j = 0; j < 6; ++j)
            {
                quotient[j] = grad[j] / step[j];
                params[j] += step[j];
            }
        }
        else
        {
            for (int j = 0; j < 3; ++j)
            {
                params[j] -= 0.005 * grad[j];
            }
            for (int j = 3; j < 6; ++j)
            {
                params[j] -= 0.005 * grad[j];
            }
        }
        bool unused = false;
    }
    initial_pose = best_pose;
    return best_pose;
}

glm::mat4 NewtonPoseGetter::getPose(const cv::Mat &frame)
{
    return getPose(frame, 0);
}

void NewtonPoseGetter::setInitialPose(const glm::mat4 &pose)
{
    initial_pose = pose;
}

