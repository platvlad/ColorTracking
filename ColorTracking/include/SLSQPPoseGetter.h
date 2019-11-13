#ifndef COLORTRACKING_SLSQPPOSEGETTER_H
#define COLORTRACKING_SLSQPPOSEGETTER_H

#include <opencv2/core/mat.hpp>
#include "nlopt/nlopt.h"
#include "nlopt/nlopt.hpp"

#include <PoseEstimator.h>
#include "PoseGetter.h"

struct PassToOptimization
{
    cv::Mat frame;
    cv::Mat downsampled2;
    cv::Mat downsampled4;
    cv::Mat downsampled8;
    glm::mat4 initial_pose;
    histograms::Object3d* object;
    float delta_step[6];
    int num_iterations;
    int iteration_number;
    int mode;
};

class SLSQPPoseGetter : public PoseGetter
{
    nlopt_opt opt;
    PassToOptimization pass_to_optimization;
    float max_translation_shift;
    float max_rotation_shift;

    static double previous[6];

    static glm::mat4 params_to_transform(const double *x);

    static double getDerivativeInDirection(const histograms::Object3d &object3D,
                                           histograms::PoseEstimator &estimator,
                                           const glm::mat4 &minus_transform,
                                           const glm::mat4 &plus_transform);

    static double getDerivativeSemiAnalytically(const histograms::Object3d &object3D,
                                                histograms::PoseEstimator &estimator,
                                                const glm::mat4 &initial_pose,
                                                const double* minus_params,
                                                const double* plus_params);

    static void getGradientAnalytically(const histograms::Object3d &object3D,
                                        const glm::mat4 &initial_pose,
                                        histograms::PoseEstimator &estimator,
                                        double* grad);

    static double energy_function(unsigned n, const double *x, double *grad, void *my_func_data);

public:
    SLSQPPoseGetter(histograms::Object3d* object3d, const glm::mat4& initial_pose);
    glm::mat4 getPose(const cv::Mat& frame, int mode);
    glm::mat4 getPose(const cv::Mat& frame);
};

#endif //COLORTRACKING_SLSQPPOSEGETTER_H
