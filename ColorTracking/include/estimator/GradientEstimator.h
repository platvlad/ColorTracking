#ifndef COLORTRACKING_GRADIENTESTIMATOR_H
#define COLORTRACKING_GRADIENTESTIMATOR_H

#include <glm/ext/matrix_float4x4.hpp>
#include <PoseEstimator.h>

class GradientEstimator
{
public:
    static std::vector<cv::Mat1d> getGradientInPoint(const glm::mat4 &initial_pose,
                                                     const histograms::PoseEstimator &estimator);

    static cv::Matx12d getSignedDistanceGradient(const cv::Mat1f &signed_distance, int row, int col);

    static void getGradient(const glm::mat4 &initial_pose,
                            histograms::PoseEstimator &estimator,
                            double* grad);


};


#endif //COLORTRACKING_GRADIENTESTIMATOR_H
