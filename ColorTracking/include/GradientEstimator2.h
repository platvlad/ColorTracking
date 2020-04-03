#ifndef COLORTRACKING_GRADIENTESTIMATOR2_H
#define COLORTRACKING_GRADIENTESTIMATOR2_H

#include <glm/ext/matrix_float4x4.hpp>
#include <PoseEstimator2.h>

class GradientEstimator2
{
public:
    static std::vector<cv::Mat1d> getGradientInPoint(const glm::mat4 &initial_pose,
                                                     const histograms::PoseEstimator2 &estimator);

    static cv::Matx12d getSignedDistanceGradient(const cv::Mat1f &signed_distance, int row, int col);

    static void getGradient(const glm::mat4 &initial_pose,
                            histograms::PoseEstimator2 &estimator,
                            double* grad);


};


#endif //COLORTRACKING_GRADIENTESTIMATOR2_H
