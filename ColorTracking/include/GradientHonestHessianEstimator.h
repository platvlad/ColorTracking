#ifndef COLORTRACKING_GRADIENTHONESTHESSIANESTIMATOR_H
#define COLORTRACKING_GRADIENTHONESTHESSIANESTIMATOR_H


#include <glm/ext/matrix_float4x4.hpp>
#include <PoseEstimator.h>

class GradientHonestHessianEstimator
{
    std::vector<cv::Mat1d> B3B4;

    std::vector<cv::Mat1d> B2B3B4;

    std::vector<cv::Matx23d> Magic[6];

    std::vector<cv::Mat1d> getGradientInPoint(const glm::mat4 &initial_pose,
                                                     const histograms::PoseEstimator &estimator);

    cv::Matx12d getSignedDistanceGradient(const cv::Mat1f &signed_distance, int row, int col);

public:
    void getGradient(const glm::mat4 &initial_pose,
                            histograms::PoseEstimator &estimator,
                            double* grad,
                            double* step,
                            histograms::Object3d* object_unused);
};


#endif //COLORTRACKING_GRADIENTHONESTHESSIANESTIMATOR_H
