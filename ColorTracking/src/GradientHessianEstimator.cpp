#include <GradientEstimator.h>
#include "GradientHessianEstimator.h"

void
GradientHessianEstimator::getGradient(const glm::mat4 &initial_pose, histograms::PoseEstimator &estimator, double* grad, double* step)
{
    for (int i = 0; i < 6; ++i)
    {
        grad[i] = 0.0;
        step[i] = 0.0;
    }
    const cv::Mat1f& derivative_const_part = estimator.getDerivativeConstPart();
    const cv::Mat1i& num_voters = estimator.getNumVoters();
    const Projection& projection = estimator.getProjection();
    const cv::Mat1f& votes_foreground = estimator.getVotesForeground();
    const cv::Mat1f& signed_distance = projection.signed_distance;
    const cv::Mat1i& nearest_labels = projection.nearest_labels;

    std::vector<cv::Mat1d> on_border_gradients = GradientEstimator::getGradientInPoint(initial_pose, estimator);

    int non_zero_pixels = 0;

    cv::Mat1d hessian = cv::Mat1d::zeros(6, 6);
    cv::Mat1d jacobian_sum = cv::Mat1f::zeros(6, 1);

    for (int row = 0; row < num_voters.rows; ++row)
    {
        for (int col = 0; col < num_voters.cols; ++col)
        {
            if (num_voters(row, col) > 0)
            {
                cv::Matx12d dPhi = GradientEstimator::getSignedDistanceGradient(signed_distance, row, col);

                cv::Mat1d on_border_gradient = on_border_gradients[nearest_labels.at<int>(row, col)];

                cv::Mat1d non_const_part = cv::Mat1d(dPhi) * on_border_gradient;

                cv::Mat1d jacobian_in_pixel = derivative_const_part(row, col) * non_const_part;

                cv::Mat1d jacobian_in_pixel_transposed = cv::Mat1d::zeros(6, 1);

                cv::transpose(jacobian_in_pixel, jacobian_in_pixel_transposed);

                float heaviside_value = projection.heaviside(row, col);

                float foreground_vote = votes_foreground(row, col) / num_voters(row, col);

                float error_in_pixel = -log(heaviside_value * foreground_vote +
                                            (1 - heaviside_value) * (1 - foreground_vote));

                jacobian_in_pixel_transposed = jacobian_in_pixel_transposed / error_in_pixel;

                hessian += jacobian_in_pixel_transposed * jacobian_in_pixel;

                jacobian_sum += jacobian_in_pixel_transposed;

                for (int i = 0; i < 6; ++i)
                {
                    grad[i] += non_const_part(0, i) * derivative_const_part(row, col);

                }
                ++non_zero_pixels;
            }
        }
    }
    cv::Mat1d hessian_inverted = cv::Mat1d::zeros(6, 6);
    cv::invert(hessian, hessian_inverted);
    cv::Mat1d step_mat = hessian_inverted * jacobian_sum;
    step_mat = -step_mat;
    for (int i = 0; i < 6; ++i)
    {
        step[i] = step_mat(i, 0);
    }
    if (non_zero_pixels > 0)
    {
        for (int i = 0; i < 6; ++i)
        {
            if (abs(grad[i]) > 1e10)
            {
                int for_break_point = 1;
            }
            grad[i] /= non_zero_pixels;
            if (abs(grad[i]) > 1e5)
            {
                int for_break_point = 1;
            }
        }
    }
}
