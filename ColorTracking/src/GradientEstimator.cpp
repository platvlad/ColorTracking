#include "GradientEstimator.h"

#include <iostream>

std::vector<cv::Mat1d> GradientEstimator::getGradientInPoint(const glm::mat4 &initial_pose,
                                                             const histograms::PoseEstimator &estimator)
{
    const Projection& projection = estimator.getProjection();
    const cv::Mat1i& nearest_labels = projection.nearest_labels;
    const cv::Mat3f& depth_map = projection.depth_map;

    const glm::mat4& transform = estimator.getPose();
    glm::mat4 transform_inverted = glm::inverse(transform);

    const Renderer* renderer = estimator.getRenderer();
    glm::vec2 focal = renderer->getFocal();

    cv::Matx33d dInitF = cv::Matx33d(initial_pose[0][0], initial_pose[1][0], initial_pose[2][0],
                                     initial_pose[0][1], initial_pose[1][1], initial_pose[2][1],
                                     initial_pose[0][2], initial_pose[1][2], initial_pose[2][2]);

    std::vector<cv::Mat1d> gradients;
    gradients.push_back(cv::Mat1d());

    for (int row = 0; row < projection.mask.rows; ++row)
    {
        for (int col = 0; col < projection.mask.cols; ++col)
        {
            if (projection.signed_distance(row, col) >= -1.5 && projection.signed_distance(row, col) < 0)
            {
                int label = nearest_labels(row, col);
                cv::Vec2i pixel_on_border = cv::Vec2i(row, col);

                if (label != gradients.size())
                {
                    std::cout << "Achtung!" << std::endl;
                }
                cv::Vec3f pt_in_3d = depth_map(pixel_on_border[0], pixel_on_border[1]);
                pt_in_3d[2] = -pt_in_3d[2];
                glm::vec4 vec_in_3d = glm::vec4(pt_in_3d[0], pt_in_3d[1], pt_in_3d[2], 1);
                glm::vec4 vec_on_model = transform_inverted * vec_in_3d;
                cv::Matx23d dPi(-focal.x / pt_in_3d[2], 0, pt_in_3d[0] * focal.x / (pt_in_3d[2] * pt_in_3d[2]),
                                0, -focal.y / pt_in_3d[2], pt_in_3d[1] * focal.y / (pt_in_3d[2] * pt_in_3d[2]));

                cv::Mat1d dX = cv::Mat1d::zeros(3, 6);
                dX(0, 1) = vec_on_model[2];
                dX(0, 2) = -vec_on_model[1];
                dX(0, 3) = 1;
                dX(1, 0) = -vec_on_model[2];
                dX(1, 2) = vec_on_model[0];
                dX(1, 4) = 1;
                dX(2, 0) = vec_on_model[1];
                dX(2, 1) = -vec_on_model[0];
                dX(2, 5) = 1;

                gradients.insert(gradients.begin() + label, dPi * dInitF * dX);
            }
        }
    }
    return gradients;
}

cv::Matx12d GradientEstimator::getSignedDistanceGradient(const cv::Mat1f &signed_distance, int row, int col)
{
    double dPhi_dx = 0;
    double dPhi_dy = 0;
    if (col == 0)
    {
        dPhi_dx = signed_distance(row, col + 1) - signed_distance(row, col);
    }
    else if (col == signed_distance.cols - 1)
    {
        dPhi_dx = signed_distance(row, col) - signed_distance(row, col - 1);
    }
    else
    {
        dPhi_dx = (signed_distance(row, col + 1) - signed_distance(row, col - 1)) / 2;
    }
    if (row == 0)
    {
        dPhi_dy = signed_distance(row, col) - signed_distance(row + 1, col);
    }
    else if (row == signed_distance.rows - 1)
    {
        dPhi_dy = signed_distance(row - 1, col) - signed_distance(row, col);
    }
    else
    {
        dPhi_dy = (signed_distance(row - 1, col) - signed_distance(row + 1, col)) / 2;
    }
    return cv::Matx12d(dPhi_dx, dPhi_dy);
}

void
GradientEstimator::getGradient(const glm::mat4 &initial_pose,
                               histograms::PoseEstimator &estimator,
                               double* grad)
{
    for (int i = 0; i < 6; ++i)
    {
        grad[i] = 0.0;
    }
    const cv::Mat1f& derivative_const_part = estimator.getDerivativeConstPart();
    const cv::Mat1i& num_voters = estimator.getNumVoters();
    const Projection& projection = estimator.getProjection();
    const cv::Mat1f& signed_distance = projection.signed_distance;
    const cv::Mat1i& nearest_labels = projection.nearest_labels;

    std::vector<cv::Mat1d> on_border_gradients = getGradientInPoint(initial_pose, estimator);

    int non_zero_pixels = 0;

    for (int row = 0; row < num_voters.rows; ++row)
    {
        for (int col = 0; col < num_voters.cols; ++col)
        {
            if (num_voters(row, col) > 0)
            {
                cv::Matx12d dPhi = getSignedDistanceGradient(signed_distance, row, col);

                cv::Mat1d on_border_gradient = on_border_gradients[nearest_labels.at<int>(row, col)];

                cv::Mat1d non_const_part = dPhi * on_border_gradient;

                for (int i = 0; i < 6; ++i)
                {
                    grad[i] -= non_const_part(0, i) * derivative_const_part(row, col);
                    if (abs(non_const_part(0, i)) > 1e5 || abs(derivative_const_part(row, col)) > 1e5)
                    {
                        int for_debug = 1;
                    }
                }
                ++non_zero_pixels;
            }
        }
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