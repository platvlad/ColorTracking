#include <iostream>
#include "estimator/GradientHonestHessianEstimator.h"

glm::mat4 applyResultToPose(const glm::mat4& matr, const double* params);

std::vector<cv::Mat1d> GradientHonestHessianEstimator::getGradientInPoint(const glm::mat4 &initial_pose,
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
    B3B4.push_back(cv::Mat1d());
    B2B3B4.push_back(cv::Mat1d());
    for (int i = 0; i < 6; ++i)
    {
        Magic[i].push_back(cv::Matx23d());
    }

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

                cv::Mat1d dInitF_dX = cv::Mat1d(dInitF) * dX;

                cv::Mat1d dPi_dInitF_dX = cv::Mat1d(dPi) * dInitF_dX;

                B3B4.insert(B3B4.begin() + label, dInitF_dX);
                B2B3B4.insert(B2B3B4.begin() + label, dPi_dInitF_dX);

//                double plus_params[6] = { 0 };
//                plus_params[3] = 1;
//                double minus_params[6] = { 0 };
//                minus_params[3] = -1;
//                glm::mat4 plus_transform = applyResultToPose(transform, plus_params);
//                glm::mat4 minus_transform = applyResultToPose(transform, minus_params);
//                glm::vec4 plus_vec_in_3d = plus_transform * vec_on_model;
//                glm::vec4 minus_vec_in_3d = minus_transform * vec_on_model;
//                glm::vec3 plus_pixel = renderer->projectTransformedVertex(plus_vec_in_3d);
//                glm::vec3 pixel = renderer->projectTransformedVertex(vec_in_3d);
//                glm::vec3 minus_pixel = renderer->projectTransformedVertex(minus_vec_in_3d);

                double z2 = vec_in_3d.z * vec_in_3d.z;
                double fx_over_z2 = focal.x / z2;
                double fx_over_z3 = fx_over_z2 / vec_in_3d.z;
                double fy_over_z2 = focal.y / z2;
                double fy_over_z3 = fy_over_z2 / vec_in_3d.z;
                for (int i = 0; i < 6; ++i)
                {
                    Magic[i].insert(Magic[i].begin() + label,
                            cv::Matx23d(fx_over_z2 * dInitF_dX(2, i), 0, fx_over_z3 * (dInitF_dX(0, i) * vec_in_3d.z - 2 * vec_in_3d.x * dInitF_dX(2, i)),
                                        0, fy_over_z2 * dInitF_dX(2, i), fy_over_z3 * (dInitF_dX(1, i) * vec_in_3d.z - 2 * vec_in_3d.y * dInitF_dX(2, i))));
                }

                gradients.insert(gradients.begin() + label, dPi_dInitF_dX);
            }
        }
    }
    return gradients;
}

cv::Matx12d GradientHonestHessianEstimator::getSignedDistanceGradient(const cv::Mat1f &signed_distance, int row, int col)
{
    double dPhi_dx = 0;
    double dPhi_dy = 0;
    if (col == 0)
    {
        dPhi_dx = signed_distance(row, col) - signed_distance(row, col + 1);
    }
    else if (col == signed_distance.cols - 1)
    {
        dPhi_dx = signed_distance(row, col - 1) - signed_distance(row, col);
    }
    else
    {
        dPhi_dx = (signed_distance(row, col - 1) - signed_distance(row, col + 1)) / 2;
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

bool GradientHonestHessianEstimator::getGradient(const glm::mat4 &initial_pose, histograms::PoseEstimator &estimator,
                                                 double *grad, double *step, histograms::Object3d* object_unused)
{
    for (int i = 0; i < 6; ++i)
    {
        grad[i] = 0.0;
    }

    cv::Mat1d hessian_transposed = cv::Mat1d::zeros(6, 6);

    const cv::Mat1f& derivative_const_part = estimator.getDerivativeConstPart();   // = A
    const cv::Mat1i& num_voters = estimator.getNumVoters();
    const Projection& projection = estimator.getProjection();
    const cv::Mat1f& signed_distance = projection.signed_distance;  // = Ф
    const cv::Mat1i& nearest_labels = projection.nearest_labels;

    std::vector<cv::Mat1d> on_border_gradients = getGradientInPoint(initial_pose, estimator);   // = B (without Ф)

    int non_zero_pixels = 0;
    int hessian_estimators = 0;

    std::vector<cv::Mat1d> non_const_part_on_previous_row(num_voters.cols);
    std::vector<cv::Matx12d> dPhi_on_prev_row(num_voters.cols);

    int num_zero = 0;
    int num_strange = 0;
    int num_wrong = 0;
    int num_ok = 0;

    cv::Mat1d signed_distance_second_der = cv::Mat1d::zeros(6, 6);

    for (int row = 0; row < num_voters.rows; ++row)
    {
        cv::Mat1d non_const_part_on_previous_col;
        cv::Matx12d dPhi_on_prev_col;

        for (int col = 0; col < num_voters.cols; ++col)
        {
            cv::Matx12d dPhi = getSignedDistanceGradient(signed_distance, row, col);

            cv::Mat1d on_border_gradient = on_border_gradients[nearest_labels.at<int>(row, col)];

            cv::Mat1d B = cv::Mat1d(dPhi) * on_border_gradient;   // = B


            int near_pt_index = nearest_labels.at<int>(row, col);

            if (num_voters(row, col) > 0)
            {
                double A = derivative_const_part(row, col);
                double Phi = signed_distance(row, col);
                double s = Projection::s_heaviside;
                double s2Phi = s * s * Phi;
                double H0_scale_part = A * (A - 2 * s2Phi / (s2Phi * Phi + 1));
                double grad_here[6];
                for (int i = 0; i < 6; ++i)
                {
                    grad_here[i] = B(0, i) * A;

                }

                //compute hessian only if row * col > 0
                if (row > 0 && col > 0 && row < num_voters.rows - 1 && col < num_voters.cols - 1)
                {
                    double d2Phi_dx2 = 2 * signed_distance(row, col) -
                                       signed_distance(row, col - 1) -
                                       signed_distance(row, col + 1);
                    double d2Phi_dy2 = 2 * signed_distance(row, col) -
                                       signed_distance(row - 1, col) -
                                       signed_distance(row + 1, col);
                    double d2Phi_dxdy = (signed_distance(row + 1, col - 1) +
                                         signed_distance(row - 1, col + 1) -
                                         signed_distance(row + 1, col + 1) -
                                         signed_distance(row - 1, col - 1)) / 4;
                    cv::Matx22d d2Phi = cv::Matx22d(d2Phi_dx2, d2Phi_dxdy,
                                                    d2Phi_dxdy, d2Phi_dy2);
                    non_const_part_on_previous_col = cv::Mat1d(dPhi_on_prev_col) * on_border_gradient;

                    cv::Mat1d B2B3B4_transposed = cv::Mat1d();
                    cv::transpose(B2B3B4[near_pt_index], B2B3B4_transposed);

                    cv::Mat1d dB1 = cv::Mat1d(d2Phi) * on_border_gradient;

                    cv::Mat1d H11 = B2B3B4_transposed * cv::Mat1d(d2Phi) * B2B3B4[near_pt_index];

                    cv::Mat1d H12 = cv::Mat1d::zeros(6, 6);

                    for (int i = 0; i < 6; ++i)
                    {
                        cv::Mat1d magic_term = cv::Mat1d(dPhi * Magic[i][near_pt_index]) * B3B4[near_pt_index];
                        for (int j = 0; j < 6; ++j)
                        {
                            H12(i, j) += magic_term(j);
                        }
                    }

                    cv::Mat1d H1 = A * (H11 + H12);

                    signed_distance_second_der += H11 + H12;

                    cv::Mat1d H0 = cv::Mat1d::zeros(6, 6);
                    //H0
                    for (int i = 0; i < 6; ++i)
                    {
                        for (int j = 0; j < 6; ++j)
                        {
                            H0(i, j) += H0_scale_part * B(0, i) * B(0, j);
                        }
                    }
                    hessian_transposed += H0;
                    hessian_transposed += H1;
                    ++hessian_estimators;
                    for (int i = 0; i < 6; ++i)
                    {
                        grad[i] += grad_here[i];
                    }
                }
                ++non_zero_pixels;
            }
            non_const_part_on_previous_col = B;
            non_const_part_on_previous_row[col] = B;
            dPhi_on_prev_col = dPhi;
            dPhi_on_prev_row[col] = dPhi;
        }
    }


    //if (non_zero_pixels > 0)
    if (hessian_estimators > 0)
    {
        for (int i = 0; i < 6; ++i)
        {
            if (abs(grad[i]) > 1e10)
            {
                int for_break_point = 1;
            }
            //grad[i] /= non_zero_pixels;
            grad[i] /= hessian_estimators;
            if (abs(grad[i]) > 1e5)
            {
                int for_break_point = 1;
            }
        }
    }
    if (hessian_estimators > 0)
    {
        hessian_transposed /= hessian_estimators;
        cv::Mat1d hessian_mat = cv::Mat1d::zeros(6, 6);
        cv::transpose(hessian_transposed, hessian_mat);
        cv::Matx61d eigen_values;
        cv::eigen(hessian_mat, eigen_values);
        if (abs(hessian_mat(0, 0)) > 1e10)
        {
            return false;
        }
//        for (int i = 0; i < 6; ++i) {
//            if (eigen_values(i, 0) <= 0.001)
//            {
//                return false;
//            }
//        }
        cv::Mat1d hessian_inv = cv::Mat1d::zeros(6, 6);
        cv::invert(hessian_mat, hessian_inv);
        cv::Mat1d grad_mat = cv::Mat1d::zeros(6, 1);
        for (int i = 0; i < 6; ++i)
        {
            grad_mat(0, i) = grad[i];
        }
        cv::Mat1d step_mat = - hessian_inv * grad_mat;
        for (int i = 0; i < 6; ++i)
        {
            step[i] = step_mat(i, 0);
        }
        return true;
    }
    return false;
}
