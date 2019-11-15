#include "SLSQPPoseGetter.h"
#include <glm/gtc/matrix_transform.hpp>
#include <opencv/cv.hpp>
#include <iostream>
#include "PoseEstimator.h"


double SLSQPPoseGetter::previous[6] = { 0 };

SLSQPPoseGetter::SLSQPPoseGetter(histograms::Object3d* object3d, const glm::mat4& initial_pose)
{
    pass_to_optimization = PassToOptimization();
    pass_to_optimization.object = object3d;
    pass_to_optimization.initial_pose = initial_pose;
    pass_to_optimization.num_iterations = 12;
    pass_to_optimization.iteration_number = 0;

    const histograms::Mesh& mesh = object3d->getMesh();
    float mesh_diameter = mesh.getBBDiameter();
    max_translation_shift = 0.2f * mesh_diameter;
    max_rotation_shift = 0.5f;

    float step_size = 1e-2;


    pass_to_optimization.delta_step[0] = step_size * max_rotation_shift;
    pass_to_optimization.delta_step[1] = step_size * max_rotation_shift;
    pass_to_optimization.delta_step[2] = step_size * max_rotation_shift;
    pass_to_optimization.delta_step[3] = step_size * max_translation_shift;
    pass_to_optimization.delta_step[4] = step_size * max_translation_shift;
    pass_to_optimization.delta_step[5] = step_size * max_translation_shift;

    opt = nlopt_create(NLOPT_LD_SLSQP, 6);
    //opt = nlopt_create(NLOPT_LN_NELDERMEAD, 6);
    //opt = nlopt_create(NLOPT_LD_LBFGS, 6);
    nlopt_set_min_objective(opt, energy_function, &pass_to_optimization);
    nlopt_set_ftol_rel(opt, 1e-5);
    nlopt_set_maxeval(opt, pass_to_optimization.num_iterations);
}

glm::mat4 SLSQPPoseGetter::params_to_transform(const double *x)
{
    float rot_x = (float)x[0];
    float rot_y = (float) x[1];
    float rot_z = (float)x[2];
    float tr_x = (float)x[3];
    float tr_y = (float)x[4];
    float tr_z = (float)x[5];
    glm::mat4 transform_matrix = glm::mat4(1.0, 0, 0, 0,
                                           0, 1, 0, 0,
                                           0, 0, 1, 0,
                                           0, 0, 0, 1);
    glm::mat4 rot_x_matr = glm::rotate(transform_matrix, rot_x, glm::vec3(1, 0, 0));
    glm::mat4 rot_y_matr = glm::rotate(transform_matrix, rot_y, glm::vec3(0, 1, 0));
    glm::mat4 rot_z_matr = glm::rotate(transform_matrix, rot_z, glm::vec3(0, 0, 1));
    glm::mat4 something_interesting = rot_z_matr * rot_y_matr * rot_x_matr;
    transform_matrix = glm::translate(transform_matrix, glm::vec3(tr_x, tr_y, tr_z));
    transform_matrix = glm::rotate(transform_matrix, rot_z, glm::vec3(0, 0, 1));
    transform_matrix = glm::rotate(transform_matrix, rot_y, glm::vec3(0, 1, 0));
    transform_matrix = glm::rotate(transform_matrix, rot_x, glm::vec3(1, 0, 0));
    return transform_matrix;
}

glm::mat4 applyResultToPose(const glm::mat4& matr, const double* params)
{
    cv::Matx31f rot_vec = cv::Matx31f(params[0], params[1], params[2]);
    cv::Matx33f rot_matr = cv::Matx33f();
    cv::Rodrigues(rot_vec, rot_matr);
    glm::mat4 difference =
            glm::mat4(rot_matr(0, 0), rot_matr(1, 0), rot_matr(2, 0), 0,
                      rot_matr(0, 1), rot_matr(1, 1), rot_matr(2, 1), 0,
                      rot_matr(0, 2), rot_matr(1, 2), rot_matr(2, 2), 0,
                      params[3], params[4], params[5], 1);
    return matr * difference;
}

void plotRodriguesDirection(const histograms::Object3d &object3d,
                            const cv::Mat &frame,
                            const glm::mat4 &estimated_pose,
                            const glm::mat4 &real_pose,
                            const std::string &base_file_name);

glm::mat4 applyResultToParams(const glm::mat4& matr, double p0, double p1, double p2, double p3, double p4, double p5) {
    double params[6] = { p0, p1, p2, p3, p4, p5 };
    return applyResultToPose(matr, params);
}

std::vector<cv::Mat1d> SLSQPPoseGetter::getGradientInPoint(const glm::mat4 &initial_pose, const histograms::PoseEstimator &estimator)
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

cv::Matx12d SLSQPPoseGetter::getSignedDistanceGradient(const cv::Mat1f &signed_distance, int row, int col)
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
SLSQPPoseGetter::getGradientAnalytically(const glm::mat4 &initial_pose,
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

double SLSQPPoseGetter::energy_function(unsigned n, const double *x, double *grad, void *my_func_data)
{
    double to_debug[6];
    for (int i = 0; i < 6; ++i)
    {
        to_debug[i] = x[i];
    }

    PassToOptimization* passed_data = reinterpret_cast<PassToOptimization*>(my_func_data);
    histograms::Object3d* object = passed_data->object;
    cv::Mat& frame = passed_data->frame;
    int histo_part = (passed_data->mode == 0) ? 1 : 10;
    histo_part = 10;

    float* delta_step = passed_data->delta_step;
    glm::mat4& initial_pose = passed_data->initial_pose;

    glm::mat4 transform_matrix = applyResultToPose(initial_pose, x);
    histograms::PoseEstimator estimator;
    float current_value = estimator.estimateEnergy(*object, frame, transform_matrix, histo_part, frame.cols == 1920);
    //std::cout << "current value = " << current_value << std::endl;

    if (grad)
    {
        getGradientAnalytically(initial_pose, estimator, grad);
//        std::cout << "gradient = ";
//        for (int i = 0; i < 6; ++i)
//        {
//            std::cout << grad[i] << ' ';
//        }
//        std::cout << std::endl;



    }
    ++passed_data->iteration_number;
    for (int i = 0; i < 6; ++i)
    {
        previous[i] = x[i];
    }
    return current_value;
}



glm::mat4 SLSQPPoseGetter::getPose(const cv::Mat& frame, int mode)
{
    //std::cout << "mode " << mode << std::endl;
    switch (mode)
    {
        case 0:
            nlopt_set_maxeval(opt, 24);
            break;
        case 1:
            nlopt_set_maxeval(opt, 600);
            break;
        case 2:
            nlopt_set_maxeval(opt, 600);
            break;
        default:
            nlopt_set_maxeval(opt, 600);
            break;
    }
    pass_to_optimization.iteration_number = 0;
    pass_to_optimization.frame = frame;
    pass_to_optimization.mode = mode;
    double x[6] = { 0 };

    double lower_bounds[6] =
            { - max_rotation_shift, - max_rotation_shift, - max_rotation_shift,
              - max_translation_shift, - max_translation_shift, - max_translation_shift };
    double upper_bounds[6] =
            { max_rotation_shift, max_rotation_shift, max_rotation_shift,
              max_translation_shift, max_translation_shift, max_translation_shift };
    nlopt_set_lower_bounds(opt, lower_bounds);
    nlopt_set_upper_bounds(opt, upper_bounds);
    double minf;

    nlopt_result status = nlopt_optimize(opt, x, &minf);
    if (status >= 0)
    {
        pass_to_optimization.initial_pose = applyResultToPose(pass_to_optimization.initial_pose, x);
        return pass_to_optimization.initial_pose;
    }
    else
    {
        std::cout << "error " <<  status << ' ' << mode << std::endl;
    }
    return glm::mat4();
}

glm::mat4 SLSQPPoseGetter::getPose(const cv::Mat &frame)
{
    return getPose(frame, 0);
}
