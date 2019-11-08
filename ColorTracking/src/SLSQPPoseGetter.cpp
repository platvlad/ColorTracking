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
    nlopt_set_ftol_rel(opt, 1e-8);
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

float best_value = 0;
int best_iteration = 0;

double SLSQPPoseGetter::getDerivativeInDirection(const histograms::Object3d &object3D,
                                                 histograms::PoseEstimator& estimator,
                                                 const glm::mat4 &minus_transform,
                                                 const glm::mat4 &plus_transform)
{
    const Renderer* renderer = estimator.getRenderer();
    cv::Size frame_size = renderer->getSize();
    Projection minus_maps = Projection(frame_size);
    renderer->projectMesh(object3D.getMesh(), minus_transform, minus_maps);
    Projection plus_maps = Projection(frame_size);
    renderer->projectMesh(object3D.getMesh(), plus_transform, plus_maps);
    const cv::Rect& currentROI = estimator.getROI();
    Projection minus_maps_on_roi = minus_maps(currentROI);
    Projection plus_maps_on_roi = plus_maps(currentROI);
    const cv::Mat1i& num_voters = estimator.getNumVoters();
    float grad_sum = 0;
    int non_zero_pixels = 0;
    const cv::Mat1f& derivative_const_part = estimator.getDerivativeConstPart();
    for (int row = 0; row < num_voters.rows; ++row)
    {
        for (int col = 0; col < num_voters.cols; ++col)
        {
            if (num_voters(row, col) > 0)
            {
                grad_sum += (plus_maps_on_roi.signed_distance(row, col) - minus_maps_on_roi.signed_distance(row, col)) *
                        derivative_const_part(row, col);
                ++non_zero_pixels;
            }
        }
    }
    if (non_zero_pixels > 0)
    {
        return grad_sum / static_cast<float>(non_zero_pixels);
    }
    return 0;
}

double SLSQPPoseGetter::getDerivativeSemiAnalytically(const histograms::Object3d &object3D,
                                                      histograms::PoseEstimator &estimator,
                                                      const glm::mat4 &minus_transform, const glm::mat4 &plus_transform)
{
    const cv::Mat1f& derivative_const_part = estimator.getDerivativeConstPart();
    const cv::Mat1i& num_voters = estimator.getNumVoters();
    const Projection& projection = estimator.getProjection();
    const cv::Mat1f& signed_distance = projection.signed_distance;
    const cv::Mat3f& depth_map = projection.depth_map;
    const Renderer* renderer = estimator.getRenderer();
    const glm::mat4& transform = estimator.getPose();
    glm::mat4 transform_inverted = glm::inverse(transform);

    cv::Size frame_size = renderer->getSize();
    Projection minus_maps = Projection(frame_size);
    renderer->projectMesh(object3D.getMesh(), minus_transform, minus_maps);
    Projection plus_maps = Projection(frame_size);
    renderer->projectMesh(object3D.getMesh(), plus_transform, plus_maps);
    const cv::Rect& currentROI = estimator.getROI();
    Projection minus_maps_on_roi = minus_maps(currentROI);
    Projection plus_maps_on_roi = plus_maps(currentROI);

    std::vector<cv::Vec2i> mask_points;
    mask_points.push_back(cv::Vec2i(0, 0));

    const cv::Mat1i& nearest_labels = projection.nearest_labels;
    for (int row = 0; row < projection.mask.rows; ++row)
    {
        for (int col = 0; col < projection.mask.cols; ++col)
        {
            if (projection.signed_distance(row, col) >= -1.5 && projection.signed_distance(row, col) < 0)
            {
                int label = nearest_labels(row, col);
                mask_points.insert(mask_points.begin() + label, cv::Vec2i(row, col));
                if (label != mask_points.size() - 1)
                {
                    std::cout << "Achtung!" << std::endl;
                }
            }
        }
    }
    double sum = 0;
    int num_pixels = 0;

    int wrong_foreground = 0;
    int wrong_background = 0;
    int num_foreground = 0;
    int num_background = 0;

    for (int row = 0; row < num_voters.rows; ++row)
    {
        for (int col = 0; col < num_voters.cols; ++col)
        {
            if (num_voters(row, col) > 0)
            {
                double dPhi_dx = 0;
                double dPhi_dy = 0;

                if (col == 0)
                {
                    dPhi_dx = signed_distance(row, col + 1) - signed_distance(row, col);
                }
                else if (col == num_voters.cols - 1)
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
                else if (row == num_voters.rows - 1)
                {
                    dPhi_dy = signed_distance(row - 1, col) - signed_distance(row, col);
                }
                else
                {
                    dPhi_dy = (signed_distance(row - 1, col) - signed_distance(row + 1, col)) / 2;
                }

                cv::Vec2i pixel_on_mask = mask_points[nearest_labels.at<int>(row, col)];
                cv::Vec3f pt_in_3d = depth_map(pixel_on_mask[0],pixel_on_mask[1]);
                pt_in_3d[2] = -pt_in_3d[2];
                glm::vec4 vec_in_3d = glm::vec4(pt_in_3d[0], pt_in_3d[1], pt_in_3d[2], 1);
                glm::vec4 init_vec_pose = transform_inverted * vec_in_3d;
                glm::vec3 minus_pixel = renderer->projectVertex(glm::vec3(init_vec_pose[0],
                                                                                    init_vec_pose[1],
                                                                                    init_vec_pose[2]),
                                                                                            minus_transform);
                glm::vec3 plus_pixel = renderer->projectVertex(glm::vec3(init_vec_pose[0],
                                                                            init_vec_pose[1],
                                                                            init_vec_pose[2]),
                                                                                    plus_transform);
                glm::vec3 init_pixel = renderer->projectVertex(glm::vec3(init_vec_pose[0],
                                                                         init_vec_pose[1],
                                                                         init_vec_pose[2]),
                                                                                 transform);
                double minus_column = minus_pixel.x;
                double minus_row = minus_pixel.y;
                double plus_column = plus_pixel.x;
                double plus_row = plus_pixel.y;

                double derivative_in_pt = derivative_const_part(row, col)
                                          * (dPhi_dx * (minus_column - plus_column) + dPhi_dy * (plus_row - minus_row ))
                                          / 2;

                double delta_sd = (dPhi_dx * (minus_column - plus_column) + dPhi_dy * (plus_row - minus_row))
                                  / 2;


                if (!projection.mask(row, col))
                {
                    ++num_background;
                }
                else
                {
                    ++num_foreground;
                }

                double alternative_delta_sd =
                        plus_maps_on_roi.signed_distance(row, col) - minus_maps_on_roi.signed_distance(row, col);
                if (delta_sd * alternative_delta_sd < 0)
                {
                    if (projection.mask(row, col))
                    {
                        ++wrong_foreground;
                    }
                    else
                    {
                        ++wrong_background;
                    }
                }
                sum += derivative_in_pt;
                ++num_pixels;
            }
        }
    }
    return sum / num_pixels;
}


void
SLSQPPoseGetter::getGradientAnalytically(const histograms::Object3d &object3D,
                                         const glm::mat4 &initial_pose,
                                         histograms::PoseEstimator &estimator,
                                         double* grad)
{
    const cv::Mat1f& derivative_const_part = estimator.getDerivativeConstPart();
    const cv::Mat1i& num_voters = estimator.getNumVoters();
    const Projection& projection = estimator.getProjection();
    const cv::Mat1f& signed_distance = projection.signed_distance;
    const cv::Mat3f& depth_map = projection.depth_map;
    const Renderer* renderer = estimator.getRenderer();
    glm::vec2 focal = renderer->getFocal();

    std::vector<cv::Vec2i> mask_points;
    mask_points.push_back(cv::Vec2i(0, 0));

    const cv::Mat1i& nearest_labels = projection.nearest_labels;
    for (int row = 0; row < projection.mask.rows; ++row)
    {
        for (int col = 0; col < projection.mask.cols; ++col)
        {
            if (projection.signed_distance(row, col) >= -1.5 && projection.signed_distance(row, col) < 0)
            {
                int label = nearest_labels(row, col);
                mask_points.insert(mask_points.begin() + label, cv::Vec2i(row, col));
                if (label != mask_points.size() - 1)
                {
                    std::cout << "Achtung!" << std::endl;
                }
            }
        }
    }

    int non_zero_pixels = 0;

    for (int row = 0; row < num_voters.rows; ++row)
    {
        for (int col = 0; col < num_voters.cols; ++col)
        {
            if (num_voters(row, col) > 0)
            {
                double dPhi_dx = 0;
                double dPhi_dy = 0;
                if (col == 0)
                {
                    dPhi_dx = signed_distance(row, col + 1) - signed_distance(row, col);
                }
                else if (col == num_voters.cols - 1)
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
                else if (row == num_voters.rows - 1)
                {
                    dPhi_dy = signed_distance(row - 1, col) - signed_distance(row, col);
                }
                else
                {
                    dPhi_dy = (signed_distance(row - 1, col) - signed_distance(row + 1, col)) / 2;
                }
                cv::Matx12d dPhi(dPhi_dx, dPhi_dy);

                cv::Vec2i pixel_on_mask = mask_points[nearest_labels.at<int>(row, col)];

                cv::Vec3f pt_in_3d = depth_map(pixel_on_mask[0],pixel_on_mask[1]);
                pt_in_3d[2] = -pt_in_3d[2];

//                cv::Matx23d dPi(-focal.x / pt_in_3d[2], 0,
//                                0, focal.y / pt_in_3d[2],
//                                pt_in_3d[0] * focal.x / (pt_in_3d[2] * pt_in_3d[2]),
//                                - pt_in_3d[1] * focal.y / (pt_in_3d[2] * pt_in_3d[2]));
                cv::Matx23d dPi(-focal.x / pt_in_3d[2], 0, pt_in_3d[0] * focal.x / (pt_in_3d[2] * pt_in_3d[2]),
                                0, focal.y / pt_in_3d[2], - pt_in_3d[1] * focal.y / (pt_in_3d[2] * pt_in_3d[2]));
                cv::Matx33d dInitF = cv::Matx33d(initial_pose[0][0], initial_pose[1][0], initial_pose[2][0],
                                               initial_pose[0][1], initial_pose[1][1], initial_pose[2][1],
                                               initial_pose[0][2], initial_pose[1][2], initial_pose[2][2]);
                cv::Mat1d dX = cv::Mat1d::zeros(3, 6);
                dX(0, 1) = pt_in_3d[2];
                dX(0, 2) = -pt_in_3d[1];
                dX(0, 3) = 1;
                dX(1, 0) = -pt_in_3d[2];
                dX(1, 2) = pt_in_3d[0];
                dX(1, 4) = 1;
                dX(2, 0) = pt_in_3d[1];
                dX(2, 1) = -pt_in_3d[0];
                dX(2, 5) = 1;
                cv::Mat1d non_const_part = dPhi * dPi * dInitF * dX;

                for (int i = 0; i < 6; ++i)
                {
                    grad[i] -= non_const_part(0, i) * derivative_const_part(row, col);
                }
                ++non_zero_pixels;
            }
        }
    }
    for (int i = 0; i < 6; ++i)
    {
        grad[i] /= non_zero_pixels;
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
//    float iteration_proportion = static_cast<float>(passed_data->iteration_number) /
//            static_cast<float>(passed_data->num_iterations);
//    if (iteration_proportion < 0.25) {
//        frame = passed_data->downsampled8;
//    }
//    else if (iteration_proportion < 0.5)
//    {
//        frame = passed_data->downsampled4;
//    }
//    else if (iteration_proportion < 0.75) {
//        frame = passed_data->downsampled2;
//    }
    float* delta_step = passed_data->delta_step;
    glm::mat4& initial_pose = passed_data->initial_pose;

    glm::mat4 transform_matrix = applyResultToPose(initial_pose, x);
    histograms::PoseEstimator estimator;
    float current_value = estimator.estimateEnergy(*object, frame, transform_matrix, histo_part, frame.cols == 1920);

    if (passed_data->iteration_number == 0)
    {
        best_value = current_value;
        best_iteration = 0;
    }
    else
    {
        if (current_value < best_value)
        {
            best_value = current_value;
            best_iteration = passed_data->iteration_number;
        }
    }

    if (grad)
    {
        //getGradientAnalytically(*object, initial_pose, estimator, grad);


        //old method
        double grad2[6] = { 0.0 };
        double x_plus_delta[6];
        double x_minus_delta[6];
        for (int i = 0; i < 6; ++i)
        {
            x_plus_delta[i] = x[i];
            x_minus_delta[i] = x[i];
        }

        for (int i = 0; i < 6; ++i)
        {
            x_plus_delta[i] += delta_step[i];
            x_minus_delta[i] -= delta_step[i];

            glm::mat4 plus_transform = applyResultToPose(initial_pose, x_plus_delta);
            glm::mat4 minus_transform = applyResultToPose(initial_pose, x_minus_delta);

            grad2[i] = getDerivativeInDirection(*object,
                                               estimator,
                                               minus_transform,
                                               plus_transform);
            grad[i] = getDerivativeSemiAnalytically(*object,
                                                    estimator,
                                                    minus_transform,
                                                    plus_transform);

            x_plus_delta[i] = x[i];
            x_minus_delta[i] = x[i];
        }
        //end old method


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
    switch (mode)
    {
        case 0:
            nlopt_set_maxeval(opt, 600);
            break;
        case 1:
            nlopt_set_maxeval(opt, 600);
            break;
        case 2:
            nlopt_set_maxeval(opt, 600);
            break;
        default:
            nlopt_set_maxeval(opt, 12);
            break;
    }
    pass_to_optimization.iteration_number = 0;
    pass_to_optimization.frame = frame;
    pass_to_optimization.mode = mode;
    cv::pyrDown(pass_to_optimization.frame, pass_to_optimization.downsampled2);
    cv::pyrDown(pass_to_optimization.downsampled2, pass_to_optimization.downsampled4);
    cv::pyrDown(pass_to_optimization.downsampled4, pass_to_optimization.downsampled8);
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
    if (nlopt_optimize(opt, x, &minf) >= 0)
    {
        pass_to_optimization.initial_pose = applyResultToPose(pass_to_optimization.initial_pose, x);
        return pass_to_optimization.initial_pose;
    }
    return glm::mat4();
}

glm::mat4 SLSQPPoseGetter::getPose(const cv::Mat &frame)
{
    return getPose(frame, 0);
}

