#include "SLSQPPoseGetter.h"
#include <glm/gtc/matrix_transform.hpp>
#include <opencv/cv.hpp>

namespace histograms
{
    float estimateEnergy(const Object3d &object, const cv::Mat3b &frame, const glm::mat4 &pose, int histo_part = 1, bool debug_info = false);
}

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
    float current_value = histograms::estimateEnergy(*object, frame, transform_matrix, histo_part);

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

    if (passed_data->iteration_number == 23)
    {

    }
    if (passed_data->iteration_number == 19)
    {

    }
    if (passed_data->iteration_number == 15)
    {

    }
    if (passed_data->iteration_number == 11)
    {

    }

    if (grad)
    {
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

            float err_plus = histograms::estimateEnergy(*object, frame, plus_transform, histo_part);
            float err_minus = histograms::estimateEnergy(*object, frame, minus_transform, histo_part);
            grad[i] = (err_plus - err_minus) / (2 * delta_step[i]);
            x_plus_delta[i] = x[i];
            x_minus_delta[i] = x[i];
        }
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
