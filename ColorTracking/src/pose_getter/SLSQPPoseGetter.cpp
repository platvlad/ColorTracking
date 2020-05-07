#include "pose_getter/SLSQPPoseGetter.h"
#include <glm/gtc/matrix_transform.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <estimator/GradientEstimator.h>
#include "estimator/GradientHessianEstimator.h"
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

double SLSQPPoseGetter::energy_function(unsigned n, const double *x, double *grad, void *my_func_data)
{
    PassToOptimization* passed_data = reinterpret_cast<PassToOptimization*>(my_func_data);
    histograms::Object3d* object = passed_data->object;
    cv::Mat& frame = passed_data->frame;
    int histo_part = (passed_data->mode == 0) ? 1 : 10;
    //histo_part = 10;
    histo_part = 1;
    glm::mat4& initial_pose = passed_data->initial_pose;

    glm::mat4 transform_matrix = applyResultToPose(initial_pose, x);
    histograms::PoseEstimator estimator;
    float current_value = estimator.estimateEnergy(*object, frame, transform_matrix, histo_part, false).first;
    if (grad)
    {
        GradientEstimator::getGradient(initial_pose, estimator, grad);
        //GradientHessianEstimator::getGradient(initial_pose, estimator, grad);
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
            nlopt_set_maxeval(opt, 36);
            break;
        case 1:
            nlopt_set_maxeval(opt, 100);
            break;
        case 2:
            nlopt_set_maxeval(opt, 200);
            break;
        default:
            nlopt_set_maxeval(opt, 400);
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

void SLSQPPoseGetter::setInitialPose(const glm::mat4 &pose)
{
    pass_to_optimization.initial_pose = pose;
}
