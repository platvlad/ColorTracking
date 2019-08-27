#include "SLSQPPoseGetter.h"
#include <glm/gtc/matrix_transform.hpp>
#include <opencv/cv.hpp>

namespace histograms
{
    float estimateEnergy(const Object3d &object, const cv::Mat &frame, const glm::mat4 &pose);
}

SLSQPPoseGetter::SLSQPPoseGetter(histograms::Object3d* object3d, const glm::mat4& initial_pose)
{
    pass_to_optimization = PassToOptimization();
    pass_to_optimization.object = object3d;
    pass_to_optimization.initial_pose = initial_pose;

    const histograms::Mesh& mesh = object3d->getMesh();
    float mesh_diameter = mesh.getBBDiameter();
    max_translation_shift = 0.2f * mesh_diameter;
    max_rotation_shift = 0.5f;

    pass_to_optimization.delta_step[0] = 1e-2f * max_translation_shift;
    pass_to_optimization.delta_step[1] = 1e-2f * max_translation_shift;
    pass_to_optimization.delta_step[2] = 1e-2f * max_translation_shift;
    pass_to_optimization.delta_step[3] = 1e-2f * max_rotation_shift;
    pass_to_optimization.delta_step[4] = 1e-2f * max_rotation_shift;
    pass_to_optimization.delta_step[5] = 1e-2f * max_rotation_shift;

    opt = nlopt_create(NLOPT_LD_SLSQP, 6);
    //opt = nlopt_create(NLOPT_LN_NELDERMEAD, 6);
    nlopt_set_min_objective(opt, energy_function, &pass_to_optimization);
    nlopt_set_ftol_rel(opt, 1e-3);
    nlopt_set_maxeval(opt, 36);
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

double SLSQPPoseGetter::energy_function(unsigned n, const double *x, double *grad, void *my_func_data)
{
    PassToOptimization* passed_data = reinterpret_cast<PassToOptimization*>(my_func_data);
    histograms::Object3d* object = passed_data->object;
    cv::Mat& frame = passed_data->frame;
    float* delta_step = passed_data->delta_step;
    glm::mat4& initial_pose = passed_data->initial_pose;

    glm::mat4 transform_matrix = applyResultToPose(initial_pose, x);
    float current_value = histograms::estimateEnergy(*object, frame, transform_matrix);

    if (grad)
    {
        double x_plus_delta[6];
        double x_plus_2delta[6];
        double x_plus_3delta[6];
        double x_plus_4delta[6];
        double x_plus_5delta[6];
        double x_minus_delta[6];
        double x_minus_2delta[6];
        double x_minus_3delta[6];
        double x_minus_4delta[6];
        double x_minus_5delta[6];
        for (int i = 0; i < 6; ++i)
        {
            x_plus_delta[i] = x[i];
            x_plus_2delta[i] = x[i];
            x_plus_3delta[i] = x[i];
            x_plus_4delta[i] = x[i];
            x_plus_5delta[i] = x[i];
            x_minus_delta[i] = x[i];
            x_minus_2delta[i] = x[i];
            x_minus_3delta[i] = x[i];
            x_minus_4delta[i] = x[i];
            x_minus_5delta[i] = x[i];
        }

        for (int i = 0; i < 6; ++i)
        {
            x_plus_delta[i] += delta_step[i];
            x_plus_2delta[i] += 2 * delta_step[i];
            x_plus_3delta[i] += 3 * delta_step[i];
            x_plus_4delta[i] += 4 * delta_step[i];
            x_plus_5delta[i] += 5 * delta_step[i];
            x_minus_delta[i] -= delta_step[i];
            x_minus_2delta[i] -= 2 * delta_step[i];
            x_minus_3delta[i] -= 3 * delta_step[i];
            x_minus_4delta[i] -= 4 * delta_step[i];
            x_minus_5delta[i] -= 5 * delta_step[i];
            glm::mat4 plus_transform = applyResultToPose(initial_pose, x_plus_delta);
            glm::mat4 plus2_transform = applyResultToPose(initial_pose, x_plus_2delta);
            glm::mat4 plus3_transform = applyResultToPose(initial_pose, x_plus_3delta);
            glm::mat4 plus4_transform = applyResultToPose(initial_pose, x_plus_4delta);
            glm::mat4 plus5_transform = applyResultToPose(initial_pose, x_plus_5delta);
            glm::mat4 minus_transform = applyResultToPose(initial_pose, x_minus_delta);
            glm::mat4 minus2_transform = applyResultToPose(initial_pose, x_minus_2delta);
            glm::mat4 minus3_transform = applyResultToPose(initial_pose, x_minus_3delta);
            glm::mat4 minus4_transform = applyResultToPose(initial_pose, x_minus_4delta);
            glm::mat4 minus5_transform = applyResultToPose(initial_pose, x_minus_5delta);

            float err_plus = histograms::estimateEnergy(*object, frame, plus_transform);
            float err_plus2 = histograms::estimateEnergy(*object, frame, plus2_transform);
            float err_plus3 = histograms::estimateEnergy(*object, frame, plus3_transform);
            float err_plus4 = histograms::estimateEnergy(*object, frame, plus4_transform);
            float err_plus5 = histograms::estimateEnergy(*object, frame, plus5_transform);
            float err_minus = histograms::estimateEnergy(*object, frame, minus_transform);
            float err_minus2 = histograms::estimateEnergy(*object, frame, minus2_transform);
            float err_minus3 = histograms::estimateEnergy(*object, frame, minus3_transform);
            float err_minus4 = histograms::estimateEnergy(*object, frame, minus4_transform);
            float err_minus5 = histograms::estimateEnergy(*object, frame, minus5_transform);
            grad[i] = (err_plus - err_minus) / (2 * delta_step[i]);
            grad[i] = (42 * (err_plus - err_minus) + 48 * (err_plus2 - err_minus2) + 27 * (err_plus3 - err_minus3) +
                      8 * (err_plus4 - err_minus4) + err_plus5 - err_minus5) / (512 * delta_step[i]);
            x_plus_delta[i] = x[i];
            x_plus_2delta[i] = x[i];
            x_plus_3delta[i] = x[i];
            x_plus_4delta[i] = x[i];
            x_plus_5delta[i] = x[i];
            x_minus_delta[i] = x[i];
            x_minus_2delta[i] = x[i];
            x_minus_3delta[i] = x[i];
            x_minus_4delta[i] = x[i];
            x_minus_5delta[i] = x[i];
        }
    }
    return current_value;
}



glm::mat4 SLSQPPoseGetter::getPose(const cv::Mat& frame)
{
    pass_to_optimization.frame = frame;
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
