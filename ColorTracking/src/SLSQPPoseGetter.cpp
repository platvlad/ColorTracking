#include "SLSQPPoseGetter.h"
#include <glm/gtc/matrix_transform.hpp>
#include <opencv/cv.hpp>

namespace histograms
{
    float estimateEnergy(const Object3d &object, const cv::Mat &frame, const glm::mat4 &pose);
}

float SLSQPPoseGetter::translation_step = 0.01;
float SLSQPPoseGetter::rotation_step = 0.2;
cv::Mat SLSQPPoseGetter::frame = cv::Mat3b();
histograms::Object3d* SLSQPPoseGetter::object = nullptr;


SLSQPPoseGetter::SLSQPPoseGetter(histograms::Object3d* object3d, const glm::mat4& initial_pose) :
                                    previous_pose(initial_pose)
{
    cv::Matx33f rotation_matrix = cv::Matx33f(initial_pose[0][0], initial_pose[1][0], initial_pose[2][0],
                                              initial_pose[0][1], initial_pose[1][1], initial_pose[2][1],
                                              initial_pose[0][2], initial_pose[1][2], initial_pose[2][2]);
    cv::Matx31f rotation_vector = cv::Matx31f();
    cv::Rodrigues(rotation_matrix, rotation_vector);
    last_params[0] = rotation_vector(0, 0);
    last_params[1] = rotation_vector(1, 0);
    last_params[2] = rotation_vector(2, 0);
    last_params[3] = initial_pose[3][0];
    last_params[4] = initial_pose[3][1];
    last_params[5] = initial_pose[3][2];
    const histograms::Mesh& mesh = object3d->getMesh();
    float mesh_diameter = mesh.getBBDiameter();
    max_translation_shift = 0.2f * mesh_diameter;
    max_rotation_shift = 0.5f;
    SLSQPPoseGetter::object = object3d;
    SLSQPPoseGetter::translation_step = 1e-4f * max_translation_shift;
    SLSQPPoseGetter::rotation_step = 1e-4f * max_rotation_shift;
    opt = nlopt_create(NLOPT_LD_SLSQP, 6);
    nlopt_set_min_objective(opt, energy_function, nullptr);
    nlopt_set_ftol_rel(opt, 1e-3);
    nlopt_set_maxeval(opt, 500);
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


double SLSQPPoseGetter::energy_function(unsigned n, const double *x, double *grad, void *my_func_data)
{
    glm::mat4 transform_matrix = params_to_transform(x);

    float current_value = histograms::estimateEnergy(*object, frame, transform_matrix);

    if (grad)
    {
        glm::mat4 tr_plus_delta[6];
        glm::mat4 tr_minus_delta[6];
        tr_plus_delta[0] = glm::rotate(transform_matrix, rotation_step, glm::vec3(1, 0, 0));
        tr_minus_delta[0] = glm::rotate(transform_matrix, -rotation_step, glm::vec3(1, 0, 0));
        tr_plus_delta[1] = glm::rotate(transform_matrix, rotation_step, glm::vec3(0, 1, 0));
        tr_minus_delta[1] = glm::rotate(transform_matrix, -rotation_step, glm::vec3(0, 1, 0));
        tr_plus_delta[2] = glm::rotate(transform_matrix, rotation_step, glm::vec3(0, 0, 1));
        tr_minus_delta[2] = glm::rotate(transform_matrix, -rotation_step, glm::vec3(0, 0, 1));

        tr_plus_delta[3] = glm::translate(transform_matrix, glm::vec3(translation_step, 0, 0));
        tr_minus_delta[3] = glm::translate(transform_matrix, glm::vec3(-translation_step, 0, 0));
        tr_plus_delta[4] = glm::translate(transform_matrix, glm::vec3(0, translation_step, 0));
        tr_minus_delta[4] = glm::translate(transform_matrix, glm::vec3(0, -translation_step, 0));
        tr_plus_delta[5] = glm::translate(transform_matrix, glm::vec3(0, 0, translation_step));
        tr_minus_delta[5] = glm::translate(transform_matrix, glm::vec3(0, 0, -translation_step));

        float err_plus[6];
        float err_minus[6];
        for (int i = 0; i < 6; ++i)
        {
            err_plus[i] = histograms::estimateEnergy(*object, frame, tr_plus_delta[i]);
            err_minus[i] = histograms::estimateEnergy(*object, frame, tr_minus_delta[i]);
            if (i < 3)
            {
                grad[i] = (err_plus[i] - err_minus[i]) / (2 * rotation_step);
            }
            else
            {
                grad[i] = (err_plus[i] - err_minus[i]) / (2 * translation_step);
            }
        }
    }
    return current_value;
}

glm::mat4 SLSQPPoseGetter::getPose()
{
    float rot_x = atan2(previous_pose[1][2], previous_pose[2][2]);
    float rot_y = atan2(-previous_pose[0][2], sqrt(previous_pose[1][2] * previous_pose[1][2] + previous_pose[2][2] * previous_pose[2][2]));
    float rot_z = atan2(previous_pose[0][1], previous_pose[0][0]);
    float tr_x = previous_pose[3][0];
    float tr_y = previous_pose[3][1];
    float tr_z = previous_pose[3][2];
    double x[6] = { rot_x, rot_y, rot_z, tr_x, tr_y, tr_z };
    double old_lower_bounds[6];
    nlopt_get_lower_bounds(opt, old_lower_bounds);
    double old_upper_bounds[6];
    nlopt_get_upper_bounds(opt, old_upper_bounds);
    double lower_bounds[6] =
            { rot_x - max_rotation_shift, rot_y - max_rotation_shift, rot_z - max_rotation_shift,
              tr_x - max_translation_shift, tr_y - max_translation_shift, tr_z - max_translation_shift };
    double upper_bounds[6] =
            { rot_x + max_rotation_shift, rot_y + max_rotation_shift, rot_z + max_rotation_shift,
              tr_x + max_translation_shift, tr_y + max_translation_shift, tr_z + max_translation_shift };
    nlopt_set_lower_bounds(opt, lower_bounds);
    nlopt_set_upper_bounds(opt, upper_bounds);
    glm::mat4 some_pose = params_to_transform(x);
    double new_lower_bounds[6];
    nlopt_get_lower_bounds(opt, new_lower_bounds);
    double new_upper_bounds[6];
    nlopt_get_upper_bounds(opt, new_upper_bounds);
    double minf;
    if (nlopt_optimize(opt, x, &minf) >= 0)
    {
        previous_pose = params_to_transform(x);
        return previous_pose;
    }
    return glm::mat4();
}

void SLSQPPoseGetter::setFrame(const cv::Mat &new_frame)
{
    frame = new_frame;
}

