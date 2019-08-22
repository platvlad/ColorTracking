#ifndef COLORTRACKING_SLSQPPOSEGETTER_H
#define COLORTRACKING_SLSQPPOSEGETTER_H

#include <opencv2/core/mat.hpp>
#include "nlopt/nlopt.h"
#include "nlopt/nlopt.hpp"

#include <Object3d.h>
#include "PoseGetter.h"

class SLSQPPoseGetter : public PoseGetter
{
    nlopt_opt opt;
    glm::mat4 previous_pose;
    double last_params[6];
    float max_translation_shift;
    float max_rotation_shift;

    static glm::mat4 params_to_transform(const double *x);

    static double energy_function(unsigned n, const double *x, double *grad, void *my_func_data);
    static float translation_step;
    static float rotation_step;
    static cv::Mat frame;
    static histograms::Object3d* object;
public:
    SLSQPPoseGetter(histograms::Object3d* object3d, const glm::mat4& initial_pose);
    glm::mat4 getPose();
    void setFrame(const cv::Mat& new_frame);
};

#endif //COLORTRACKING_SLSQPPOSEGETTER_H
