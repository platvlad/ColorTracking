#pragma once

#include "nlopt.h"
#include "nlopt.hpp"

#include "lkt/PnPObjective.hpp"

#include "SLSQPPoseGetter.h"
#include "PoseGetter.h"
#include "Feature3DInfoList.h"

class SlsqpLktPoseGetter : public PoseGetter
{
    nlopt_opt opt;
    float max_translation_shift;
    float max_rotation_shift;

    glm::mat4 initial_pose;
    size_t num_iterations;
    histograms::Object3d* object;
    lkt::PnPObjective::Ptr pnp_obj;

    Feature3DInfoList feature_info_list;
    cv::Mat1b prev_frame;
    cv::Mat3b frame;
    lkt::Mesh mesh;
    glm::mat4 projection_matrix;

    static double energy_function(unsigned n, const double *x, double *grad, void *my_func_data);

    double energy_function_for_plot(const double *x, std::string plot_type = "mixed");

    void plotAxis(const std::string &file_name, const std::vector<float> &axis_values, const glm::mat4 &pose, size_t axis_number);


public:
    SlsqpLktPoseGetter(histograms::Object3d* object3d, const glm::mat4& initial_pose, const cv::Mat3b &init_frame);
    glm::mat4 getPose(const cv::Mat& frame, int mode, std::string directory_name, int frame_number);
    glm::mat4 getPose(const cv::Mat& frame);
    glm::mat4 getPose(const cv::Mat& frame, int mode);
    virtual void setInitialPose(const glm::mat4 &pose);

    void plotEnergy(const glm::mat4 &pose, const std::string &directory_name, int frame_number);
    ~SlsqpLktPoseGetter() {}
};