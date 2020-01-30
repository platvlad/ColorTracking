#pragma once

#include "renderer.h"

#include <opencv2/core.hpp>

#include "lkt/Features.hpp"
#include "lkt/unprojection.hpp"

class FeatureTracker
{
    lkt::Features frame_features;
    lkt::FeatureInfoList feature_list;
    cv::Mat1b prev_frame;
    lkt::Mesh mesh;
    glm::mat4 init_model;
    glm::mat4 projection;
    std::map<size_t, glm::vec3> feat_positions;
    glm::mat4 prev_model;

public:
    FeatureTracker(const histograms::Mesh &mesh,
        const glm::mat4 &init_pose,
        const glm::mat4 &camera_matrix,
        const cv::Size &frame_size);

    glm::mat4 handleFrame(cv::Mat3b &frame);
};