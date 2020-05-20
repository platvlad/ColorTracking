#pragma once

#include <set>

#include "renderer.h"

#include <opencv2/core.hpp>

#include "lkt/Features.hpp"
#include "lkt/unprojection.hpp"

#include "Feature3DInfoList.h"

class LkPoseGetter
{
    Feature3DInfoList feature_info_list;
    cv::Mat1b prev_frame;
    lkt::Mesh mesh;
    glm::mat4 init_model;
    glm::mat4 projection;
    glm::mat4 prev_model;
    cv::Size frame_size;


public:
    LkPoseGetter(const histograms::Mesh &mesh,
        const glm::mat4 &init_pose,
        const glm::mat4 &camera_matrix,
        const cv::Size &frame_size);

    glm::mat4 handleFrame(const cv::Mat3b &frame);

    double estimateEnergy(const glm::mat4 &pose);

    double getAvgReprojectionError(const glm::mat4 &transform) const;

    void addNewFeatures(const glm::mat4 &pose);

    void setPrevModel(const glm::mat4 &model);
};