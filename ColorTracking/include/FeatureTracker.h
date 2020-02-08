#pragma once

#include <set>

#include "renderer.h"

#include <opencv2/core.hpp>

#include "lkt/Features.hpp"
#include "lkt/unprojection.hpp"

class FeatureTracker
{
    lkt::Features frame_features;
    lkt::FeatureInfoList feature_list;
    std::map<size_t, size_t> feature_ages;
    cv::Mat1b prev_frame;
    lkt::Mesh mesh;
    glm::mat4 init_model;
    glm::mat4 projection;
    std::map<size_t, glm::vec3> object_points;
    std::map<size_t, size_t> feature_faces;
    glm::mat4 prev_model;
    cv::Size frame_size;

    void filterObjectPoints(std::vector<glm::vec3> &pts_3d, 
        std::vector<glm::vec2> &pts_2d, 
        const std::vector<size_t> &valid_points);

    //return feature list indices of points
    std::vector<size_t> getValidObjectImagePoints(std::vector<glm::vec3> &pts_3d, std::vector<glm::vec2> &pts_2d, int min_age);
    std::set<int> getFaceSet(cv::Mat1i &faceIds);
    void unprojectFeatures(cv::Mat3b& flipped_frame);
    void filterFeatureListIndices(const std::vector<size_t> &valid_indices);

public:
    FeatureTracker(const histograms::Mesh &mesh,
        const glm::mat4 &init_pose,
        const glm::mat4 &camera_matrix,
        const cv::Size &frame_size);

    glm::mat4 handleFrame(cv::Mat3b &frame);
};