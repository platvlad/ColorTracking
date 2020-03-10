#pragma once

#include <set>

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include "lkt/Features.hpp"
#include "lkt/lmsolver/loss_functions.hpp"
#include "lkt/unprojection.hpp"

struct Feature3DInfo
{
    lkt::FeatureInfo feature_info;
    glm::vec3 object_pos;
    size_t face_id;
};

class Feature3DInfoList
{
    std::vector<Feature3DInfo> featInfos;
    lkt::Features featureDetector;

    lkt::FeatureInfoList getFeatureInfoList();

    static bool lkFailed(Feature3DInfo& feat_3d_info);

    void filterByIndices(const std::vector<size_t> & indices);

    void filterInvisible(const lkt::Mesh &mesh, const glm::mat4 &model, const glm::mat4 &projection, const cv::Size &frame_size);

    static std::set<int> getFaceSet(cv::Mat1i &faceIds);

    static std::map<size_t, size_t> getIdPositions(const std::vector<Feature3DInfo> &feat_infos);

    static lkt::FeatureInfoList getFeatureInfoList(const std::vector<Feature3DInfo> &feature_3d_info_list);

    static std::vector< boost::optional<std::pair<glm::vec3, size_t> > > unprojectFeatures(
        const lkt::FeatureInfoList &feature_info_list,
 
        const lkt::Mesh &mesh,
 
        const glm::mat4 &model,
 
        const glm::mat4 &projection, cv::Mat1i & face_ids);


public:
    Feature3DInfoList(size_t totalPixelCount): featureDetector(totalPixelCount) {}

    //both frames should be flipped around X axis
    void moveFeaturesBySparseFlow(const cv::Mat1b &prev_frame, const cv::Mat1b &curr_frame);

    void filterLKSuccess();

    std::vector<glm::vec3> getObjectPosVector();
    std::vector<glm::vec2> getImagePtsVector();

    size_t size() const;

    glm::mat4 solveEPnPRansac(
        const glm::mat4 &prev_model, 
        const glm::mat4 &view, 
        const glm::mat4 &projection, 
        size_t iter_count, 
        float maxInlierError);

    glm::mat4 solvePnP(
        const glm::mat4 &model, 
        const glm::mat4 &view, 
        const glm::mat4 &projection, 
        const std::vector<lkt::lm::LossFunction::Ptr> &lossFunctions);

    void filterOutliers(const glm::mat4 &mvp, float maxInlierError);

    void addNewFeatures(const cv::Mat1b &frame, const lkt::Mesh &mesh, const glm::mat4 &model, const glm::mat4 &projection);

    int getFeatureCount() const;

    void drawFeatures(cv::Mat3b &frame, const glm::mat4 &mvp);

    static void drawMask(
        cv::Mat3b &frame, 
        const lkt::Mesh &mesh, 
        const glm::mat4 &model, 
        const glm::mat4 &projection, 
        const cv::Size &frame_size);
};