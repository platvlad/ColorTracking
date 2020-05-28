#pragma once

#include <set>
#include <iostream>

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include "lkt/Features.hpp"
#include "lkt/lmsolver/loss_functions.hpp"
#include "lkt/unprojection.hpp"
#include "lkt/inliers.hpp"

struct Feature3DInfo
{
    lkt::FeatureInfo feature_info;
    glm::vec3 object_pos;
    size_t face_id;
    std::vector<glm::vec2> feats_2d;
    double best_reproj_2_sum = 0;

private:
    double getSumReprojError2(const std::vector<glm::mat4> & tracked_mvps, const glm::vec3 &pt_3d)
    {
        size_t track_size = feats_2d.size();
        size_t stream_size = tracked_mvps.size();
        double sum_reprojection_error2 = 0;
        for (size_t i = 0; i < track_size; ++i)
        {
            glm::vec2& feat_info = feats_2d[i];
            size_t pose_num = stream_size - track_size + i;
            const glm::mat4& mvp = tracked_mvps[pose_num];
            sum_reprojection_error2 += lkt::computeReprojectionError2(mvp, pt_3d, feat_info);
        }
        return sum_reprojection_error2;
    }

public:
    bool update3DIfNecessary(const std::vector<glm::mat4> &tracked_mvps,
        const glm::vec3 &new_pt_3d, 
        size_t new_face_id)
    {
        glm::vec2 feat_2d(feature_info.x, feature_info.y);

        feats_2d.push_back(feat_2d);
        size_t stream_size = tracked_mvps.size();
        const glm::mat4& last_mvp = tracked_mvps[stream_size - 1];
        
        if (feats_2d.size() == 1)
        {
            //track is new
            object_pos = new_pt_3d;
            face_id = new_face_id;
            best_reproj_2_sum = lkt::computeReprojectionError2(last_mvp, new_pt_3d, feat_2d);
            return false;
        }

        best_reproj_2_sum += lkt::computeReprojectionError2(last_mvp, object_pos, feat_2d);
        double new_reproj_2_sum = getSumReprojError2(tracked_mvps, new_pt_3d);
        
        if (new_reproj_2_sum < best_reproj_2_sum)
        {
            object_pos = new_pt_3d;
            face_id = new_face_id;
            best_reproj_2_sum = new_reproj_2_sum;
            return true;
        }
        return false;
    }
};

class Feature3DInfoList
{
    std::vector<Feature3DInfo> featInfos;
    lkt::Features featureDetector;
    std::vector<glm::mat4> tracked_mvps;

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

    float getAvgReprojectionError(const glm::mat4 &mvp) const;

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