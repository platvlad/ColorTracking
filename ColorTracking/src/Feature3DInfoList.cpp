#include "Feature3DInfoList.h"

#include <iostream>

#include "lkt/optflow.hpp"
#include "lkt/ransac.hpp"
#include "lkt/inliers.hpp"
#include "lkt/solvers.hpp"


// get vector of feature_info fields of featInfos
lkt::FeatureInfoList Feature3DInfoList::getFeatureInfoList()
{
    lkt::FeatureInfoList result(featInfos.size());
    for (int i = 0; i < featInfos.size(); ++i) {
        result[i] = featInfos[i].feature_info;
    }
    return result;
}

// get vector of object_pos fields of featInfos
std::vector<glm::vec3> Feature3DInfoList::getObjectPosVector()
{
    std::vector<glm::vec3> result(featInfos.size());
    for (int i = 0; i < featInfos.size(); ++i)
    {
        result[i] = featInfos[i].object_pos;
    }
    return result;
}

// get vector of points on image for featInfos
std::vector<glm::vec2> Feature3DInfoList::getImagePtsVector()
{
    std::vector<glm::vec2> result(featInfos.size());
    for (int i = 0; i < featInfos.size(); ++i)
    {
        lkt::FeatureInfo& feat_info = featInfos[i].feature_info;
        result[i] = glm::vec2(feat_info.x, feat_info.y);
    }
    return result;
}

// get vector of valid indices and keep in featInfos only elements on these indices
void Feature3DInfoList::filterByIndices(const std::vector<size_t> & indices)
{
    size_t indices_size = indices.size();
    for (int i = 0; i < indices_size; ++i)
    {
        size_t valid_index = indices[i];
        if (i < valid_index)
        {
            featInfos[i] = featInfos[valid_index];
        }
    }
    featInfos.resize(indices_size);
}

// update featInfos to change FeatureInfo image coordinates getting these coordinates on next frame. Keep only succeeded points
void Feature3DInfoList::moveFeaturesBySparseFlow(const cv::Mat1b &prev_frame, const cv::Mat1b &curr_frame)
{
    lkt::FeatureInfoList feature_info_list = getFeatureInfoList();
    lkt::FeatureInfoList new_feat_info_list = lkt::moveFeaturesBySparseFlow(prev_frame, curr_frame, feature_info_list);
    for (int i = 0; i < featInfos.size(); ++i)
    {
        featInfos[i].feature_info = new_feat_info_list[i];
    }
    filterLKSuccess();
}

// return true if feature failed to trackby Lucas-Kanade
bool Feature3DInfoList::lkFailed(Feature3DInfo& feat_3d_info)
{
    return !feat_3d_info.feature_info.flowQuality.lukasKanadeSuccess;
}

// remove all features whose tracking failed
void Feature3DInfoList::filterLKSuccess()
{
    std::remove_if(featInfos.begin(), featInfos.end(), lkFailed);
}

//find new pose solving epnp ransac on current features
glm::mat4 Feature3DInfoList::solveEPnPRansac(
    const glm::mat4 &prev_model, 
    const glm::mat4 &view, 
    const glm::mat4 &projection, 
    size_t iter_count, 
    float maxInlierError)
{
    std::vector<glm::vec3> pts_3d = getObjectPosVector();
    std::vector<glm::vec2> pts_2d = getImagePtsVector();
    return lkt::solveEPnPRansac(pts_3d, pts_2d, prev_model, view, projection, iter_count, maxInlierError).get_value_or(prev_model);
}

glm::mat4 Feature3DInfoList::solvePnP(
    const glm::mat4 &model,
    const glm::mat4 &view,
    const glm::mat4 &projection,
    const std::vector<lkt::lm::LossFunction::Ptr> &lossFunctions)
{
    std::vector<glm::vec3> pts_3d = getObjectPosVector();
    std::vector<glm::vec2> pts_2d = getImagePtsVector();
    std::cout << "pts_3d size = " << pts_3d.size() << std::endl;
    std::cout << "pts_2d size = " << pts_2d.size() << std::endl;
    std::cout << "Solving PnP on mvp = ";
    glm::mat4 mvp = projection * view * model;
    for (int i = 0; i < 4; ++i)
    {
        std::cout << "[";
        for (int j = 0; j < 4; ++j)
        {
            std::cout << mvp[j][i] << ' ';
        }
        std::cout << "]" << std::endl;
    }
    glm::mat4 result = lkt::solvePnP(pts_3d, pts_2d, model, view, projection, lossFunctions);
    std::cout << "Got ";
    for (int i = 0; i < 4; ++i)
    {
        std::cout << "[";
        for (int j = 0; j < 4; ++j)
        {
            std::cout << result[j][i] << ' ';
        }
        std::cout << "]" << std::endl;
    }
    return result;
}

void Feature3DInfoList::filterOutliers(const glm::mat4 &mvp, float maxInlierError)
{
    std::cout << "Before outliers filtering: " << featInfos.size() << std::endl;
    std::vector<glm::vec3> pts_3d = getObjectPosVector();
    std::vector<glm::vec2> pts_2d = getImagePtsVector();    
    std::vector<size_t> inliers = lkt::findInliers(mvp, maxInlierError, pts_3d, pts_2d);
    std::cout << "inliers size = " << inliers.size() << std::endl;
    filterByIndices(inliers);
    std::cout << "After outliers filtering " << featInfos.size() << std::endl;
}

void Feature3DInfoList::filterInvisible(const lkt::Mesh &mesh, const glm::mat4 &model, const glm::mat4 &projection, const cv::Size &frame_size)
{
    cv::Mat1i faceIds = lkt::getFaceIds(mesh, model, projection, frame_size);
    size_t num_features = featInfos.size();
    std::vector<glm::vec2> imagePoints(num_features);
    for (int i = 0; i < num_features; ++i)
    {
        lkt::FeatureInfo& feature = featInfos[i].feature_info;
        imagePoints[i] = glm::vec2(feature.x, feature.y);
    }
    std::vector< boost::optional<std::pair<glm::vec3, size_t> > > unprojected =
        lkt::unproject(mesh, model, projection, faceIds, imagePoints);
    std::set<int> face_set = getFaceSet(faceIds);
    int filtered_feat_counter = 0;
    for (int i = 0; i < num_features; ++i)
    {
        if (unprojected[i]) {
            if (face_set.count(featInfos[i].face_id)) {
                if (filtered_feat_counter < i) {
                    featInfos[filtered_feat_counter] = featInfos[i];
                }
                ++filtered_feat_counter;
            }
        }
    }
    featInfos.resize(filtered_feat_counter);
}

std::set<int> Feature3DInfoList::getFaceSet(cv::Mat1i &faceIds)
{
    std::set<int> result;
    for (int row = 0; row < faceIds.rows; ++row)
    {
        for (int col = 0; col < faceIds.cols; ++col)
        {
            int face_id = faceIds(row, col);
            if (face_id >= 0) {
                result.insert(face_id);
            }
        }
    }
    return result;
}

std::vector< boost::optional<std::pair<glm::vec3, size_t> > > Feature3DInfoList::unprojectFeatures(
    const lkt::FeatureInfoList &feature_info_list, 
    const lkt::Mesh &mesh, 
    const glm::mat4 &model, 
    const glm::mat4 &projection, 
    cv::Size &frame_size)
{
    cv::Mat1i faceIds = lkt::getFaceIds(mesh, model, projection, frame_size);

    size_t num_features = feature_info_list.size();
    std::vector<glm::vec2> imagePoints(num_features);
    for (int i = 0; i < num_features; ++i)
    {
        const lkt::FeatureInfo& feature = feature_info_list[i];
        imagePoints[i] = glm::vec2(feature.x, feature.y);
    }

    return lkt::unproject(mesh, model, projection, faceIds, imagePoints);
}

std::map<size_t, size_t> Feature3DInfoList::getIdPositions(const std::vector<Feature3DInfo> &feat_infos)
{
    std::map<size_t, size_t> result;
    for (int i = 0; i < feat_infos.size(); ++i)
    {
        result[feat_infos[i].feature_info.id] = i;
    }
    return result;
}

lkt::FeatureInfoList Feature3DInfoList::getFeatureInfoList(const std::vector<Feature3DInfo> &feature_3d_info_list)
{
    size_t feature_3d_info_list_size = feature_3d_info_list.size();
    lkt::FeatureInfoList result(feature_3d_info_list_size);
    for (int i = 0; i < feature_3d_info_list_size; ++i) {
        result[i] = feature_3d_info_list[i].feature_info;
    }
    return result;
}

void Feature3DInfoList::addNewFeatures(const cv::Mat1b &frame, const lkt::Mesh &mesh, const glm::mat4 &model, const glm::mat4 &projection)
{
    lkt::Features::FeaturesAndCorners new_features = featureDetector.detectFeaturesAndCorners(frame);
    std::vector< boost::optional<std::pair<glm::vec3, size_t> > > unprojected = 
        unprojectFeatures(new_features.first, mesh, model, projection, frame.size());
    if (!featInfos.empty())
    {
        lkt::FeatureInfoList old_feature_list = getFeatureInfoList();
        std::cout << "old features size " << old_feature_list.size() << std::endl;
        std::cout << "new features size " << new_features.first.size() << std::endl;
        lkt::FeatureInfoList merged_feature_info_list = featureDetector.mergeFeatureLists(old_feature_list, new_features.first, frame.size());
        std::cout << "merged list size = " << merged_feature_info_list.size() << std::endl;
        std::map<size_t, size_t> old_feat_positions = getIdPositions(featInfos);
        std::vector<Feature3DInfo> new_vector(merged_feature_info_list.size());
        size_t new_vector_counter = 0;
        for (int i = 0; i < merged_feature_info_list.size(); ++i)
        {
            lkt::FeatureInfo& feat = merged_feature_info_list[i];
            std::map<size_t, size_t>::iterator old_pos_iter = old_feat_positions.find(feat.id);
            if (old_pos_iter != old_feat_positions.end())
            {
                new_vector[new_vector_counter] = featInfos[old_pos_iter->second];
                ++new_vector_counter;
            }
            else {
                if (unprojected[i])
                {
                    std::pair<glm::vec3, size_t> unprojected_data = unprojected[i].get();
                    Feature3DInfo feat_3d_info;
                    feat_3d_info.feature_info = merged_feature_info_list[i];
                    feat_3d_info.object_pos = unprojected_data.first;
                    feat_3d_info.face_id = unprojected_data.second;
                    new_vector[new_vector_counter] = feat_3d_info;
                    ++new_vector_counter;
                }
            }
        }
        new_vector.resize(new_vector_counter);
        featInfos = new_vector;
        /*filterInvisible(mesh, model, projection, frame.size());
        std::cout << "old counter = " << featInfos.size() << std::endl;
        std::vector<Feature3DInfo> new_features_3d_infos = get3DInfo(new_features.first, mesh, model, projection, frame.size());
        std::cout << "new counter = " << new_features_3d_infos.size() << std::endl;
        std::vector<Feature3DInfo> old_features_3d_infos = featInfos;
        std::map<size_t, size_t> idsToOldVectorPos = getIdPositions(old_features_3d_infos);
        std::map<size_t, size_t> idsToNewVectorPos = getIdPositions(new_features_3d_infos);
        lkt::FeatureInfoList old_feature_list = getFeatureInfoList();
        lkt::FeatureInfoList new_feature_list = getFeatureInfoList(new_features_3d_infos);
        lkt::FeatureInfoList merged_feature_list = featureDetector.mergeFeatureLists(old_feature_list, new_feature_list, frame.size());
        size_t merged_feature_list_size = merged_feature_list.size();
        featInfos = std::vector<Feature3DInfo>(merged_feature_list_size);*/
        //for (int i = 0; i < merged_feature_list_size; ++i)
        //{
        //    lkt::FeatureInfo& feat_info = merged_feature_list[i];
        //    std::map<size_t, size_t>::iterator old_vector_pos_iter = idsToOldVectorPos.find(feat_info.id);
        //    if (old_vector_pos_iter != idsToOldVectorPos.end()) {
        //        featInfos[i] = old_features_3d_infos[old_vector_pos_iter->second];
        //    }
        //    else {
        //        std::map<size_t, size_t>::iterator new_vector_pos_iter = idsToNewVectorPos.find(feat_info.id);
        //        featInfos[i] = new_features_3d_infos[new_vector_pos_iter->second];
        //    }
        //}
    }
    else {
        for (int i = 0; i < unprojected.size(); ++i)
        {
            if (unprojected[i])
            {
                std::pair<glm::vec3, size_t> unprojected_data = unprojected[i].get();
                Feature3DInfo feat_3d_info;
                feat_3d_info.feature_info = new_features.first[i];
                feat_3d_info.object_pos = unprojected_data.first;
                feat_3d_info.face_id = unprojected_data.second;
                featInfos.push_back(feat_3d_info);
            }
        }
    }
}

void drawLine(cv::Mat3b& frame, cv::Point red_cross, cv::Point blue_point)
{
    cv::Scalar color = cv::Vec3b(0, 255, 0);
    cv::line(frame, red_cross, blue_point, color, 1, CV_AA);
    cv::Vec3b red = cv::Vec3b(0, 0, 255);
    int cross_row = red_cross.y;
    int cross_col = red_cross.x;
    frame(cross_row, cross_col) = red;
    frame(cross_row - 1, cross_col - 1) = red;
    frame(cross_row - 1, cross_col + 1) = red;
    frame(cross_row + 1, cross_col - 1) = red;
    frame(cross_row + 1, cross_col + 1) = red;
    frame(blue_point.y, blue_point.x) = cv::Vec3b(255, 0, 0);
}


//filpped frame
void Feature3DInfoList::drawFeatures(cv::Mat3b &frame, const glm::mat4 &mvp)
{
    for (int i = 0; i < featInfos.size(); ++i)
    {
        Feature3DInfo& feat_3d_info = featInfos[i];
        lkt::FeatureInfo &feat_info = feat_3d_info.feature_info;
        int row = floor(feat_info.y);
        int col = floor(feat_info.x);
        glm::vec4 proj_pt = mvp * glm::vec4(feat_3d_info.object_pos, 1.0);
        int proj_row = floor(proj_pt.y / proj_pt.w);
        int proj_col = floor(proj_pt.x / proj_pt.w);
        drawLine(frame, cv::Point(col, row), cv::Point(proj_col, proj_row));
    }
}

void Feature3DInfoList::drawMask(
    cv::Mat3b &frame,
    const lkt::Mesh &mesh,
    const glm::mat4 &model,
    const glm::mat4 &projection,
    const cv::Size &frame_size)
{
    cv::Mat1i faceIds = lkt::getFaceIds(mesh, model, projection, frame_size);
    for (int row = 0; row < faceIds.rows; ++row)
    {
        for (int col = 0; col < faceIds.cols; ++col)
        {
            if (faceIds(row, col) >= 0)
            {
                float blue = frame(row, col)[0] / 2;
                float green = frame(row, col)[1] / 2;
                float red = frame(row, col)[2] / 2;
                frame(row, col) = cv::Vec3b(blue, green + 128, red);
            }
        }
    }
}
