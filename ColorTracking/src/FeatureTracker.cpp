#include "FeatureTracker.h"

#include "mesh.h"

#include <opencv2/highgui.hpp>

#include <boost/optional.hpp>

#include "lkt/optflow.hpp"
#include "lkt/ransac.hpp"
#include "lkt/inliers.hpp"
#include "lkt/lmsolver/loss_functions.hpp"
#include "lkt/solvers.hpp"

#include <iostream>


FeatureTracker::FeatureTracker(const histograms::Mesh &mesh,
        const glm::mat4 &init_pose,
        const glm::mat4 &camera_matrix,
        const cv::Size &frame_size) : 
    frame_size(frame_size),
    frame_features(lkt::Features(frame_size.area())),
    mesh(lkt::Mesh(mesh.getVertices(), mesh.getFaces())),
    init_model(init_pose),
    projection(camera_matrix)
{
}

void FeatureTracker::filterObjectPoints(std::vector<glm::vec3> &pts_3d, std::vector<glm::vec2> &pts_2d, const std::vector<size_t> &valid_points)
{
    size_t valid_points_size = valid_points.size();
    size_t first_point_index_to_delete = 0;
    for (int i = 0; i < valid_points_size; ++i)
    {
        size_t point_index = valid_points[i];
        if (point_index > i)
        {
            feature_list[i] = feature_list[point_index];

            pts_3d[i] = pts_3d[point_index];
            pts_2d[i] = pts_2d[point_index];
        }
        for (int j = first_point_index_to_delete; j < point_index; ++j)
        {
            object_points.erase(feature_list[j].id);
            feature_faces.erase(feature_list[j].id);
        }
        first_point_index_to_delete = point_index + 1;
    }
    feature_list.resize(valid_points_size);
    pts_3d.resize(valid_points_size);
    pts_2d.resize(valid_points_size);
    
}

void FeatureTracker::getValidObjectImagePoints(std::vector<glm::vec3> &pts_3d, std::vector<glm::vec2> &pts_2d)
{
    int valid_points_count = 0;
    for (int i = 0; i < feature_list.size(); ++i)
    {
        lkt::FeatureInfo &feat = feature_list[i];

        if (feat.flowQuality.lukasKanadeSuccess) {
            int pts_2d_size = pts_2d.size();
            if (valid_points_count < i) {
                feature_list[valid_points_count] = feature_list[i];
            }
            ++valid_points_count;
            pts_2d.push_back(glm::vec2(feat.x, feat.y));
            pts_3d.push_back(object_points[feat.id]);
        }
        else {
            object_points.erase(feat.id);
            feature_faces.erase(feat.id);
        }
    }
    feature_list.resize(valid_points_count);
}

std::set<int> FeatureTracker::getFaceSet(cv::Mat1i &faceIds)
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

void FeatureTracker::unprojectFeatures(cv::Mat3b& flipped_frame)
{
    cv::Mat1i faceIds = lkt::getFaceIds(mesh, prev_model, projection, frame_size);

    for (int row = 0; row < flipped_frame.rows; ++row)
    {
        for (int col = 0; col < flipped_frame.cols; ++col)
        {
            if (faceIds(row, col) >= 0)
            {
                float blue = flipped_frame(row, col)[0] / 2;
                float green = flipped_frame(row, col)[1] / 2;
                float red = flipped_frame(row, col)[2] / 2;
                flipped_frame(row, col) = cv::Vec3b(blue, green + 128, red);
            }
        }
    }

    std::vector<glm::vec2> imagePoints(feature_list.size());
    for (int i = 0; i < feature_list.size(); ++i) {
        lkt::FeatureInfo& feature = feature_list[i];
        imagePoints[i] = glm::vec2(feature.x, feature.y);
    }
    std::vector< boost::optional<std::pair<glm::vec3, size_t> > > unprojected = 
        lkt::unproject(mesh, prev_model, projection, faceIds, imagePoints);
    int feature_list_index = 0;

    std::set<int> face_set = getFaceSet(faceIds);
    for (int i = 0; i < unprojected.size(); ++i)
    {

        size_t feat_id = feature_list[i].id;
        boost::optional<std::pair<glm::vec3, size_t> >& feat_3d = unprojected[i];

        std::map<size_t, size_t>::iterator feat_face_iter = feature_faces.find(feat_id);
        bool feat_is_new = feat_face_iter == feature_faces.end();
        bool is_visible = feat_is_new;
        if (!feat_is_new)
        {
            if (face_set.count(feat_face_iter->second)) { 
                is_visible = true; 
            }
        }

        if (feat_3d && is_visible){
            std::pair<glm::vec3, size_t> feat_3d_data = feat_3d.get();
            glm::vec3 &feat_3d_pos = feat_3d_data.first;

            if (feat_is_new) {
                object_points[feat_id] = feat_3d_pos;
                feature_faces[feat_id] = feat_3d_data.second;
            }
            feature_list[feature_list_index] = feature_list[i];
            ++feature_list_index;
        }
        else {
            object_points.erase(feat_id);
            feature_faces.erase(feat_id);
        }
    }
    feature_list.resize(feature_list_index);
}

void drawLine(cv::Mat3b& frame, cv::Point red_cross, cv::Point blue_point) 
{
    //red_cross.y -= 40;
    //blue_point.y -= 40;
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


glm::mat4 FeatureTracker::handleFrame(cv::Mat3b &frame)
{
    cv::Mat3b flipped_frame;
    cv::flip(frame, flipped_frame, 0);
    cv::Mat1b gray_frame;
    cv::cvtColor(flipped_frame, gray_frame, CV_BGR2GRAY);
    if (prev_frame.empty())
    {
        lkt::Features::FeaturesAndCorners features = frame_features.detectFeaturesAndCorners(gray_frame);
        feature_list = features.first;
        prev_frame = gray_frame;
        prev_model = init_model;
        unprojectFeatures(flipped_frame);
        cv::imwrite("data\\ir_ir_5_r\\flipped.png", flipped_frame);
        return init_model;
    }
    
    feature_list = lkt::moveFeaturesBySparseFlow(prev_frame, gray_frame, feature_list);
    std::vector<glm::vec2> pts_2d;
    std::vector<glm::vec3> pts_3d;
    getValidObjectImagePoints(pts_3d, pts_2d);
    float maxInlierError = lkt::computeMaxPnPErr(frame.cols, frame.rows);
    prev_model = lkt::solveEPnPRansac(pts_3d, pts_2d, prev_model, glm::mat4(1.0), projection, 1000, maxInlierError).get_value_or(prev_model);
    glm::mat4 mvp = projection * prev_model;
    std::vector<size_t> inliers = lkt::findInliers(mvp, maxInlierError, pts_3d, pts_2d);
    filterObjectPoints(pts_3d, pts_2d, inliers);

    if (pts_3d.size() > 0)
    {
        prev_model = lkt::solvePnP(pts_3d, pts_2d, prev_model, glm::mat4(1.0), projection, lkt::lm::buildHuberAndTukeyBundle());
    }

    mvp = projection * prev_model;
    
    inliers = lkt::findInliers(mvp, maxInlierError, pts_3d, pts_2d);
    filterObjectPoints(pts_3d, pts_2d, inliers);

    lkt::Features::FeaturesAndCorners new_features = frame_features.detectFeaturesAndCorners(gray_frame);

    feature_list = frame_features.mergeFeatureLists(feature_list, new_features.first, frame_size);
    unprojectFeatures(flipped_frame);
    for (int i = 0; i < feature_list.size(); ++i) {
        lkt::FeatureInfo &feat_info = feature_list[i];
        int row = floor(feat_info.y);
        int col = floor(feat_info.x);
        mvp = projection * prev_model;
        glm::vec4 proj_pt = mvp * glm::vec4(object_points[feat_info.id], 1.0);
        int proj_row = floor(proj_pt.y / proj_pt.w);
        int proj_col = floor(proj_pt.x / proj_pt.w);
        drawLine(flipped_frame, cv::Point(col, row), cv::Point(proj_col, proj_row));
    }
    prev_frame = gray_frame;
    cv::flip(flipped_frame, frame, 0);
    return prev_model;
}
