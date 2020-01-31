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

void FeatureTracker::filterObjectPoints(std::vector<glm::vec2> &pts_2d, const std::vector<size_t> &valid_points)
{
    size_t valid_points_size = valid_points.size();
    for (int i = 0; i < valid_points_size; ++i)
    {
        size_t point_index = valid_points[i];
        if (point_index > i)
        {
            object_points[i] = object_points[point_index];
            feature_list[i] = feature_list[point_index];
            pts_2d[i] = pts_2d[point_index];
        }
    }
    object_points.resize(valid_points_size);
    feature_list.resize(valid_points_size);
    pts_2d.resize(valid_points_size);
}

void FeatureTracker::getValidImagePoints(std::vector<glm::vec2> &pts_2d)
{
    for (int i = 0; i < feature_list.size(); ++i)
    {
        lkt::FeatureInfo &feat = feature_list[i];

        if (feat.flowQuality.lukasKanadeSuccess) {
            int pts_2d_size = pts_2d.size();
            if (pts_2d_size < i) {
                object_points[pts_2d_size] = object_points[i];
                feature_list[pts_2d_size] = feature_list[i];
            }
            pts_2d.push_back(glm::vec2(feat.x, feat.y));
        }
    }
    object_points.resize(pts_2d.size());
    feature_list.resize(pts_2d.size());
}

void FeatureTracker::unprojectFeatures()
{
    cv::Mat1i faceIds = lkt::getFaceIds(mesh, prev_model, projection, frame_size);
    std::vector<glm::vec2> imagePoints(feature_list.size());
    for (int i = 0; i < feature_list.size(); ++i) {
        lkt::FeatureInfo& feature = feature_list[i];
        imagePoints[i] = glm::vec2(feature.x, feature.y);
    }
    std::vector< boost::optional<std::pair<glm::vec3, size_t> > > unprojected = 
        lkt::unproject(mesh, prev_model, projection, faceIds, imagePoints);
    int object_pose_index = 0;
    for (int i = 0; i < unprojected.size(); ++i)
    {
        lkt::FeatureInfo& feature = feature_list[i];
        glm::vec2 imagePoint(feature.x, feature.y);
        //boost::optional<std::pair<glm::vec3, size_t> > feat_3d_alt = lkt::unproject(mesh, prev_model, projection, imagePoint);
        boost::optional<std::pair<glm::vec3, size_t> >& feat_3d = unprojected[i];

        if (feat_3d){
            glm::vec3 feat_3d_pos = feat_3d.get().first;
            size_t object_points_size = object_points.size();
            feature_list[object_pose_index] = feature_list[i];
            if (object_points.size() > object_pose_index) {
                object_points[object_pose_index] = feat_3d_pos;
            }
            else {
                object_points.push_back(feat_3d_pos);
            }
            ++object_pose_index;
        }
    }
    feature_list.resize(object_points.size());
}

glm::mat4 FeatureTracker::handleFrame(cv::Mat3b &frame)
{
    cv::Mat1b gray_frame;
    cv::cvtColor(frame, gray_frame, CV_RGB2GRAY);
    cv::Mat1b flipped_frame;
    cv::flip(gray_frame, flipped_frame, 0);
    cv::imwrite("data/foxes/flipped.png", flipped_frame);
    if (prev_frame.empty())
    {
        lkt::Features::FeaturesAndCorners features = frame_features.detectFeaturesAndCorners(flipped_frame);
        
        feature_list = features.first;
        prev_frame = flipped_frame;
        prev_model = init_model;
        unprojectFeatures();
        return init_model;
    }
    
    feature_list = lkt::moveFeaturesBySparseFlow(prev_frame, flipped_frame, feature_list);
    std::vector<glm::vec2> pts_2d;
    getValidImagePoints(pts_2d);
    float maxInlierError = lkt::computeMaxPnPErr(frame.cols, frame.rows);
    std::cout << "Before solveEPnPRansac" << std::endl;
    prev_model = lkt::solveEPnPRansac(object_points, pts_2d, prev_model, glm::mat4(1.0), projection, 1000, maxInlierError).get_value_or(prev_model);
    std::cout << "After solveEPnPRansac" << std::endl;
    glm::mat4 mvp = projection * prev_model;
    std::vector<size_t> inliers = lkt::findInliers(mvp, maxInlierError, object_points, pts_2d);
    std::cout << "After inliers finding" << std::endl;
    filterObjectPoints(pts_2d, inliers);
    std::cout << "After object filtering" << std::endl;
    std::cout << "object points size = " << object_points.size() << std::endl;
    std::cout << "pts_2d size = " << pts_2d.size() << std::endl;
    prev_model = lkt::solvePnP(object_points, pts_2d, prev_model, glm::mat4(1.0), projection, lkt::lm::buildHuberAndTukeyBundle());
    std::cout << "After solvePnP" << std::endl;

    mvp = projection * prev_model;
    inliers = lkt::findInliers(mvp, maxInlierError, object_points, pts_2d);
    filterObjectPoints(pts_2d, inliers);

    std::cout << "feature list size = " << feature_list.size() << std::endl;

    lkt::Features::FeaturesAndCorners new_features = frame_features.detectFeaturesAndCorners(flipped_frame);
    std::cout << "new feature list size = " << new_features.first.size() << std::endl;

    feature_list = frame_features.mergeFeatureLists(feature_list, new_features.first, frame_size);
    unprojectFeatures();
    for (int i = 0; i < feature_list.size(); ++i)
    {

    }

    prev_frame = flipped_frame;
    return prev_model;
}
