#include "FeatureTracker.h"

#include "mesh.h"

#include <opencv2/highgui.hpp>

#include <boost/optional.hpp>

#include "lkt/optflow.hpp"
#include "lkt/ransac.hpp"
#include "lkt/inliers.hpp"

#include <iostream>


FeatureTracker::FeatureTracker(const histograms::Mesh &mesh,
        const glm::mat4 &init_pose,
        const glm::mat4 &camera_matrix,
        const cv::Size &frame_size) : 
    frame_features(lkt::Features(frame_size.area())),
    mesh(lkt::Mesh(mesh.getVertices(), mesh.getFaces())),
    init_model(init_pose),
    projection(camera_matrix)
{
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
        cv::Mat1i faceIds = lkt::getFaceIds(mesh, init_model, projection, frame.size());
        std::vector<glm::vec2> imagePoints(feature_list.size());
        for (int i = 0; i < feature_list.size(); ++i) {
            lkt::FeatureInfo& feature = feature_list[i];
            imagePoints[i] = glm::vec2(feature.x, feature.y);
        }
        std::vector< boost::optional<std::pair<glm::vec3, size_t> > > unprojected = lkt::unproject(mesh, init_model, projection, faceIds, imagePoints);
        
        size_t counter = 0;
        for (int i = 0; i < unprojected.size(); ++i)
        {
            lkt::FeatureInfo& feature = feature_list[i];
            glm::vec2 imagePoint(feature.x, feature.y);
            //boost::optional<std::pair<glm::vec3, size_t> > feat_3d_alt = lkt::unproject(mesh, init_model, projection, imagePoint);
            boost::optional<std::pair<glm::vec3, size_t> >& feat_3d = unprojected[i];
            if (feat_3d)
            {
                glm::vec3 feat_3d_pos = feat_3d.get().first;
                feat_positions.insert(std::pair<size_t, glm::vec3>(feature.id, feat_3d_pos));
                
            }
        }
        return init_model;
    }
    
    feature_list = lkt::moveFeaturesBySparseFlow(prev_frame, flipped_frame, feature_list);
    std::vector<glm::vec3> pts_3d;
    std::vector<glm::vec2> pts_2d;
    for (std::map<size_t, glm::vec3>::iterator feat_pos_it = feat_positions.begin(); feat_pos_it != feat_positions.end(); ++feat_pos_it)
    {
        size_t feat_id = feat_pos_it->first;
        if (feat_id < feature_list.size())
        {
            lkt::FeatureInfo &feat = feature_list[feat_id];
            if (feat.flowQuality.lukasKanadeSuccess)
            {
                pts_3d.push_back(feat_pos_it->second);
                pts_2d.push_back(glm::vec2(feat.x, feat.y));
            }
        }
    }
    float maxInlierError = lkt::computeMaxPnPErr(frame.cols, frame.rows);
    prev_model = lkt::solveEPnPRansac(pts_3d, pts_2d, prev_model, glm::mat4(1.0), projection, 1000, maxInlierError).get_value_or(prev_model);
    
    prev_frame = flipped_frame;
    return prev_model;
}
