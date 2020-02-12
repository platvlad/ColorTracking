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
    feature_info_list(frame_size.area()),
    mesh(lkt::Mesh(mesh.getVertices(), mesh.getFaces())),
    init_model(init_pose),
    projection(camera_matrix)
{
}

glm::mat4 FeatureTracker::handleFrame(cv::Mat3b &frame)
{
    cv::Mat3b flipped_frame;
    cv::flip(frame, flipped_frame, 0);
    cv::Mat1b gray_frame;
    cv::cvtColor(flipped_frame, gray_frame, CV_BGR2GRAY);
    if (prev_frame.empty())
    {
        feature_info_list.addNewFeatures(gray_frame, mesh, init_model, projection);
        prev_frame = gray_frame;
        prev_model = init_model;
        cv::imwrite("data\\ir_ir_5_r\\flipped.png", flipped_frame);
        return init_model;
    }
    feature_info_list.moveFeaturesBySparseFlow(prev_frame, gray_frame);
    float maxInlierError = lkt::computeMaxPnPErr(frame.cols, frame.rows);
    prev_model = feature_info_list.solveEPnPRansac(prev_model, glm::mat4(1.0), projection, 1000, maxInlierError);
    glm::mat4 mvp = projection * prev_model;
    feature_info_list.filterOutliers(mvp, maxInlierError);
    
    prev_model = feature_info_list.solvePnP(prev_model, glm::mat4(1.0), projection, lkt::lm::buildHuberAndTukeyBundle());

    mvp = projection * prev_model;
    feature_info_list.filterOutliers(mvp, maxInlierError);
    feature_info_list.addNewFeatures(gray_frame, mesh, prev_model, projection);

    feature_info_list.drawFeatures(flipped_frame, mvp);
    Feature3DInfoList::drawMask(flipped_frame, mesh, prev_model, projection, frame_size);

    prev_frame = gray_frame;
    cv::flip(flipped_frame, frame, 0);
    return prev_model;
}
