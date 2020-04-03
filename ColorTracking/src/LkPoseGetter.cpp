#include "LkPoseGetter.h"

#include "mesh.h"

#include <opencv2/highgui.hpp>

#include <boost/optional.hpp>
#include <boost/make_shared.hpp>

#include "lkt/optflow.hpp"
#include "lkt/ransac.hpp"
#include "lkt/inliers.hpp"
#include "lkt/lmsolver/loss_functions.hpp"
#include "lkt/solvers.hpp"
#include "lkt/PnPObjective.hpp"

#include <iostream>

std::vector<double> rodriguesFromTransform(const glm::mat4& matr);


double LkPoseGetter::estimateEnergy(const glm::mat4 & pose)
{
    std::vector<double> params = std::vector<double>(6, 0);
    lkt::lm::ParametersFixer fixer(params, std::vector<bool>(params.size(), false));
    std::vector<glm::vec3> obj_poses = feature_info_list.getObjectPosVector();
    std::vector<glm::vec2> img_pts = feature_info_list.getImagePtsVector();
    std::cout << obj_poses.size() << std::endl;
    lkt::PnPObjective::Ptr pnp_obj = boost::make_shared<lkt::PnPObjective>(
        obj_poses,
        img_pts,
        glm::mat4(1.0),
        projection,
        fixer,
        boost::make_shared<lkt::lm::HuberLoss>());
   
    std::cout << obj_poses[0].x << ' ' << obj_poses[1].y << ' ' << obj_poses[2].z << std::endl;
    int num_features = feature_info_list.getFeatureCount();
    int num_measurements = 2 * num_features;
    std::vector<double> feat_errors(num_measurements, 0);
    lkt::lm::VectorViewD err(feat_errors);
    lkt::lm::DenseMatrix jacobian(num_measurements, 6, 0);

    std::vector<double> rodrigue_params = rodriguesFromTransform(pose);
    pnp_obj->computeView(rodrigue_params, err, jacobian, true);
    cv::Mat1d err_vector(num_measurements, 1);
    for (int i = 0; i < num_measurements; ++i)
    {
        err_vector(i, 0) = err[i];
    }
    cv::Mat1d err_transposed;
    cv::transpose(err_vector, err_transposed);
    cv::Mat1d err_matr = err_transposed * err_vector;
    return err_matr(0, 0) / num_features;
}

LkPoseGetter::LkPoseGetter(const histograms::Mesh &mesh,
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

//assume input frame is flipped
glm::mat4 LkPoseGetter::handleFrame(const cv::Mat3b &frame)
{
    cv::Mat1b gray_frame;
    cv::cvtColor(frame, gray_frame, CV_BGR2GRAY);
    if (prev_frame.empty())
    {
        feature_info_list.addNewFeatures(gray_frame, mesh, init_model, projection);
        prev_frame = gray_frame;
        prev_model = init_model;
        return init_model;
    }
    feature_info_list.moveFeaturesBySparseFlow(prev_frame, gray_frame);
    float maxInlierError = lkt::computeMaxPnPErr(frame.cols, frame.rows);


    prev_model = feature_info_list.solveEPnPRansac(prev_model, glm::mat4(1.0), projection, 1000, maxInlierError);
    glm::mat4 mvp = projection * prev_model;
    feature_info_list.filterOutliers(mvp, maxInlierError);

    std::cout << "num features = " << feature_info_list.getFeatureCount() << std::endl;
    
    glm::mat4 solved_pnp = feature_info_list.solvePnP(prev_model, glm::mat4(1.0), projection, lkt::lm::buildHuberAndTukeyBundle());

    if (!std::isnan(solved_pnp[0][0]))
    {
        prev_model = solved_pnp;
    }
    else
    {
        std::cout << "Nan happened" << std::endl;
    }
    /*mvp = projection * prev_model;
    feature_info_list.filterOutliers(mvp, maxInlierError);
    feature_info_list.addNewFeatures(gray_frame, mesh, prev_model, projection);*/

    //feature_info_list.drawFeatures(frame, mvp);
    //Feature3DInfoList::drawMask(frame, mesh, prev_model, projection, frame_size);

    prev_frame = gray_frame;
    //cv::flip(flipped_frame, frame, 0);
    return prev_model;
}



void LkPoseGetter::addNewFeatures(const glm::mat4 &pose)
{
    /*double old_energy = estimateEnergy(prev_model);
    double new_energy = estimateEnergy(pose);
    std::cout << "old energy = " << old_energy << std::endl;
    std::cout << "new energy = " << new_energy << std::endl;*/
    prev_model = pose;
    glm::mat4 mvp = projection * prev_model;
    float maxInlierError = lkt::computeMaxPnPErr(prev_frame.cols, prev_frame.rows);
    feature_info_list.filterOutliers(mvp, maxInlierError);
    feature_info_list.addNewFeatures(prev_frame, mesh, prev_model, projection);
}

void LkPoseGetter::setPrevModel(const glm::mat4 &model)
{
    prev_model = model;
}