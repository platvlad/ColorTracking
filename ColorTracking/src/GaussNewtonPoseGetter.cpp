//
// Created by Vladislav Platonov on 18/11/2019.
//

#include <PoseEstimator.h>
#include <GradientHessianEstimator.h>
#include "GaussNewtonPoseGetter.h"

#include <fstream>

glm::mat4 applyResultToPose(const glm::mat4& matr, const double* params);

GaussNewtonPoseGetter::GaussNewtonPoseGetter(histograms::Object3d *object3d, const glm::mat4& initial_pose) :
                                                object3d(object3d),
                                                initial_pose(initial_pose)
{

}

void GaussNewtonPoseGetter::plot2Points(const cv::Mat &frame, const glm::mat4 &init, double* params, const std::string &file_name)
{
    size_t num_points = 100;
    size_t anchor_pts_dist = num_points / 2;
    int left = -anchor_pts_dist;
    int right = 3 * anchor_pts_dist;
    double iteration_step[6] = { params[0] / anchor_pts_dist, params[1] / anchor_pts_dist, params[2] / anchor_pts_dist,
                                 params[3] / anchor_pts_dist, params[4] / anchor_pts_dist, params[5] / anchor_pts_dist };
    std::ofstream fout(file_name);
    fout << "frames:" << std::endl;
    for (int step_num = left; step_num <= right; ++step_num)
    {
        double current_step[6] = { iteration_step[0] * step_num, iteration_step[1] * step_num, iteration_step[2] * step_num,
                                   iteration_step[3] * step_num, iteration_step[4] * step_num, iteration_step[5] * step_num };
        glm::mat4 current_pose = applyResultToPose(init, current_step);
        histograms::PoseEstimator estimator;
        float current_value = estimator.estimateEnergy(*object3d, frame, current_pose, 10, false).first;
        fout << "  - frame: " << step_num - left + 1 << std::endl;
        fout << "    error: " << current_value << std::endl;
    }
    fout.close();
}

glm::mat4 GaussNewtonPoseGetter::getPose(const cv::Mat &frame, int mode)
{
    double x[6] = { 0 };
    double grad[6] = { 0 };
    double step[6] = { 0 };
    for (int i = 0; i < 2; ++i)
    {
        initial_pose = applyResultToPose(initial_pose, x);
        int histo_part = 10;
        histograms::PoseEstimator estimator;
        float current_value = estimator.estimateEnergy(*object3d, frame, initial_pose, histo_part, false).first;
        GradientHessianEstimator::getGradient(initial_pose, estimator, grad, step);
        for (int j = 0; j < 6; ++j)
        {
            x[j] = 0.5 * step[j];
            //x[j] = step[j];
        }
    }
    initial_pose = applyResultToPose(initial_pose, x);
    histograms::PoseEstimator estimator;
    float current_value = estimator.estimateEnergy(*object3d, frame, initial_pose, 10, false).first;
    return initial_pose;
}

glm::mat4 GaussNewtonPoseGetter::getPose(const cv::Mat &frame)
{
    return getPose(frame, 0);
}

void GaussNewtonPoseGetter::setInitialPose(const glm::mat4 &pose)
{
    initial_pose = pose;
}
