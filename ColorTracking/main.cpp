//#pragma comment(lib, "glew32.lib")

#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <glm/gtc/matrix_transform.hpp>

//#define GLEW_STATIC
#include <GL/glew.h>

#include <PoseEstimator.h>
#include <GroundTruthPoseGetter.h>
#include <SLSQPPoseGetter.h>
#include <SlsqpLktPoseGetter.h>
#include <opencv2/calib3d.hpp>
#include <GaussNewtonPoseGetter.h>
#include <NewtonPoseGetter.h>
#include "Tracker.h"
#include "PyramideTracker.h"
#include "LktInitTracker.h"
#include "GroundTruthTracker.h"
#include "SlsqpLktTracker.h"
#include "LktTracker.h"

#include "DataIO.h"
#include "DataIO2.h"
#include "tests.h"

#include "lkt/Features.hpp"

using namespace histograms;

glm::mat4 applyResultToPose(const glm::mat4& matr, const double* params);

std::vector<double> params_from_diff(const glm::mat4& prev, const glm::mat4& curr)
{
    glm::mat4 prev_inv = glm::inverse(prev);
    glm::mat4 diff = curr * prev_inv;
    cv::Matx33d rot_diff = cv::Matx33d(diff[0][0], diff[1][0], diff[2][0],
                                       diff[0][1], diff[1][1], diff[2][1],
                                       diff[0][2], diff[1][2], diff[2][2]);
    cv::Vec3d rot_params = cv::Vec3d();
    cv::Rodrigues(rot_diff, rot_params);
    std::vector<double> result;
    result.reserve(6);
    for (int i = 0; i < 3; ++i)
    {
        result.push_back(rot_params(i));
    }
    for (int i = 0; i < 3; ++i)
    {
        result.push_back(diff[3][i]);
    }
    return result;

}

glm::mat4 get_diff_matr(const glm::mat4& prev, const glm::mat4& curr)
{
    glm::mat4 prev_inv = glm::inverse(prev);
    return curr * prev_inv;
}

void plotColorRotation(
    const Object3d &object3d, 
    const cv::Mat3b &frame,
    const glm::mat4& pose, 
    const std::vector<float> &angles, 
    const glm::vec3& axis, 
    std::string file_name)
{
    std::ofstream fout_rot(file_name);
    fout_rot << "frames:" << std::endl;

    for (int i = 0; i < angles.size(); ++i)
    {
        glm::mat4 transform = glm::rotate(pose, angles[i], axis);
        PoseEstimator estimator;
        fout_rot << "  - frame: " << i + 1 << std::endl;
        fout_rot << "    error: " << estimator.estimateEnergy(object3d, frame, transform).first << std::endl;
    }
    fout_rot.close();
}

void plotColorTranslation(
    const Object3d &object3d,
    const cv::Mat3b &frame,
    const glm::mat4& pose,
    const std::vector<float> &offsets,
    const glm::vec3& axis,
    std::string file_name)
{
    std::ofstream fout_tr(file_name);
    fout_tr << "frames:" << std::endl;
    for (int i = 0; i < offsets.size(); ++i)
    {
        glm::vec3 offset_vector(axis.x * offsets[i], axis.y * offsets[i], axis.z * offsets[i]);
        glm::mat4 transform = glm::translate(pose, offset_vector);
        PoseEstimator estimator;
        fout_tr << "  - frame: " << i + 1 << std::endl;
        fout_tr << "    error: " << estimator.estimateEnergy(object3d, frame, transform).first << std::endl;
    }
    fout_tr.close();
}

void plotEnergy(const Object3d& object3d, const cv::Mat3b& frame, const glm::mat4& pose, int frame_number, std::string directory_name)
{
    int num_points = 100;
    float max_rotation = 0.1f;
    float max_translation = 0.1f * object3d.getMesh().getBBDiameter();
    float rotation_step = max_rotation / static_cast<float>(num_points);
    float translation_step = max_translation / static_cast<float>(num_points);
    std::string base_file_name =
            directory_name + "/plots/" + std::to_string(frame_number);
    std::string rot_x_file = base_file_name + "rot_x.yml";
    std::string rot_y_file = base_file_name + "rot_y.yml";
    std::string rot_z_file = base_file_name + "rot_z.yml";
    std::string tr_x_file = base_file_name + "tr_x.yml";
    std::string tr_y_file = base_file_name + "tr_y.yml";
    std::string tr_z_file = base_file_name + "tr_z.yml";
    std::vector<float> angles(2 * num_points + 1);
    std::vector<float> offsets(2 * num_points + 1);
    for (int pt = -num_points; pt <= num_points; ++pt)
    {
        int pose_number = pt + num_points;
        angles[pose_number] = rotation_step * pt;
        offsets[pose_number] = translation_step * pt;
    }
    plotColorRotation(object3d, frame, pose, angles, glm::vec3(1.0, 0.0, 0.0), rot_x_file);
    plotColorRotation(object3d, frame, pose, angles, glm::vec3(0.0, 1.0, 0.0), rot_y_file);
    plotColorRotation(object3d, frame, pose, angles, glm::vec3(0.0, 0.0, 1.0), rot_z_file);
    plotColorTranslation(object3d, frame, pose, offsets, glm::vec3(1.0, 0.0, 0.0), tr_x_file);
    plotColorTranslation(object3d, frame, pose, offsets, glm::vec3(1.0, 0.0, 0.0), tr_x_file);
    plotColorTranslation(object3d, frame, pose, offsets, glm::vec3(1.0, 0.0, 0.0), tr_x_file);
    
}

void plotRodriguesDirection(const Object3d &object3d,
                            const cv::Mat &frame,
                            const glm::mat4 &estimated_pose,
                            const glm::mat4 &real_pose,
                            const std::string &base_file_name)
{
    glm::mat4 er_tr = glm::inverse(estimated_pose) * real_pose;
    cv::Matx33f rot_matr = cv::Matx33f(er_tr[0][0], er_tr[1][0], er_tr[2][0],
                                       er_tr[0][1], er_tr[1][1], er_tr[2][1],
                                       er_tr[0][2], er_tr[1][2], er_tr[2][2]);
    cv::Matx31f rot_vec = cv::Matx31f();
    cv::Rodrigues(rot_matr, rot_vec);
    std::cout << "difference = " << rot_vec(0, 0) << ' ' << rot_vec(1, 0) << ' ' <<
        rot_vec(2, 0) << ' ' << er_tr[3][0] << ' ' << er_tr[3][1] << ' ' << er_tr[3][2] << std::endl;
    float max_rot_x = 2 * rot_vec(0, 0);
    float max_rot_y = 2 * rot_vec(1, 0);
    float max_rot_z = 2 * rot_vec(2, 0);
    float max_tr_x = 2 * er_tr[3][0];
    float max_tr_y = 2 * er_tr[3][1];
    float max_tr_z = 2 * er_tr[3][2];

    int num_points = 100;
    float steps[6] = { max_rot_x, max_rot_y, max_rot_z,
                       max_tr_x, max_tr_y, max_tr_z };
    float err_size = 0;
    for (int i = 0; i < 6; ++i)
    {
        err_size += steps[i] * steps[i];
    }
    err_size = sqrt(err_size);
    float step_factor = 2e-4f / err_size;
    std::cout << "steps to real on plot: " << 1 / (2 * step_factor) << std::endl;
    for (int i = 0; i < 6; ++i)
    {
        steps[i] *= step_factor;
    }
//    double step_size = 2e-4f / sqrt(6);
//    double steps[6] = { step_size, step_size, step_size, step_size, step_size, step_size };
    std::ofstream fout(base_file_name + "tr_dir.yml");
    fout << "frames:" << std::endl;
    PoseEstimator estimator;
    for (int pt = -num_points; pt <= num_points; ++pt)
    {
        int pose_number = pt + num_points + 1;
        double params[6] = { steps[0] * pt, steps[1] * pt, steps[2] * pt, steps[3] * pt, steps[4] * pt, steps[5] * pt };
        glm::mat4 transform = applyResultToPose(estimated_pose, params);
        fout << "  - frame: " << pose_number << std::endl;
        fout << "    error: " << estimator.estimateEnergy(object3d, frame, transform).first << std::endl;
    }
    fout.close();
}

void process_frame(const cv::Mat3b &input, cv::Mat3b &output)
{
    if (!input.empty())
    {
        cv::Mat3b flipped_frame;
        cv::flip(input, output, 0);
        //cv::cvtColor(flipped_frame, output, CV_BGR2HSV);
        //std::vector<cv::Mat1b> hsv_channels;
        //cv::split(output, hsv_channels);
        //cv::equalizeHist(hsv_channels[2], hsv_channels[2]);
        //cv::merge(hsv_channels, output);
    }
}

void track(const std::string &directory_name, const std::string &method)
{
    Tracker* tracker = nullptr;
    //if (method == "newton")
    //{
    //    tracker = new PyramideTracker<NewtonPoseGetter>(directory_name);
    //}
    //else if (method == "gauss_newton")
    //{
    //    tracker = new PyramideTracker<GaussNewtonPoseGetter>(directory_name);
    //}
    if (method == "slsqp")
    {
        Tracker2* tracker2 = new PyramideTracker<SLSQPPoseGetter>(directory_name, 0);
        tracker2->run();
        delete tracker2;
        return;
    }
    //else if (method == "lkt_init")
    //{
    //    tracker = new LktInitTracker(directory_name);
    //}
    else if (method == "ground_truth")
    {
        tracker = new GroundTruthTracker(directory_name);
    }
    else if (method == "slsqp_lkt")
    {
        tracker = new SlsqpLktTracker(directory_name);
    }
    else if (method == "lkt")
    {
        tracker = new LktTracker(directory_name);
    }
    tracker->run();
    if (tracker != nullptr)
    {
        delete tracker;
    }
}

int main()
{
//    Tests::runTests();
    //GLuint VAO;
    //glGenVertexArrays(1, &VAO);
   // std::cout << glGetString(GL_VERSION) << std::endl;
    track("data/ir_ir_5_r", "slsqp");
    return 0;
}
