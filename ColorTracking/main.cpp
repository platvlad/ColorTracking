#pragma comment(lib, "glew32.lib")

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

#include "DataIO.h"
#include "tests.h"

#include "lkt/Features.hpp"
#include "FeatureTracker.h"

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
        fout_rot << "    error: " << estimator.estimateEnergy(object3d, frame, transform) << std::endl;
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
        fout_tr << "    error: " << estimator.estimateEnergy(object3d, frame, transform) << std::endl;
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
        fout << "    error: " << estimator.estimateEnergy(object3d, frame, transform) << std::endl;
    }
    fout.close();
}

void runOptimization(const std::string &directory_name, const std::string &method) {
    DataIO data = DataIO(directory_name);
    Object3d& object3D = data.object3D;
    cv::VideoCapture& videoCapture = data.videoCapture;
    boost::filesystem::path& gt_path = data.ground_truth_path;
    glm::mat4 pose = data.getPose(1);
    glm::mat4 prev_pose = pose;
    int frame_number = 1;
    cv::Mat3b frame;
    videoCapture >> frame;
    cv::Mat3b flipped_frame;
    cv::flip(frame, flipped_frame, 0);
    PoseGetter* poseGetter = nullptr;
    if (method == "newton")
    {
        poseGetter = new NewtonPoseGetter(&object3D, pose);
    }
    else if (method == "slsqp")
    {
        poseGetter = new SLSQPPoseGetter(&object3D, pose);
    }
    else if (method == "ground_truth")
    {
        poseGetter = new GroundTruthPoseGetter(data);
    }
    else if (method == "slsqp_lkt")
    {
        poseGetter = new SlsqpLktPoseGetter(&object3D, pose, flipped_frame);
    }
    
    PoseEstimator estimator;
    const Renderer& renderer = object3D.getRenderer();
    const glm::mat4& camera_matrix = renderer.getCameraMatrix();
    FeatureTracker f_tracker(object3D.getMesh(), pose, camera_matrix,  frame.size());
    while (true)
    {
        std::cout << "Frame " << frame_number << std::endl;
        //pose = f_tracker.handleFrame(frame);
        
        object3D.updateHistograms(flipped_frame, pose);
        data.estimated_poses[frame_number] = pose;
        data.writePng(flipped_frame, frame_number);
        
        if (frame_number > 1)
        {
            glm::mat4 diff = get_diff_matr(prev_pose, pose);
            prev_pose = pose;
            pose = diff * pose;
            poseGetter->setInitialPose(pose);
        }

        videoCapture >> frame;
        cv::flip(frame, flipped_frame, 0);
        ++frame_number;
        if (flipped_frame.empty())
        {
            break;
        }
        if (method == "ground_truth")
        {
            if (frame_number == 2)
            {
                poseGetter->getPose(flipped_frame, 0);
            }
        }
        else if (method != "slsqp_lkt" && method != "slsqp")
        {
            cv::Mat downsampled2;
            cv::Mat downsampled4;
            cv::Mat downsampled8;
            cv::pyrDown(flipped_frame, downsampled2);
            cv::pyrDown(downsampled2, downsampled4);
            cv::pyrDown(downsampled4, downsampled8);
            poseGetter->getPose(downsampled8, 3);
            poseGetter->getPose(downsampled4, 2);
            poseGetter->getPose(downsampled2, 1);
        }

        bool plot_energy = frame_number == 2;
        if (plot_energy && method == "slsqp_lkt")
        {
            SlsqpLktPoseGetter* slsqp_lkt_pose_getter = reinterpret_cast<SlsqpLktPoseGetter*>(poseGetter);
            pose = slsqp_lkt_pose_getter->getPose(flipped_frame, 0, directory_name, frame_number);
        }
        else
        {
            pose = poseGetter->getPose(flipped_frame, 0);
        }
        std::cout << frame_number << ' ' << estimator.estimateEnergy(object3D, flipped_frame, pose, true) << std::endl;
        if (plot_energy)
        {
            //GroundTruthPoseGetter ground_truth_pose_getter = GroundTruthPoseGetter(data);
            //glm::mat4 real_pose = ground_truth_pose_getter.getPose(frame_number);
            //std::cout << "real pose error: " << estimator.estimateEnergy(object3D, flipped_frame, /*real_pose*/pose, 10, true) << std::endl;
            ////plotRodriguesDirection(object3D, frame, pose, real_pose, directory_name + "/plot/" + std::to_string(frame_number));
            //plotEnergy(object3D, flipped_frame, pose, frame_number, directory_name);
            ////data.writePlots(frame, frame_number, pose);
        }
    }
    data.writePositions();

    if (poseGetter != nullptr)
    {
        delete poseGetter;
    }
}

int main()
{
//    Tests::runTests();
    //GLuint VAO;
    //glGenVertexArrays(1, &VAO);
   // std::cout << glGetString(GL_VERSION) << std::endl;
    runOptimization("data/ir_ir_5_r", "slsqp_lkt");
    return 0;
}
