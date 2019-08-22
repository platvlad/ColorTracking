#include <iostream>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Object3d.h>
#include <GroundTruthPoseGetter.h>
#include <SLSQPPoseGetter.h>

#include "InputReader.h"
#include "tests.h"

using namespace histograms;

namespace histograms
{
    float estimateEnergy(const Object3d &object, const cv::Mat &frame, const glm::mat4 &pose);
}

glm::mat4 getHouseCameraMatrix()
{
    return glm::mat4(1060.197, 0, 0, 0,
                     0.0, 1060.273, 0, 0,
                     -964.809, -560.952, -1, -1,
                     0, 0, -0.2, 0);
}

glm::mat4 getHousePose()
{
    return glm::mat4(-0.5629978404903484, -0.4150358824112204, 0.7146877975132525, 0,
                     0.8263772439031485, -0.29481795030308744, 0.4797739331582349, 0,
                     0.011579393863255183, 0.8607134206518601, 0.5089580779863246, 0,
                     -0.046678, 0.079228, -0.811222, 1
    );
}

void plotEnergy(const Object3d& object3d, const cv::Mat& frame, const glm::mat4& pose, int frame_number)
{
    float rotation_step = 0.075f;
    float translation_step = 0.05f;
    std::string base_file_name =
            "/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/ir_ir_5_r/plots/" + std::to_string(frame_number);
    std::ofstream fout_rot_x( base_file_name + "rot_x.yml");
    std::ofstream fout_rot_y(base_file_name + "rot_y.yml");
    std::ofstream fout_rot_z(base_file_name + "rot_z.yml");
    std::ofstream fout_tr_x(base_file_name + "tr_x.yml");
    std::ofstream fout_tr_y(base_file_name + "tr_y.yml");
    std::ofstream fout_tr_z(base_file_name + "tr_z.yml");
    fout_rot_x << "frames:" << std::endl;
    fout_rot_y << "frames:" << std::endl;
    fout_rot_z << "frames:" << std::endl;
    fout_tr_x << "frames:" << std::endl;
    fout_tr_y << "frames:" << std::endl;
    fout_tr_z << "frames:   " << std::endl;
    for (int pt = -10; pt <= 10; ++pt)
    {
        int pose_number = pt + 11;
        float angle = rotation_step * pt;
        float offset = translation_step * pt;
        glm::mat4 rot_x = glm::rotate(pose, angle, glm::vec3(1, 0, 0));
        glm::mat4 rot_y = glm::rotate(pose, angle, glm::vec3(0, 1, 0));
        glm::mat4 rot_z = glm::rotate(pose, angle, glm::vec3(0, 0, 1));
        glm::mat4 tr_x = glm::translate(pose, glm::vec3(offset, 0, 0.0f));
        glm::mat4 tr_y = glm::translate(pose, glm::vec3(0, offset, 0.0f));
        glm::mat4 tr_z = glm::translate(pose, glm::vec3(0, 0, offset));
        fout_rot_x << "  - frame: " << pose_number << std::endl;
        fout_rot_x << "    error: " << estimateEnergy(object3d, frame, rot_x) << std::endl;
        fout_rot_y << "  - frame: " << pose_number << std::endl;
        fout_rot_y << "    error: " << estimateEnergy(object3d, frame, rot_y) << std::endl;
        fout_rot_z << "  - frame: " << pose_number << std::endl;
        fout_rot_z << "    error: " << estimateEnergy(object3d, frame, rot_z) << std::endl;
        fout_tr_x << "  - frame: " << pose_number << std::endl;
        fout_tr_x << "    error: " << estimateEnergy(object3d, frame, tr_x) << std::endl;
        fout_tr_y << "  - frame: " << pose_number << std::endl;
        fout_tr_y << "    error: " << estimateEnergy(object3d, frame, tr_y) << std::endl;
        fout_tr_z << "  - frame: " << pose_number << std::endl;
        fout_tr_z << "    error: " << estimateEnergy(object3d, frame, tr_z) << std::endl;
    }
    fout_rot_x.close();
    fout_rot_y.close();
    fout_rot_z.close();
    fout_tr_x.close();
    fout_tr_y.close();
    fout_tr_z.close();
}

void testEvaluateEnergyHouse()
{
    std::string input_file = "/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/ho_fm_f/mesh.obj";

    boost::filesystem::path mesh_path
            (input_file);
    Mesh house = InputReader::getMesh(mesh_path);

    glm::mat4 camera_matrix = getHouseCameraMatrix();
    glm::mat4 pose = getHousePose();
    Renderer renderer(camera_matrix, 0.08, 800, 1920, 1080);
    Object3d object3D = Object3d(house, renderer);
    cv::Mat first_image =
            cv::imread("/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/ho_fm_f/rgb/0001.png");
    object3D.updateHistograms(first_image, pose);
    float energy = estimateEnergy(object3D, first_image, pose);
    std::cout << "energy = " << energy << std::endl;
    glm::mat4 translated = glm::translate(pose, glm::vec3(0, 0, 0.1));
    float energy_translated = estimateEnergy(object3D, first_image, translated);
    std::cout << "energy translated = " << energy_translated << std::endl;
    glm::mat4 rotated = glm::rotate(pose, 6.28f, glm::vec3(0, 1, 0));
    float energy_rotated = estimateEnergy(object3D, first_image, rotated);
    std::cout << "energy rotated = " << energy_rotated << std::endl;
}

void slsqpOptimization()
{
    InputReader data = InputReader("/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/ir_ir_5_r");
    Object3d& object3D = data.object3D;
    cv::VideoCapture& videoCapture = data.videoCapture;
    boost::filesystem::path& gt_path = data.ground_truth_path;
    glm::mat4 pose = InputReader::getPose(gt_path);
    int frame_number = 1;
    SLSQPPoseGetter poseGetter = SLSQPPoseGetter(&object3D, pose);
    cv::Mat frame;
    videoCapture >> frame;
    while (true)
    {
        object3D.updateHistograms(frame, pose);

        videoCapture >> frame;
        poseGetter.setFrame(frame);
        ++frame_number;
        if (frame.empty())
        {
            break;
        }
        pose = poseGetter.getPose();

        std::cout << frame_number << ' ' << estimateEnergy(object3D, frame, pose) << std::endl;
        //plotEnergy(object3D, frame, pose, frame_number);

    }
}

int main()
{
//    Tests::runTests();

//    testEvaluateEnergyHouse();
    InputReader data = InputReader("/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/ir_ir_5_r");
    Object3d& object3D = data.object3D;
    cv::VideoCapture& videoCapture = data.videoCapture;
    boost::filesystem::path& gt_path = data.ground_truth_path;
    glm::mat4 pose = InputReader::getPose(gt_path);
//    GroundTruthPoseGetter poseGetter = GroundTruthPoseGetter(gt_path);
//    int frame_number = 1;
//    while (true)
//    {
//        cv::Mat frame;
//        videoCapture >> frame;
//        if (frame.empty())
//        {
//            break;
//        }
//        glm::mat4 new_pose = poseGetter.getPose();
//        object3D.updateHistograms(frame, new_pose);
//        std::cout << frame_number << ' ' << estimateEnergy(object3D, frame, new_pose) << std::endl;
//        plotEnergy(object3D, frame, new_pose, frame_number);
//        ++frame_number;
//    }
    slsqpOptimization();


    return 0;
}