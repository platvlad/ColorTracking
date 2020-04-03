#include <boost/filesystem/operations.hpp>
#include <opencv2/highgui.hpp>
#include <PoseEstimator2.h>
#include <iostream>
#include <fstream>


#include "DataIO.h"

#include "test_runner/io.hpp"

#include "DataIO2.h"

glm::mat4 applyResultToPose(const glm::mat4& matr, const double* params);

DataIO2::DataIO2(const std::string& directory_name) : directory_name(directory_name)
{
    boost::filesystem::path mesh_path(directory_name + "/mesh.obj");
    boost::filesystem::path camera_path(directory_name + "/camera.yml");
    ground_truth_path = boost::filesystem::path(directory_name + "/ground_truth.yml");
    boost::filesystem::path video_path(directory_name + "/rgb");
    histograms::Mesh mesh = DataIO::getMesh(mesh_path);
    mesh_scale_factor = mesh.getBBDiameter() / 5;
    mesh.fitDiameterToFive();
    //mesh_scale_factor = 1;
    videoCapture = getVideo(video_path);
    int height = videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
    int width = videoCapture.get(CV_CAP_PROP_FRAME_WIDTH);
    glm::mat4 camera_matrix = getCamera(camera_path, height);
    glm::mat4 pose = getPose(1);
    float zNear = getZNear(pose);
    Renderer renderer = Renderer(camera_matrix, zNear, zNear * 10000, width, height);
    object3D2 = histograms::Object3d2(mesh, renderer);
    object3D2 = histograms::Object3d2(mesh, renderer);
}

glm::mat4 DataIO2::getCamera(const boost::filesystem::path& path, int height)
{
    glm::mat4 input_camera = testrunner::readCamera(path);
    return input_camera;
}

glm::mat4 DataIO2::getPose(int frame_number) const
{
    std::map<int, testrunner::Pose> poses = testrunner::readPoses(ground_truth_path);
    glm::mat4 pose = poses[frame_number].pose;
    pose[3] /= mesh_scale_factor;
    pose[3][3] = 1;
    return pose;
}

int DataIO2::getNumFrames() const
{
    std::map<int, testrunner::Pose> poses = testrunner::readPoses(ground_truth_path);
    return poses.size();
}

cv::VideoCapture DataIO2::getVideo(const boost::filesystem::path& filePath)
{
    using namespace boost::filesystem;
    std::string fileName = filePath.string();
    if (is_directory(filePath))
    {
        fileName += path::preferred_separator;
        fileName += "%04d.png";
    }
    return cv::VideoCapture(fileName);
}

float DataIO2::getZNear(const glm::mat4& pose)
{
    float x = pose[3][0];
    float y = pose[3][1];
    float z = pose[3][2];
    float distance = sqrt(x * x + y * y + z * z);
    return distance * 0.01f;
}

void DataIO2::writePositions()
{
    boost::filesystem::path output_yml_path(directory_name + "/output.yml");
    for (int i = 1; i <= estimated_poses.size(); ++i)
    {
        estimated_poses[i].pose[3] *= mesh_scale_factor;
        estimated_poses[i].pose[3][3] = 1;
    }
    testrunner::writePoses(estimated_poses, output_yml_path);
}

void DataIO2::writePng(cv::Mat3b frame, int frame_number)
{
    cv::Mat3b output = frame.clone();
    const Renderer& renderer = object3D2.getRenderer();
    const histograms::Mesh& mesh = object3D2.getMesh();
    glm::mat4 pose = estimated_poses[frame_number].pose;
    renderer.renderMesh(mesh, output, pose);
    std::string frame_name = std::to_string(frame_number);
    frame_name = std::string(4 - frame_name.length(), '0') + frame_name;
    cv::Mat3b flipped_output;
    cv::flip(output, flipped_output, 0);
    cv::imwrite(directory_name + "/output_frames/" + frame_name + ".png", flipped_output);
}

void DataIO2::writePlots(const cv::Mat3b &frame, int frame_number, const glm::mat4 &pose)
{
    int num_points = 100;
    float max_rotation = 1;
    float max_translation = 0.2f * object3D2.getMesh().getBBDiameter();
    float rotation_step = max_rotation / static_cast<float>(num_points);
    float translation_step = max_translation / static_cast<float>(num_points);
    rotation_step = 1e-3f;
    translation_step = 1e-3f;
    std::string base_file_name = directory_name + "/plots/" + std::to_string(frame_number);
    std::ofstream rot_x_file(base_file_name + "rot_x.yml");
    std::ofstream rot_y_file(base_file_name + "rot_y.yml");
    std::ofstream rot_z_file(base_file_name + "rot_z.yml");
    std::ofstream tr_x_file(base_file_name + "tr_x.yml");
    std::ofstream tr_y_file(base_file_name + "tr_y.yml");
    std::ofstream tr_z_file(base_file_name + "tr_z.yml");
    std::ofstream* output_files[6] = {
            &rot_x_file,
            &rot_y_file,
            &rot_z_file,
            &tr_x_file,
            &tr_y_file,
            &tr_z_file
    };
    for (int i = 0; i < 6; ++i)
    {
        *output_files[i] << "frames:" << std::endl;
    }

    for (int pt = -num_points; pt <= num_points; ++pt)
    {
        int pose_number = pt + num_points + 1;
        float angle = rotation_step * static_cast<float>(pt);
        float offset = translation_step * pt;
        double transform_params[6][6] = { 0 };
        for (int i = 0; i < 6; ++i) {
            if (i < 3)
            {
                transform_params[i][i] = pt * rotation_step;
            }
            else
            {
                transform_params[i][i] = pt * translation_step;
            }
        }
        glm::mat4 transforms[6];
        for (int i = 0; i < 6; ++i)
        {
            transforms[i] = applyResultToPose(pose, transform_params[i]);
        }

        histograms::PoseEstimator2 estimator;

        for (int i = 0; i < 6; ++i)
        {
            *output_files[i] << "  - frame: " << pose_number << std::endl;
            *output_files[i] << "    error: " << estimator.estimateEnergy(object3D2, frame, transforms[i]).first << std::endl;
        }

    }
    for (int i = 0; i < 6; ++i)
    {
        output_files[i]->close();
    }
}
