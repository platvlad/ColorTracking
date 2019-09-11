#include <iostream>
#include <boost/filesystem/operations.hpp>
#include <opencv2/imgcodecs.hpp>


#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

#include "test_runner/io.hpp"

#include "DataIO.h"

DataIO::DataIO(const std::string& directory_name) : directory_name(directory_name)
{
    boost::filesystem::path mesh_path(directory_name + "/mesh.obj");
    boost::filesystem::path camera_path(directory_name+ "/camera.yml");
    ground_truth_path = boost::filesystem::path(directory_name+ "/ground_truth.yml");
    boost::filesystem::path video_path(directory_name + "/rgb.mov");
    histograms::Mesh mesh = DataIO::getMesh(mesh_path);
    videoCapture = DataIO::getVideo(video_path);
    int height = videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int width = videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
    glm::mat4 camera_matrix = DataIO::getCamera(camera_path, height);
    glm::mat4 pose = DataIO::getPose(ground_truth_path);
    float zNear = DataIO::getZNear(pose);
    Renderer renderer = Renderer(camera_matrix, zNear, zNear * 10000, width, height);
    histograms::Object3d some_other_object = histograms::Object3d(mesh, renderer);
    object3D = some_other_object;
}

histograms::Mesh DataIO::getMesh(const boost::filesystem::path& path)
{
    std::string file_name = path.string();
    std::vector<glm::vec3> vertices;
    std::vector<glm::uvec3> faces;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, file_name.c_str());
    if (ret)
    {
        for (size_t i = 2; i < attrib.vertices.size(); i += 3)
        {
            vertices.push_back(glm::vec3(attrib.vertices[i - 2], attrib.vertices[i - 1], attrib.vertices[i]));
        }
        for (size_t s = 0; s < shapes.size(); ++s)
        {
            size_t index_offset = 0;
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                int fv = shapes[s].mesh.num_face_vertices[f];

                //only 3 vertices faces are supported
                tinyobj::index_t idx0 = shapes[s].mesh.indices[index_offset + 0];
                tinyobj::index_t idx1 = shapes[s].mesh.indices[index_offset + 1];
                tinyobj::index_t idx2 = shapes[s].mesh.indices[index_offset + 2];
                faces.push_back(glm::uvec3(idx0.vertex_index, idx1.vertex_index, idx2.vertex_index));
                index_offset += fv;
            }
        }
    }
    return histograms::Mesh(vertices, faces);
}

glm::mat4 DataIO::getCamera(const boost::filesystem::path& path, int height)
{
    glm::mat4 input_camera = testrunner::readCamera(path);
    bool kinect_matrix = false;
    if (kinect_matrix)
    {
        input_camera[2][1] = -height - input_camera[2][1];
    }
    return input_camera;
}

glm::mat4 DataIO::getPose(const boost::filesystem::path &path, int frame_number)
{
    std::map<int, testrunner::Pose> poses = testrunner::readPoses(path);
    return poses[frame_number].pose;
}

int DataIO::getNumFrames(const boost::filesystem::path& path)
{
    std::map<int, testrunner::Pose> poses = testrunner::readPoses(path);
    return poses.size();
}

cv::VideoCapture DataIO::getVideo(const boost::filesystem::path& filePath)
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

float DataIO::getZNear(const glm::mat4& pose)
{
    float x = pose[3][0];
    float y = pose[3][1];
    float z = pose[3][2];
    float distance = sqrt(x * x + y * y + z * z);
    return distance * 0.01f;
}

void DataIO::writePositions()
{
    boost::filesystem::path output_yml_path(directory_name + "/output.yml");
    testrunner::writePoses(estimated_poses, output_yml_path);
}

void DataIO::writePng(cv::Mat3b frame, int frame_number)
{
    cv::Mat3b output = frame.clone();
    const Renderer& renderer = object3D.getRenderer();
    const histograms::Mesh& mesh = object3D.getMesh();
    glm::mat4 pose = estimated_poses[frame_number].pose;
//    renderer.renderMesh(mesh, output, pose);
    Maps maps = renderer.projectMesh(mesh, pose);
    cv::Mat& mask = maps.mask;
    cv::Rect roi =  maps.roi;
    for (int row = 0; row < output.rows; ++row)
    {
        for (int column = 0; column < output.cols; ++column)
        {
            if (mask.at<uchar>(row, column))
            {
                float blue = frame(row, column)[0] / 2;
                float green = frame(row, column)[1] / 2;
                float red = frame(row, column)[2] / 2;
                output.at<cv::Vec3b>(row, column) = cv::Vec3b(blue, green + 128, red);
                cv::Vec3b abracadabra = frame.at<cv::Vec3b>(row, column);
            }
        }
    }
    std::string frame_name = std::to_string(frame_number);
    frame_name = std::string(4 - frame_name.length(), '0') + frame_name;

    cv::imwrite(directory_name + "/output_frames/" + frame_name + ".png", output);
}
