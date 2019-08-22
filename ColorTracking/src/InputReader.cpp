#include <iostream>
#include <boost/filesystem/operations.hpp>


#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

#include "test_runner/io.hpp"

#include "InputReader.h"

InputReader::InputReader(const std::string& directory_name)
{
    boost::filesystem::path mesh_path(directory_name + "/mesh.obj");
    boost::filesystem::path camera_path(directory_name+ "/camera.yml");
    ground_truth_path = boost::filesystem::path(directory_name+ "/ground_truth.yml");
    boost::filesystem::path video_path(directory_name + "/rgb");
    histograms::Mesh mesh = InputReader::getMesh(mesh_path);
    videoCapture = InputReader::getVideo(video_path);
    int height = videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int width = videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
    glm::mat4 camera_matrix = InputReader::getCamera(camera_path, height);
    glm::mat4 pose = InputReader::getPose(ground_truth_path);
    float zNear = InputReader::getZNear(pose);
    Renderer renderer(camera_matrix, zNear, zNear * 10000, width, height);
    histograms::Object3d some_other_object = histograms::Object3d(mesh, renderer);
    object3D = some_other_object;
}

histograms::Mesh InputReader::getMesh(const boost::filesystem::path& path)
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

glm::mat4 InputReader::getCamera(const boost::filesystem::path& path, int height)
{
    glm::mat4 input_camera = testrunner::readCamera(path);
    bool kinect_matrix = true;
    if (kinect_matrix)
    {
        input_camera[2][1] = -height - input_camera[2][1];
    }
    return input_camera;
}

glm::mat4 InputReader::getPose(const boost::filesystem::path &path, int frame_number)
{
    std::map<int, testrunner::Pose> poses = testrunner::readPoses(path);
    return poses[frame_number].pose;
}

int InputReader::getNumFrames(const boost::filesystem::path& path)
{
    std::map<int, testrunner::Pose> poses = testrunner::readPoses(path);
    return poses.size();
}

cv::VideoCapture InputReader::getVideo(const boost::filesystem::path& filePath)
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

float InputReader::getZNear(const glm::mat4& pose)
{
    float x = pose[3][0];
    float y = pose[3][1];
    float z = pose[3][2];
    float distance = sqrt(x * x + y * y + z * z);
    return distance * 0.01f;
}