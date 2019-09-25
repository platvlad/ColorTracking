#include <iostream>

#include <maps.h>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <renderer.h>
#include <opencv2/imgcodecs.hpp>
#include <Object3d.h>
#include "tests.h"

namespace histograms
{
float estimateEnergy(const Object3d &object, const cv::Mat3b &frame, const glm::mat4 &pose);
}

histograms::Mesh Tests::getPyramidMesh()
{
    glm::vec3 v0(-1, -1, 0);
    glm::vec3 v1(-1, 1, 0);
    glm::vec3 v2(1, 1, 0);
    glm::vec3 v3(1, -1, 0);
    glm::vec3 v4(0, 0, 2);

    std::vector<glm::vec3> vertices;
    vertices.push_back(v0);
    vertices.push_back(v1);
    vertices.push_back(v2);
    vertices.push_back(v3);
    vertices.push_back(v4);

    glm::uvec3 f0(3, 4, 0);
    glm::uvec3 f1(0, 4, 1);
    glm::uvec3 f2(1, 4, 2);
    glm::uvec3 f3(2, 4, 3);
    glm::uvec3 f4(2, 3, 1);
    glm::uvec3 f5(3, 0, 1);
    std::vector<glm::uvec3> faces;
    faces.push_back(f0);
    faces.push_back(f1);
    faces.push_back(f2);
    faces.push_back(f3);
    faces.push_back(f4);
    faces.push_back(f5);

    return histograms::Mesh(vertices, faces);
}

glm::mat4 Tests::getPyramidCameraMatrix()
{
    return glm::mat4(650.048, 0, 0, 0,
                     0.0, 647.183, 0, 0,
                     -324.328, -257.323, -1, -1,
                     0, 0, -0.2, 0);
}

glm::mat4 Tests::getPyramidPose()
{
    return glm::mat4(1, 0, 0, 0,
                     0, -1, 0, 0,
                     0, 0, -1, 0,
                     0, 0, -20, 1
    );
}

Maps Tests::projectPyramid()
{
    histograms::Mesh mesh = getPyramidMesh();
    glm::mat4 camera_matrix = getPyramidCameraMatrix();
    int width = 640;
    int height = 512;
    Renderer renderer(camera_matrix, 16, 160000, width, height);
    glm::mat4 pose = getPyramidPose();
    cv::Mat3b color = cv::Mat3b::zeros(height, width);
    Maps maps = Maps(color);
    renderer.projectMesh(mesh, pose, maps);
    return maps;
}

void Tests::testProjection()
{
    Maps maps = projectPyramid();
    const cv::Mat& depth = maps.depth_map;
    bool ok = true;
    for (int row = 0; row < depth.rows; ++row)
    {
        for (int col = 0; col < depth.cols; ++col)
        {
            if (row > 224 && row < 290 && col > 291 && col < 358)
            {
                ok &= (depth.at<float>(row, col) == 20);
            }
            else
            {
                ok &= (depth.at<float>(row, col) == std::numeric_limits<float>::max());
            }
        }
    }
    if (ok)
    {
        std::cout << "Correct" << std::endl;
    }
    else
    {
        std::cout << "Incorrect" << std::endl;
    }
}

void Tests::testGetROI()
{
    Maps maps = projectPyramid();

    cv::Rect roi = maps.getExtendedROI(40);
    if (roi.x == 252 && roi.y == 185 && roi.width == 146 && roi.height == 145)
    {
        std::cout << "Correct" << std::endl;
    }
    else
    {
        std::cout << "Incorrect" << std::endl;
    }
}

void Tests::testHeaviside()
{
    Maps maps = projectPyramid();
    const cv::Mat& heaviside = maps.heaviside;
    const cv::Mat& signed_distance = maps.signed_distance;
    bool ok = true;
    for (int row = 0; row < heaviside.rows; ++row)
    {
        for (int col = 0; col < heaviside.cols; ++col)
        {
            float signed_distance_value = signed_distance.at<float>(row, col);
            if (signed_distance_value == -8.0f)
            {
                float heaviside_value = heaviside.at<float>(row, col);
                ok &= (heaviside_value > 0.96696183 && heaviside_value < 0.96696189);
            }
            if (signed_distance_value == -7.0f)
            {
                float heaviside_value = heaviside.at<float>(row, col);
                ok &= (heaviside_value > 0.96228344 && heaviside_value < 0.9622835);
            }
            if (signed_distance_value >= -6.5f)
            {
                if (signed_distance_value < -5.5f)
                {
                    float heaviside_value = heaviside.at<float>(row, col);
                }
                else if (signed_distance_value < -4.5f)
                {
                    float heaviside_value = heaviside.at<float>(row, col);
                }
                else if (signed_distance_value < -3.5f)
                {
                    float heaviside_value = heaviside.at<float>(row, col);
                }
                else if (signed_distance_value < -2.5f)
                {
                    float heaviside_value = heaviside.at<float>(row, col);
                }
                else if (signed_distance_value < -1.5f)
                {
                    float heaviside_value = heaviside.at<float>(row, col);
                }
                else if (signed_distance_value < -0.5f)
                {
                    float heaviside_value = heaviside.at<float>(row, col);
                }
                else if (signed_distance_value < 0.5f)
                {
                    float heaviside_value = heaviside.at<float>(row, col);
                }
                else if (signed_distance_value < 1.5f)
                {
                    float heaviside_value = heaviside.at<float>(row, col);
                }
                else if (signed_distance_value < 2.5f)
                {
                    float heaviside_value = heaviside.at<float>(row, col);
                }
                else if (signed_distance_value < 3.5f)
                {
                    float heaviside_value = heaviside.at<float>(row, col);
                }
                else if (signed_distance_value < 4.5f)
                {
                    float heaviside_value = heaviside.at<float>(row, col);
                }
                else if (signed_distance_value < 5.5f)
                {
                    float heaviside_value = heaviside.at<float>(row, col);
                }
                else if (signed_distance_value < 6.5f)
                {
                    float heaviside_value = heaviside.at<float>(row, col);
                }
                else if (signed_distance_value < 7.5f)
                {
                    float heaviside_value = heaviside.at<float>(row, col);
                }
                else if (signed_distance_value < 8.5f)
                {
                    float heaviside_value = heaviside.at<float>(row, col);
                }
            }
        }
    }
    if (ok)
    {
        std::cout << "Correct" << std::endl;
    }
    else
    {
        std::cout << "Incorrect" << std::endl;
    }
}

void Tests::testUpdateHistograms()
{
    histograms::Mesh mesh = getPyramidMesh();
    glm::mat4 camera_matrix = getPyramidCameraMatrix();
    Renderer renderer(camera_matrix, 16, 160000, 640, 512);
    glm::mat4 pose = getPyramidPose();
   histograms::Object3d object3D = histograms::Object3d(mesh, renderer);
    cv::Mat blue_image = cv::imread("/Users/vladislav.platonov/repo/RBOT2/RBOT/data/primitive/rgb/0002.png");

    object3D.updateHistograms(blue_image, pose);
}

void Tests::testEvaluateEnergy()
{
    histograms::Mesh mesh = getPyramidMesh();
    glm::mat4 camera_matrix = getPyramidCameraMatrix();
    Renderer renderer(camera_matrix, 16, 160000, 640, 512);
    glm::mat4 pose = getPyramidPose();
    histograms::Object3d object3D = histograms::Object3d(mesh, renderer);
    cv::Mat blue_image = cv::imread("/Users/vladislav.platonov/repo/RBOT2/RBOT/data/primitive/rgb/0002.png");
    object3D.updateHistograms(blue_image, pose);
    float err = estimateEnergy(object3D, blue_image, pose);
    std::cout << err << std::endl;
    glm::mat4 translated = glm::translate(pose, glm::vec3(0, 0, 1));
    glm::mat4 rotated = glm::rotate(pose, 0.5f, glm::vec3(0, 1, 0));
    float err2 = estimateEnergy(object3D, blue_image, translated);
    std::cout << err2 << std::endl;
    float err3 = estimateEnergy(object3D, blue_image, rotated);
    std::cout << err3 << std::endl;
    if (err < 0.05 && err < err2 && err < err3)
    {
        std::cout << "Correct" << std::endl;
    }
    else
    {
        std::cout << "Incorrect" << std::endl;
    }
}

void Tests::runTests()
{
    testProjection();
    testGetROI();
    testHeaviside();
    testUpdateHistograms();
    testEvaluateEnergy();

}
