#ifndef RENDERER_RENDERER_H
#define RENDERER_RENDERER_H

#include <OpenGL/gl.h>

#include <glm/mat4x4.hpp>
#include <opencv2/core/mat.hpp>

#include "mesh.h"
#include "maps.h"


class Renderer
{
private:
    static const float s_heaviside;

    glm::mat4 camera_matrix;
    float z_near;
    float z_far;
    int width;
    int height;


    void renderTriangle(Maps& maps, glm::vec3& p0, glm::vec3& p1, glm::vec3& p2) const;

    static void renderTriangleWireframe(cv::Mat3b& color_map, glm::vec3& p0, glm::vec3& p1, glm::vec3& p2);

    static void updateROI(cv::Rect& roi, int x, int y);

    static void invertMask(cv::Mat1b& mask);

    static void computePixelBoardsMask(cv::Mat1b& origin_mask, cv::Mat1b& board_mask);

    static void computeDistanceOnUpsampled(cv::Mat1b& mask, cv::Mat1f& distance);

    static void computeSignedDistance(cv::Mat1b& mask, cv::Mat1f& signed_distance);

    static void computeHeaviside(cv::Mat1f& signed_distance, cv::Mat1f& heaviside);

    static void roundXY(glm::vec3& point);

public:
    Renderer(const glm::mat4& camera_matrix, float z_near, float z_far, size_t width, size_t height);

    Renderer();

    Renderer(const Renderer &other, const cv::Size &size);

    void projectMesh(const histograms::Mesh& mesh, const glm::mat4& pose, Maps &maps) const;

    void renderMesh(const histograms::Mesh& mesh, cv::Mat3b& frame, const glm::mat4& pose) const;

    glm::vec3 projectVertex(const glm::vec3& vertex, const glm::mat4& pose) const;

    size_t getWidth() const;

    cv::Size getSize() const;

};

#endif //RENDERER_RENDERER_H