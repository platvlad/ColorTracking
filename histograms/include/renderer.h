#ifndef RENDERER_RENDERER_H
#define RENDERER_RENDERER_H

//#include <OpenGL/gl.h>

#include <glm/mat4x4.hpp>
#include <opencv2/core/mat.hpp>

#include "mesh.h"
#include "projection.h"


class Renderer
{
private:

    glm::mat4 camera_matrix;
    float z_near;
    float z_far;
    int width;
    int height;


    void renderTriangle(Projection& maps, const std::vector<glm::vec4> &vertex_transforms, glm::uvec3& face) const;

    static void renderTriangleWireframe(cv::Mat3b& color_map, glm::vec3& p0, glm::vec3& p1, glm::vec3& p2);

    static void updateROI(cv::Rect& roi, int x, int y);

    static void invertMask(cv::Mat1b& mask);

    static void roundXY(glm::vec3& point);

public:
    Renderer(const glm::mat4& camera_matrix, float z_near, float z_far, size_t width, size_t height);

    Renderer();

    Renderer(const Renderer &other, const cv::Size &size);

    Projection projectMesh(const histograms::Mesh& mesh,
                     const glm::mat4& pose,
                     const cv::Mat3b &frame,
                     int frame_offset,
                     bool compute_signed_distance = true) const;

    void renderMesh(const histograms::Mesh& mesh, cv::Mat3b& frame, const glm::mat4& pose) const;

    glm::vec4 transformVertex(const glm::vec3& vertex, const glm::mat4& pose) const;

    glm::vec3 projectTransformedVertex(const glm::vec4& vertex) const;

    size_t getWidth() const;

    cv::Size getSize() const;

    glm::vec2 getFocal() const;

};

#endif //RENDERER_RENDERER_H