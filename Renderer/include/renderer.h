#ifndef RENDERER_RENDERER_H
#define RENDERER_RENDERER_H

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


    void renderTriangle(Maps& maps, int color, glm::vec3& p0, glm::vec3& p1, glm::vec3& p2) const;

    static void updateROI(cv::Rect& roi, int x, int y);

    static void invertMask(cv::Mat1b& mask);

    static void computeSignedDistance(cv::Mat1b& mask, cv::Mat1f& signed_distance);

    static void computeHeaviside(cv::Mat1f& signed_distance, cv::Mat1f& heaviside);

    static void roundXY(glm::vec3& point);

public:
    Renderer(const glm::mat4& camera_matrix, float z_near, float z_far, size_t width, size_t height);

    Renderer();

    Maps projectMesh(const histograms::Mesh& mesh, const glm::mat4& pose) const;

    glm::vec3 projectVertex(const glm::vec3& vertex, const glm::mat4& pose) const;

//    const cv::Mat1f& getDepthMap() const;
//
//    const cv::Mat1f& getSignedDistance() const;
//
//    const cv::Mat1f& getHeaviside() const;
//
//    const cv::Mat1b& getMask() const;
//
//    cv::Rect2i getROI(int frame_size = 0) const;

    size_t getWidth() const;

};

#endif //RENDERER_RENDERER_H