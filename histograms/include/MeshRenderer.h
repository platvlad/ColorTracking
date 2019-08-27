#ifndef HISTOGRAMS_MESHRENDERER_H
#define HISTOGRAMS_MESHRENDERER_H

#include <glm/mat4x4.hpp>
#include "mesh.h"

namespace histograms
{
    class MeshRenderer
    {
    public:
        MeshRenderer(const Mesh &mesh, const glm::mat4 &cameraMatrix, float zNear, float zFar, int width, int height);

    private:
        Mesh mesh;
        glm::mat4 camera_matrix;
        float z_near;
        float z_far;
        int width;
        int height;
    };
}
#endif //HISTOGRAMS_MESHRENDERER_H
