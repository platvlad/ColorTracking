#ifndef HISTOGRAMS_OBJECT3D_H
#define HISTOGRAMS_OBJECT3D_H

#include <glm/vec3.hpp>
#include <vector>
#include <mesh.h>
#include <glm/mat4x4.hpp>

#include "renderer.h"

#include "histogram.h"
#include "CircleWindow.h"

namespace histograms
{
    class Object3d
    {
        static const float lambda;
        float dist_to_contour;
        Renderer renderer;
        Mesh mesh;
        std::vector<Histogram> histograms;
        unsigned int histogram_radius;
        CircleWindow circle_window;

    public:
        Object3d(const Mesh& mesh, const Renderer& renderer);

        Object3d();

        void updateHistograms(const cv::Mat& frame, const glm::mat4& pose);

        const Mesh& getMesh() const;

        const Renderer& getRenderer() const;

        std::vector<Histogram>& getHistograms();

        unsigned int getHistogramRadius() const;
    };
}
#endif //HISTOGRAMS_OBJECT3D_H
