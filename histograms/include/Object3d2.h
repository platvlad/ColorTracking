#ifndef HISTOGRAMS_OBJECT3D2_H
#define HISTOGRAMS_OBJECT3D2_H

#include <glm/vec3.hpp>
#include <vector>
#include <mesh.h>
#include <glm/mat4x4.hpp>

#include "renderer.h"

#include "histogram2.h"
#include "CircleWindow.h"

namespace histograms
{
    class Object3d2
    {
        static const float lambda;
        int frame_offset;
        Renderer renderer;
        Mesh mesh;
        std::vector<Histogram2> histograms;

        Histogram2* common_histogram;

        int getHistogram(cv::Vec3f transformed_pt, const glm::mat4 &pose_inv) const;

        static cv::Mat3f getTransformedBorderPoints(const Projection &projection);

    public:
        Object3d2(const Mesh& mesh = Mesh(), const Renderer& renderer = Renderer());

        Object3d2(const Object3d2 &other);

        Object3d2& operator=(const Object3d2 &other);

        void updateHistograms(const cv::Mat3b& frame, const glm::mat4& pose, bool debug_info = false);

        const Mesh& getMesh() const;

        int getFrameOffset() const;

        const Renderer& getRenderer() const;

        cv::Mat1f findColorForeground(
            const Projection &projection, 
            const cv::Mat3b &frame, 
            const glm::mat4 &pose, 
            int debug_number = 0) const;

        ~Object3d2();
    };
}
#endif //HISTOGRAMS_OBJECT3D2_H
