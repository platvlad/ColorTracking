#include <set>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "mesh.h"
#include "Object3d.h"

namespace histograms
{
    const float Object3d::lambda = 0.1;

    Object3d::Object3d(const Mesh& mesh, const Renderer& renderer) : mesh(mesh),
                                                                     renderer(renderer)
    {
        int default_radius = 40;
        int default_width = 640;
        size_t width = renderer.getWidth();
        //histogram_radius = width * default_radius / default_width;
        histogram_radius = default_radius;
        dist_to_contour = lambda * static_cast<float>(histogram_radius);
        size_t vertex_num = mesh.getVertices().size();
        histograms = std::vector<Histogram>(vertex_num, Histogram());
    }

    Object3d::Object3d() : mesh(Mesh()),
                           renderer(Renderer()),
                           histogram_radius(40),
                           dist_to_contour(lambda * static_cast<float>(histogram_radius))
    {
    }

    void Object3d::updateHistograms(const cv::Mat3b& frame, const glm::mat4& pose)
    {
        Projection projection = renderer.projectMesh2(mesh, pose, frame, histogram_radius);

        const std::vector<glm::vec3>& vertices = mesh.getVertices();

        if (projection.hasEmptyProjection())
        {
            return;
        }
        const cv::Mat1f& signed_distance = projection.signed_distance;
        const cv::Mat1b& mask = projection.mask;
        const cv::Mat1f& heaviside = projection.heaviside;
        cv::Size projection_size = projection.getSize();

        //project vertices
        for (size_t i = 0; i < vertices.size(); ++i)
        {
            glm::vec3 pixel = projection.vertex_projections[i];
//            int column = static_cast<int>(pixel.x);
//            int row = static_cast<int>(pixel.y);
            int column = floor(pixel.x);
            int row = floor(pixel.y);
            if (abs(signed_distance.at<float>(row, column)) < dist_to_contour + 0.5f)
            {
                // check visibility?
                // signed distance computed is a bit larger since it is distance from exterior pixels, not from real curve
                // therefore real distance is smaller up to 0.5 px
                if (row >= 0 && row < projection_size.height && column >= 0 && column < projection_size.width)
                {
                    int center_on_patch_x = std::min(column, histogram_radius);
                    int center_on_patch_y = std::min(row, histogram_radius);
                    cv::Rect patch_square = projection.getPatchSquare(column, row);
                    Projection patch = projection(patch_square);

                    histograms[i].update(patch, center_on_patch_x, center_on_patch_y);

                }
            }
        }

    }

    const Mesh& Object3d::getMesh() const
    {
        return mesh;
    }

    const Renderer& Object3d::getRenderer() const
    {
        return renderer;
    }

    const std::vector<Histogram>& Object3d::getHistograms() const
    {
        return histograms;
    }

    unsigned int Object3d::getHistogramRadius() const
    {
        return histogram_radius;
    }

}
