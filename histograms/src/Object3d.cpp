#include <set>
#include <opencv2/imgcodecs.hpp>

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
        Maps maps = Maps(frame);
        renderer.projectMesh(mesh, pose, maps);

        const std::vector<glm::vec3>& vertices = mesh.getVertices();

        cv::Rect roi = maps.getExtendedROI(histogram_radius);

        const Maps maps_on_roi = maps(roi);
        if (maps_on_roi.hasEmptyProjection())
        {
            return;
        }
        const cv::Mat1f& signed_distance = maps.signed_distance;
        const cv::Mat1b& mask = maps.mask;
        const cv::Mat1f& heaviside = maps.heaviside;

        //project vertices
        for (size_t i = 0; i < vertices.size(); ++i)
        {
            glm::vec3 pixel = renderer.projectVertex(vertices[i], pose);
            int column = static_cast<int>(round(pixel.x));
            int row = static_cast<int>(round(pixel.y));
            if (abs(signed_distance.at<float>(row, column)) < dist_to_contour + 0.5f)
            {
                // check visibility?
                int roi_column = column - roi.x;
                int roi_row = row - roi.y;
                // signed distance computed is a bit larger since it is distance from exterior pixels, not from real curve
                // therefore real distance is smaller up to 0.5 px
                if (roi_row >= 0 && roi_row < roi.height && roi_column >= 0 && roi_column < roi.width)
                {
                    int center_on_patch_x = std::min(roi_column, histogram_radius);
                    int center_on_patch_y = std::min(roi_row, histogram_radius);
                    cv::Rect patch_square = maps.getPatchSquare(column, row, histogram_radius);
                    Maps patch = maps(patch_square);

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
