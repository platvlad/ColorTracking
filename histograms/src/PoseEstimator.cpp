#include <set>
#include <opencv2/imgcodecs.hpp>

#include "Object3d.h"

namespace histograms
{
    float estimateEnergy(const Object3d &object, const cv::Mat3b &frame, const glm::mat4 &pose)
    {
        const Mesh& mesh = object.getMesh();
        const Renderer& object_renderer = object.getRenderer();
        const Renderer& renderer = frame.size() == object_renderer.getSize() ?
                object_renderer
                : Renderer(object_renderer, frame.size());

        float frame_size_scale = static_cast<float>(renderer.getWidth()) /
                                 static_cast<float>(object_renderer.getWidth());

        Maps maps = Maps(frame);
        renderer.projectMesh(mesh, pose, maps);
        int histogram_radius = std::ceil(static_cast<float>(object.getHistogramRadius()) * frame_size_scale);
        cv::Rect roi = maps.getExtendedROI(histogram_radius);
        Maps maps_on_roi = maps(roi);
        const cv::Mat1f& signed_distance = maps_on_roi.signed_distance;
        cv::Mat1f votes_foreground = cv::Mat1f::zeros(maps_on_roi.heaviside.size());
        cv::Mat1i num_voters = cv::Mat1i::zeros(votes_foreground.size());

        const std::vector<glm::vec3>& vertices = mesh.getVertices();
        const std::vector<Histogram>& histograms = object.getHistograms();
        for (size_t i = 0; i < vertices.size(); ++i)
        {
            glm::vec3 pixel = renderer.projectVertex(vertices[i], pose);
            int column = (int)round(pixel.x);
            int row = (int)round(pixel.y);
            int roi_column = column - roi.x;
            int roi_row = row - roi.y;
            if (roi_column > 0 && roi_column < roi.width && roi_row >= 0 && roi_row < roi.height)
            {
//                if (pixel.z <= maps.depth_map.at<float>(pixel.y, pixel.x))
//                {
//                    histogram_centers_on_image[roi_row][roi_column].push_back(&histograms[i]);
//                }
                if (abs(signed_distance(roi_row, roi_column)) < 5)
                {
                    int center_on_patch_x = std::min(roi_column, histogram_radius);
                    int center_on_patch_y = std::min(roi_row, histogram_radius);
                    cv::Rect patch_square = maps_on_roi.getPatchSquare(roi_column, roi_row, histogram_radius);
                    Maps patch = maps_on_roi(patch_square);
                    //Maps square_patch = maps.getSquarePatch(column, row, histogram_radius);
                    cv::Mat1f votes_on_patch = votes_foreground(patch_square);
                    cv::Mat1i num_voters_on_patch = num_voters(patch_square);
                    if (histograms[i].isVisited())
                    {
                        histograms[i].votePatch(patch, center_on_patch_x, center_on_patch_y, votes_on_patch,
                                                num_voters_on_patch);
                    }
                }
            }
        }

        float error_sum = 0;
        int num_error_estimators = 0;

        const cv::Mat1f& heaviside = maps_on_roi.heaviside;
        for (int row = 0; row < votes_foreground.rows; ++row)
        {
            for (int col = 0; col < votes_foreground.cols; ++col)
            {
                if (num_voters(row, col)) {
                    float foreground_vote = votes_foreground(row, col) / num_voters(row, col);
                    float heaviside_value = heaviside(row, col);
                    error_sum -= log(heaviside_value * foreground_vote +
                                     (1 - heaviside_value) * (1 - foreground_vote));
                    ++num_error_estimators;
                }
            }
        }

        if (num_error_estimators)
        {
            return error_sum / static_cast<float>(num_error_estimators);
        }
        return std::numeric_limits<float>::max();
    }
}
