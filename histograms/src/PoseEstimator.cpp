#include <set>

#include "Object3d.h"

namespace histograms
{
    float estimateEnergy(Object3d &object, const cv::Mat& frame, const glm::mat4& pose)
    {
        const cv::Mat3b discretized_color = cv::Mat3b(frame.size());
        frame.convertTo(discretized_color, CV_8UC3);
        const Mesh& mesh = object.getMesh();
        const Renderer& renderer = object.getRenderer();
        Maps maps = renderer.projectMesh(mesh, pose);
        int histogram_radius = object.getHistogramRadius();
        cv::Rect roi = maps.getExtendedROI(histogram_radius);
        const cv::Mat1f& signed_distance = maps.signed_distance(roi);

        std::vector< std::vector <std::vector<const Histogram* > > > histogram_centers_on_image(roi.height);
        for (int row = 0; row < roi.height; ++row)
        {
            histogram_centers_on_image[row] = std::vector< std::vector<const Histogram*> >(roi.width);
            for (int column = 0; column < roi.width; ++column)
            {
                histogram_centers_on_image[row][column] = std::vector<const Histogram *>();
            }
        }

        const std::vector<glm::vec3>& vertices = mesh.getVertices();
        std::vector<Histogram>& histograms = object.getHistograms();
        const cv::Mat1f& heaviside = maps.heaviside(roi);
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
                    int patch_x = std::max(0, roi_column - histogram_radius);
                    int patch_y = std::max(0, roi_row - histogram_radius);
                    int patch_x_right = std::min(heaviside.cols, roi_column + histogram_radius + 1);
                    int patch_y_down = std::min(heaviside.rows, roi_row + histogram_radius + 1);
                    cv::Rect patch = cv::Rect(patch_x, patch_y, patch_x_right - patch_x, patch_y_down - patch_y);
                    cv::Mat1f heaviside_on_patch = heaviside(patch);
                    histogram_centers_on_image[roi_row][roi_column].push_back(&histograms[i]);
                    histograms[i].updateEtaFEtaB(heaviside_on_patch, roi_column - patch_x, roi_row - patch_y);
                }
            }
        }

        const cv::Mat1b& color = discretized_color(roi);

        CircleWindow circle_window = CircleWindow(histogram_radius);
        unsigned int window_size = circle_window.getWindowSize();
        const std::vector<int>& edge_curve = circle_window.getEdgeCurve();

        float error_sum = 0;
        int num_error_estimators = 0;

        for (int row = 0; row < roi.height; ++row)
        {
            std::set<const Histogram*> histograms_set = std::set<const Histogram*>();
            std::vector<const Histogram*> histograms_vector = histogram_centers_on_image[row][histogram_radius];
            for (size_t i = 0; i < histograms_vector.size(); ++i)
            {
                histograms_set.insert(histograms_vector[i]);
            }

            for (int column = 0; column < roi.width; ++ column)
            {
                if (row == 0 && column == 193)
                {
                    uchar real_blue = color.at<cv::Vec3b>(row, column)[0];
                    uchar real_green = color.at<cv::Vec3b>(row, column)[1];
                    uchar real_red = color.at<cv::Vec3b>(row, column)[2];
                }
                uchar blue = color.at<cv::Vec3b>(row, column)[0] / 8;
                uchar green = color.at<cv::Vec3b>(row, column)[1] / 8;
                uchar red = color.at<cv::Vec3b>(row, column)[2] / 8;
                float heaviside_value = heaviside.at<float>(row, column);
                float histo_vote_foreground = 0;
                int num_voters = 0;
                for (std::set<const Histogram*>::iterator it = histograms_set.begin(); it != histograms_set.end(); ++it)
                {
                    float foreground_vote = (*it)->voteForeground(blue, green, red);
                    if (foreground_vote >= 0)
                    {
                        histo_vote_foreground += foreground_vote;
                        ++num_voters;
                    }
                }
                if (num_voters)
                {
                    histo_vote_foreground /= num_voters;
                    float histo_vote_background = 1 - histo_vote_foreground;
                    float new_error = -log(heaviside_value * histo_vote_foreground +
                                          (1 - heaviside_value) * histo_vote_background);
                    if (new_error > 0.15f)
                    {

                    }
                    error_sum -= log(heaviside_value * histo_vote_foreground +
                            (1 - heaviside_value) * histo_vote_background);
                    ++num_error_estimators;
                }
                //move histogram set
                for (unsigned int edge_row = 0; edge_row < window_size; ++edge_row)
                {
                    int edge_row_on_image = row + edge_row - histogram_radius;
                    if (edge_row_on_image >= 0 && edge_row_on_image < roi.height)
                    {
                        int left_edge_column_on_image = column - edge_curve[edge_row];
                        if (left_edge_column_on_image >= 0)
                        {
                            std::vector<const Histogram *> &histograms_on_left_edge_pixel =
                                    histogram_centers_on_image[edge_row_on_image][left_edge_column_on_image];
                            for (size_t i = 0; i < histograms_on_left_edge_pixel.size(); ++i)
                            {
                                histograms_set.erase(histograms_on_left_edge_pixel[i]);
                            }
                        }
                        int right_edge_column_on_image = column + edge_curve[edge_row] + 1;
                        if (right_edge_column_on_image < roi.width)
                        {
                            std::vector<const Histogram*>& histograms_on_right_edge_pixel =
                                    histogram_centers_on_image[edge_row_on_image][right_edge_column_on_image];
                            for (size_t i = 0; i < histograms_on_right_edge_pixel.size(); ++i)
                            {
                                histograms_set.insert(histograms_on_right_edge_pixel[i]);
                            }
                        }
                    }
                }
            }
        }

        if (num_error_estimators)
        {
            return error_sum / num_error_estimators;
        }
        return std::numeric_limits<float>::max();
    }
}