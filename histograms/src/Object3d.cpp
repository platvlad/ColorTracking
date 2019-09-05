#include <set>
#include <iostream>

#include "Object3d.h"

namespace histograms
{
    const float Object3d::lambda = 0.1;

    Object3d::Object3d(const Mesh& mesh, const Renderer& renderer) : mesh(mesh),
                                                                     renderer(renderer)
    {
        size_t default_radius = 40;
        size_t default_width = 640;
        size_t width = renderer.getWidth();
        // histogram_radius = width * default_radius / default_width;
        histogram_radius = default_radius;
        dist_to_contour = lambda * histogram_radius;
        size_t vertex_num = mesh.getVertices().size();
        histograms = std::vector<Histogram>(vertex_num, Histogram(histogram_radius));
        circle_window = CircleWindow(histogram_radius);
    }

    Object3d::Object3d() : mesh(Mesh()),
                           renderer(Renderer())
    {
    }

    void Object3d::updateHistograms(const cv::Mat& frame, const glm::mat4& pose)
    {
        const Maps maps = renderer.projectMesh(mesh, pose);
        const cv::Mat3b discretized_color = cv::Mat3b(frame.size());
        frame.convertTo(discretized_color, CV_8UC3);
        const std::vector<glm::vec3>& vertices = mesh.getVertices();
        cv::Rect roi = maps.getExtendedROI(histogram_radius);
        if (roi.empty())
        {
            return;
        }
        const cv::Mat1f& signed_distance = maps.signed_distance;
        const cv::Mat1b& mask = maps.mask;
        const cv::Mat1f& heaviside = maps.heaviside;

        std::vector< std::vector <std::vector< Histogram* > > > histogram_centers_on_image(roi.height);
        for (int row = 0; row < roi.height; ++row)
        {
            histogram_centers_on_image[row] = std::vector< std::vector<Histogram*> >(roi.width);
            for (int column = 0; column < roi.width; ++column)
            {
                histogram_centers_on_image[row][column] = std::vector<Histogram*>();
            }
        }

        //project vertices
        for (size_t i = 0; i < vertices.size(); ++i)
        {
            glm::vec3 pixel = renderer.projectVertex(vertices[i], pose);
            int column = (int)round(pixel.x);
            int row = (int)round(pixel.y);
            int roi_column = column - roi.x;
            int roi_row = row - roi.y;
            // signed distance computed is a bit larger since it is distance from exterior pixels, not from real curve
            // therefore real distance is smaller up to 0.5 px
            if (roi_row >= 0 && roi_row < roi.height && roi_column >= 0 && roi_column < roi.width)
            {
                if (abs(signed_distance.at<float>(row, column)) < dist_to_contour + 0.5f)
                {
                    histogram_centers_on_image[roi_row][roi_column].push_back(&histograms[i]);
                }
            }
        }

        unsigned int window_size = circle_window.getWindowSize();
        const std::vector<int>& edge_curve = circle_window.getEdgeCurve();

        cv::Mat1b mask_on_roi = mask(roi);
        cv::Mat1f heaviside_on_roi = heaviside(roi);

        //compute number of foreground and background pixels for each histogram
        size_t initial_foreground_pixel_num = 0;
        size_t initial_background_pixel_num = 0;
        for (unsigned int row = 0; row <= histogram_radius; ++row)
        {
            for (int column = 0; column <= edge_curve[row + histogram_radius]; ++column)
            {
                if (mask_on_roi.at<uchar>((int)row, column))
                {
                    ++initial_foreground_pixel_num;
                }
                else
                {
                    ++initial_background_pixel_num;
                }
            }
        }
        for (int row = 0; row < roi.height; ++row)
        {
            size_t foreground_pixel_num = initial_foreground_pixel_num;
            size_t background_pixel_num = initial_background_pixel_num;
            for (int column = 0; column < roi.width; ++column)
            {
                std::vector<Histogram*> pixel_histograms = histogram_centers_on_image[row][column];
                for (size_t i = 0; i < pixel_histograms.size(); ++i)
                {
                    pixel_histograms[i]->setForegroundBackground(foreground_pixel_num, background_pixel_num);
                }
                for (unsigned int edge_row = 0; edge_row < window_size; ++edge_row)
                {
                    int edge_row_on_image = row + (int)edge_row - (int)histogram_radius;
                    if (edge_row_on_image >= 0 && edge_row_on_image < roi.height)
                    {
                        int left_edge_column_on_image = column - edge_curve[edge_row];
                        if (left_edge_column_on_image >= 0 && left_edge_column_on_image < roi.width)
                        {
                            if (mask_on_roi[edge_row_on_image][left_edge_column_on_image])
                            {
                                --foreground_pixel_num;
                            }
                            else
                            {
                                --background_pixel_num;
                            }
                        }
                        int right_edge_column_on_image = column + edge_curve[edge_row] + 1;
                        if (right_edge_column_on_image >= 0 && right_edge_column_on_image < roi.width)
                        {
                            if (mask_on_roi[edge_row_on_image][right_edge_column_on_image])
                            {
                                ++foreground_pixel_num;
                            }
                            else
                            {
                                ++background_pixel_num;
                            }
                        }
                    }
                }
            }
            for (unsigned int edge_column = 0; edge_column <= histogram_radius; ++edge_column)
            {
                int top_edge_row_on_image = row - edge_curve[edge_column + histogram_radius];
                if (top_edge_row_on_image >= 0 && top_edge_row_on_image < roi.height)
                {
                    if (mask_on_roi[top_edge_row_on_image][edge_column])
                    {
                        --initial_foreground_pixel_num;
                    }
                    else
                    {
                        --initial_background_pixel_num;
                    }
                }
                int bottom_edge_row_on_image = row + edge_curve[edge_column + histogram_radius] + 1;
                if (bottom_edge_row_on_image >= 0 && bottom_edge_row_on_image < roi.height)
                {
                    if (mask_on_roi[bottom_edge_row_on_image][edge_column])
                    {
                        ++initial_foreground_pixel_num;
                    }
                    else
                    {
                        ++initial_background_pixel_num;
                    }
                }
            }
        }

        cv::Mat3b color_on_roi = discretized_color(roi);
        std::set<Histogram*> row_beginning_set = std::set<Histogram*>();
        for (int row = 0; row < color_on_roi.rows; ++row)
        {
            std::set<Histogram*> histograms_set = row_beginning_set;
            for (int column = 0; column < color_on_roi.cols; ++column)
            {
                uchar blue = color_on_roi.at<cv::Vec3b>(row, column)[0] / 8;
                uchar green = color_on_roi.at<cv::Vec3b>(row, column)[1] / 8;
                uchar red = color_on_roi.at<cv::Vec3b>(row, column)[2] / 8;
                for (std::set<Histogram*>::iterator it = histograms_set.begin(); it != histograms_set.end(); ++it)
                {
                    (*it)->incrementColor(blue, green, red, heaviside_on_roi[row][column]);
                }
                for (unsigned int edge_row = 0; edge_row < window_size; ++edge_row)
                {
                    int edge_row_on_image = row + edge_row - histogram_radius;
                    if (edge_row_on_image >= 0 && edge_row_on_image < roi.height)
                    {
                        int left_edge_column_on_image = column - edge_curve[edge_row];
                        if (left_edge_column_on_image >= 0)
                        {
                            std::vector<Histogram *> &histograms_on_left_edge_pixel =
                                    histogram_centers_on_image[edge_row_on_image][left_edge_column_on_image];
                            for (size_t i = 0; i < histograms_on_left_edge_pixel.size(); ++i)
                            {
                                histograms_set.erase(histograms_on_left_edge_pixel[i]);
                            }
                        }
                        int right_edge_column_on_image = column + edge_curve[edge_row] + 1;
                        if (right_edge_column_on_image < roi.width)
                        {
                            if (edge_row_on_image == 40 && right_edge_column_on_image == 40)
                            {

                            }
                            std::vector<Histogram*>& histograms_on_right_edge_pixel =
                                    histogram_centers_on_image[edge_row_on_image][right_edge_column_on_image];
                            for (size_t i = 0; i < histograms_on_right_edge_pixel.size(); ++i)
                            {
                                histograms_set.insert(histograms_on_right_edge_pixel[i]);
                            }
                        }
                    }
                }
            }

            for (unsigned int column = histogram_radius; column < window_size; ++column)
            {
                int column_on_image = column - histogram_radius;
                int top_edge_row_on_image = row - edge_curve[column];
                if (top_edge_row_on_image >= 0)
                {
                    std::vector<Histogram*> &histograms_on_top_edge_pixel =
                            histogram_centers_on_image[top_edge_row_on_image][column_on_image];
                    for (size_t i = 0; i < histograms_on_top_edge_pixel.size(); ++i)
                    {
                        row_beginning_set.erase(histograms_on_top_edge_pixel[i]);
                    }
                }
                int bottom_edge_row_on_image = row + edge_curve[column] + 1;
                if (bottom_edge_row_on_image < roi.height)
                {
                    std::vector<Histogram*>& histograms_on_bottom_edge_pixel =
                            histogram_centers_on_image[bottom_edge_row_on_image][column_on_image];
                    for (size_t i = 0; i < histograms_on_bottom_edge_pixel.size(); ++i)
                    {
                        row_beginning_set.insert(&(*histograms_on_bottom_edge_pixel[i]));
                    }
                }
            }

        }

        for (size_t i = 0; i < histograms.size(); ++i)
        {
            if (histograms[i].isVisited())
            {
                histograms[i].resetCurrentForegroundBackground();
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
