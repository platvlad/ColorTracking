#include <set>
#include <opencv2/imgcodecs.hpp>
#include <PoseEstimator.h>


namespace histograms
{
    float PoseEstimator::estimateEnergy(const Object3d &object, const cv::Mat3b &frame, const glm::mat4 &pose, int histo_part, bool debug_info)
    {
        const Mesh& mesh = object.getMesh();
        const Renderer& object_renderer = object.getRenderer();
        on_downsampled_frame = frame.size() != object_renderer.getSize();
        renderer = on_downsampled_frame ?
                new Renderer(object_renderer, frame.size())
                : &object_renderer;

        float frame_size_scale = static_cast<float>(renderer->getWidth()) /
                                 static_cast<float>(object_renderer.getWidth());

        Maps maps = Maps(frame);
        renderer->projectMesh(mesh, pose, maps);
        int histogram_radius = std::ceil(static_cast<float>(object.getHistogramRadius()) * frame_size_scale);
        roi = maps.getExtendedROI(histogram_radius);
        Maps maps_on_roi = maps(roi);
        signed_distance = maps_on_roi.signed_distance;
        votes_foreground = cv::Mat1f::zeros(roi.size());
        num_voters = cv::Mat1i::zeros(roi.size());

        const std::vector<glm::vec3>& vertices = mesh.getVertices();
        const std::vector<Histogram>& histograms = object.getHistograms();
        cv::Mat3b color_copy;
        if (debug_info)
        {
            color_copy = maps_on_roi.color_map.clone();
        }
        for (size_t i = 0; i < vertices.size(); i += histo_part)
        {
            glm::vec3 pixel = renderer->projectVertex(vertices[i], pose);
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
                    if (debug_info) {
                        color_copy(roi_row, roi_column) = cv::Vec3b(0, 0, 0);
                    }
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
        heaviside = maps_on_roi.heaviside;

        cv::Mat1f errors;
        if (debug_info)
        {
            errors = cv::Mat1f::zeros(heaviside.size());
        }

        for (int row = 0; row < votes_foreground.rows; ++row)
        {
            for (int col = 0; col < votes_foreground.cols; ++col)
            {
                if (num_voters(row, col)) {
                    float foreground_vote = votes_foreground(row, col) / num_voters(row, col);
                    float heaviside_value = heaviside(row, col);
                    if (debug_info)
                    {
                        float current_error = -log(heaviside_value * foreground_vote +
                                                   (1 - heaviside_value) * (1 - foreground_vote));
                        errors(row, col) = current_error * 255 / 2;
                    }
                    error_sum -= log(heaviside_value * foreground_vote +
                                     (1 - heaviside_value) * (1 - foreground_vote));

                    ++num_error_estimators;
                }
            }
        }
        if (debug_info)
        {
            cv::Mat1b heaviside_copy = cv::Mat1b();
            heaviside.convertTo(heaviside_copy, CV_8UC1, 255.0);
            cv::imwrite("/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/debug_frames/heaviside.png",
                        heaviside_copy);
            cv::imwrite("/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/debug_frames/errors.png",
                        errors);
            cv::imwrite("/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/debug_frames/color.png",
                        maps_on_roi.color_map);
            cv::imwrite("/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/debug_frames/color_copy.png", color_copy);
        }

        if (num_error_estimators)
        {
            return error_sum / static_cast<float>(num_error_estimators);
        }
        return std::numeric_limits<float>::max();
    }

    double PoseEstimator::getDirac(int row, int col) const
    {
        float s = 1.2;
        return s / (M_PI * signed_distance(row, col) * signed_distance(row, col) * s * s + M_PI);
    }

    const cv::Mat1f& PoseEstimator::getDerivativeConstPart()
    {
        if (derivative_const_part.empty())
        {
            derivative_const_part = cv::Mat1f(heaviside.size(), -1);
            for (int row = 0; row < derivative_const_part.rows; ++row)
            {
                for (int col = 0; col < derivative_const_part.cols; ++col)
                {
                    if (!num_voters(row, col))
                    {
                        derivative_const_part(row, col) = 0;
                    }
                    float pf = votes_foreground(row, col) / num_voters(row, col);
                    derivative_const_part(row, col) =
                            (2 * pf - 1) * getDirac(row, col) / (heaviside(row, col) * (2 * pf - 1) + 1 - pf);
                }
            }
        }
        return derivative_const_part;
    }

    const cv::Rect& PoseEstimator::getROI() const
    {
        return roi;
    }

    PoseEstimator::~PoseEstimator()
    {
        if (on_downsampled_frame)
        {
            delete renderer;
        }
    }

    const Renderer* PoseEstimator::getRenderer() const
    {
        return renderer;
    }

    const cv::Mat1i &PoseEstimator::getNumVoters() const
    {
        return num_voters;
    }


}
