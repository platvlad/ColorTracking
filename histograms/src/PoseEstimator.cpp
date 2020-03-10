#include <set>
#include <fstream>
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <PoseEstimator.h>
#include <mesh.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <cmath>


namespace histograms
{
    std::pair<float, size_t> PoseEstimator::estimateEnergy(const Object3d &object, const cv::Mat3b &frame, const glm::mat4 &pose, int histo_part, bool debug_info)
    {
        object_pose = pose;
        const Mesh& mesh = object.getMesh();
        const Renderer& object_renderer = object.getRenderer();
        on_downsampled_frame = frame.size() != object_renderer.getSize();
        renderer = on_downsampled_frame ?
                new Renderer(object_renderer, frame.size())
                : &object_renderer;

        float frame_size_scale = static_cast<float>(renderer->getWidth()) /
                                 static_cast<float>(object_renderer.getWidth());

        int histogram_radius = std::ceil(static_cast<float>(object.getHistogramRadius()) * frame_size_scale);
        projection = renderer->projectMesh2(mesh, pose, frame, histogram_radius);

        cv::Mat1f& signed_distance = projection.signed_distance;
        cv::Size projection_size = projection.getSize();
        votes_foreground = cv::Mat1f::zeros(projection_size);
        num_voters = cv::Mat1i::zeros(projection_size);

        const std::vector<glm::vec3>& vertices = mesh.getVertices();
        const std::vector<Histogram>& histograms = object.getHistograms();
        cv::Mat3b color_copy;
        if (debug_info)
        {
            color_copy = projection.color_map.clone();
        }
        for (size_t i = 0; i < vertices.size(); i += histo_part)
        {
            glm::vec3 pixel = projection.vertex_projections[i];
            int column = static_cast<int>(pixel.x);
            int row = static_cast<int>(pixel.y);
            if (column > 0 && column < projection_size.width && row >= 0 && row < projection_size.height)
            {
//                if (pixel.z <= maps.depth_map.at<float>(pixel.y, pixel.x))
//                {
//                    histogram_centers_on_image[roi_row][roi_column].push_back(&histograms[i]);
//                }
                if (abs(signed_distance(row, column)) < 5)
                {
                    if (debug_info) {
                        color_copy(row, column) = cv::Vec3b(0, 0, 0);
                    }
                    int center_on_patch_x = std::min(column, histogram_radius);
                    int center_on_patch_y = std::min(row, histogram_radius);
                    cv::Rect patch_square = projection.getPatchSquare(column, row);
                    Projection patch = projection(patch_square);
                    //Projection square_patch = maps.getSquarePatch(column, row, histogram_radius);
                    cv::Mat1f votes_on_patch = votes_foreground(patch_square);
                    cv::Mat1i num_voters_on_patch = num_voters(patch_square);
                    if (histograms[i].isVisited())
                    {
                        histograms[i].votePatch(patch, center_on_patch_x, center_on_patch_y, votes_on_patch,
                                                num_voters_on_patch, histogram_radius);
                    }
                }
            }
        }
        float error_sum = 0;
        int num_error_estimators = 0;
        cv::Mat1f& heaviside = projection.heaviside;

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
                        projection.color_map);
            cv::imwrite("/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/debug_frames/color_copy.png", color_copy);
            cv::imwrite("/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/debug_frames/signed_distance.png", projection.signed_distance);
            std::ofstream fout("/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/debug_frames/signed_distance.txt");
            for (int row = 0; row < projection.signed_distance.rows; ++row)
            {
                for (int col = 0; col < projection.signed_distance.cols; ++col)
                {
                    fout << projection.signed_distance(row, col) << ' ';
                }
                fout << std::endl;
            }
        }

        if (num_error_estimators)
        {
            return std::pair<float, size_t>(error_sum / static_cast<float>(num_error_estimators), num_error_estimators);
        }
        return std::pair<float, size_t>(std::numeric_limits<float>::max(), 0);
    }

    double PoseEstimator::getDirac(int row, int col) const
    {
        float s = 1.2;
        const cv::Mat1f& signed_distance = projection.signed_distance;
        return s / (M_PI * signed_distance(row, col) * signed_distance(row, col) * s * s + M_PI);
    }

    const cv::Mat1f& PoseEstimator::getDerivativeConstPart()
    {
        if (derivative_const_part.empty())
        {
            const cv::Mat1f& heaviside = projection.heaviside;
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

    const Projection &PoseEstimator::getProjection() const
    {
        return projection;
    }

    PoseEstimator::PoseEstimator() : projection(),
                                     renderer(nullptr),
                                     on_downsampled_frame(false)
    {
    }

    const glm::mat4 &PoseEstimator::getPose() const
    {
        return object_pose;
    }

    const cv::Mat1f &PoseEstimator::getVotesForeground() const
    {
        return votes_foreground;
    }

}
