#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "PoseEstimator2.h"

namespace histograms
{
    std::pair<float, size_t> PoseEstimator2::estimateEnergy(const Object3d2 &object, const cv::Mat3b &frame,
        const glm::mat4 &pose, int histo_part, int debug_number)
    {
        this->pose = pose;
        renderer = &(object.getRenderer());
        const Mesh& mesh = object.getMesh();
        frame_offset = object.getFrameOffset();
        projection = renderer->projectMesh2(mesh, pose, frame, frame_offset);
        cv::Mat1f& signed_distance = projection.signed_distance;
        votes_fg = object.findColorForeground(projection, frame, pose, debug_number);
        
        float error_sum = 0;
        size_t num_estimators = 0;
        cv::Mat1f errors = cv::Mat1f::zeros(projection.getSize());
        for (int row = 0; row < votes_fg.rows; ++row)
        {
            for (int col = 0; col < votes_fg.cols; ++col)
            {
                if (abs(signed_distance(row, col)) <= frame_offset)
                {
                    float heaviside_value = projection.heaviside(row, col);
                    float foreground_vote = votes_fg(row, col);
                    if (debug_number > 0)
                    {
                        float current_error = -log(heaviside_value * foreground_vote +
                            (1 - heaviside_value) * (1 - foreground_vote));
                        errors(row, col) = current_error * 255 / 2;
                        error_sum += current_error;
                    }
                    else
                    {
                        error_sum -= log(heaviside_value * foreground_vote +
                            (1 - heaviside_value) * (1 - foreground_vote));
                    }
                    ++num_estimators;
                }
            }
        }

        if (debug_number > 0)
        {
            cv::Mat1b heaviside_copy = cv::Mat1b();
            cv::imwrite("C:\\MyProjects\\repo\\VSProjects\\ColorTracking\\ColorTracking\\build\\data\\debug_frames\\errors" + 
                std::to_string(debug_number) + ".png",
                errors);
            cv::imwrite("C:\\MyProjects\\repo\\VSProjects\\ColorTracking\\ColorTracking\\build\\data\\debug_frames\\color" + 
                std::to_string(debug_number) + ".png",
                projection.color_map);
        }

        if (num_estimators > 0)
        {
            return std::pair<float, size_t>(error_sum / num_estimators, num_estimators);
        }
        return std::pair<float, size_t>(std::numeric_limits<float>::max(), 0);
    }

    double PoseEstimator2::getDirac(int row, int col) const
    {
        float s = 1.2;
        const cv::Mat1f& signed_distance = projection.signed_distance;
        return s / (M_PI * signed_distance(row, col) * signed_distance(row, col) * s * s + M_PI);
    }

    const cv::Mat1f& PoseEstimator2::getDerivativeConstPart()
    {
        if (derivative_const_part.empty())
        {
            cv::Mat1f& heaviside = projection.heaviside;
            cv::Mat1f& signed_distance = projection.signed_distance;
            derivative_const_part = cv::Mat1f(heaviside.size(), -1);
            for (int row = 0; row < derivative_const_part.rows; ++row)
            {
                for (int col = 0; col < derivative_const_part.cols; ++col)
                {
                    if (abs(signed_distance(row, col)) <= frame_offset)
                    {
                        float pf = votes_fg(row, col);
                        derivative_const_part(row, col) =
                            (2 * pf - 1) * getDirac(row, col) / (heaviside(row, col) * (2 * pf - 1) + 1 - pf);
                    }
                }
            }
        }
        return derivative_const_part;
    }

    const Projection& PoseEstimator2::getProjection() const
    {
        return projection;
    }
    const Renderer* PoseEstimator2::getRenderer() const
    {
        return renderer;
    }
    const glm::mat4 PoseEstimator2::getPose() const
    {
        return pose;
    }

    int PoseEstimator2::getFrameOffset() const
    {
        return frame_offset;
    }
}