#pragma once

#include <utility>
#include <glm/mat4x4.hpp>

#include "Object3d2.h"

namespace histograms
{
    class PoseEstimator2
    {
        glm::mat4 pose;
        Projection projection;
        cv::Mat1f derivative_const_part;
        cv::Mat1f votes_fg;
        int frame_offset;
        const Renderer* renderer;

        double getDirac(int row, int col) const;


    public:
        PoseEstimator2(): pose(glm::mat4()) {}

        std::pair<float, size_t> estimateEnergy(const Object3d2 &object, const cv::Mat3b &frame, 
            const glm::mat4 &pose, int histo_part = 1, int debug_number = 0);
        
        const cv::Mat1f& getDerivativeConstPart();

        const Projection& getProjection() const;

        const Renderer* getRenderer() const;

        const glm::mat4 getPose() const;

        int getFrameOffset() const;

    };
}