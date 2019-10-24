#ifndef HISTOGRAMS_POSEESTIMATOR_H
#define HISTOGRAMS_POSEESTIMATOR_H

#include "Object3d.h"

namespace histograms
{
    class PoseEstimator
    {
        cv::Mat1f votes_foreground;
        cv::Mat1i num_voters;
        cv::Mat1f signed_distance;
        cv::Mat1f heaviside;
        cv::Mat1f derivative_const_part;
        const Renderer* renderer;
        cv::Rect roi;
        bool on_downsampled_frame;
        double getDirac(int row, int col) const;

    public:
        float estimateEnergy(const Object3d &object, const cv::Mat3b &frame, const glm::mat4 &pose, int histo_part = 1, bool debug_info = false);

        const cv::Mat1f& getDerivativeConstPart();

        const cv::Rect& getROI() const;

        const Renderer* getRenderer() const;

        const cv::Mat1i& getNumVoters() const;

        ~PoseEstimator();
    };
}
#endif //HISTOGRAMS_POSEESTIMATOR_H
