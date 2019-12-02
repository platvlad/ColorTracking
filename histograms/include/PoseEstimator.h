#ifndef HISTOGRAMS_POSEESTIMATOR_H
#define HISTOGRAMS_POSEESTIMATOR_H

#include "Object3d.h"

namespace histograms
{
    class PoseEstimator
    {
        cv::Mat1f votes_foreground;

    private:
        cv::Mat1i num_voters;
        cv::Mat1f derivative_const_part;
        Projection projection;
        glm::mat4 object_pose;

        const Renderer* renderer;
        bool on_downsampled_frame;
        double getDirac(int row, int col) const;

    public:

        PoseEstimator();

        float estimateEnergy(const Object3d &object, const cv::Mat3b &frame, const glm::mat4 &pose, int histo_part = 1, bool debug_info = false);

        const cv::Mat1f& getDerivativeConstPart();

        const Renderer* getRenderer() const;

        const cv::Mat1f &getVotesForeground() const;

        const cv::Mat1i& getNumVoters() const;

        const Projection& getProjection() const;

        const glm::mat4& getPose() const;

        ~PoseEstimator();
    };
}
#endif //HISTOGRAMS_POSEESTIMATOR_H
