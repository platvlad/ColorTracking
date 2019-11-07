#ifndef HISTOGRAMS_HISTOGRAM_H
#define HISTOGRAMS_HISTOGRAM_H

#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
#include <map>
#include "CircleWindow.h"

namespace histograms
{
    class Histogram
    {
        static const unsigned int COLORS_PER_CHANNEL = 32;

        float prob_fg[COLORS_PER_CHANNEL][COLORS_PER_CHANNEL][COLORS_PER_CHANNEL];
        float prob_bg[COLORS_PER_CHANNEL][COLORS_PER_CHANNEL][COLORS_PER_CHANNEL];
        bool visited;

        static float alpha_f;
        static float alpha_b;

        static std::map<int, const Mask> window_masks;

    public:
        static const unsigned int radius = 40;

        explicit Histogram();

        void update(const Projection &local_square, int center_x, int center_y);

        static std::pair<float, float> get_eta_f_eta_b(const Projection &local_square, int center_x, int center_y, int circle_radius = radius);

        bool isVisited() const;

        void votePatch(const Projection &local_square, int center_x, int center_y, cv::Mat1f &votes, cv::Mat1i &numVoters, int circle_radius = radius) const;

        float voteColor(uchar blue, uchar green, uchar red, float eta_f, float eta_b) const;

    };
}

#endif //HISTOGRAMS_HISTOGRAM_H
