#pragma once

#include <opencv2/core/core.hpp>

namespace histograms
{

    class Histogram2
    {
        static const unsigned int COLORS_PER_CHANNEL = 32;

        static const unsigned int bin_size = 256.0 / COLORS_PER_CHANNEL;

        float prob_fg[COLORS_PER_CHANNEL][COLORS_PER_CHANNEL][COLORS_PER_CHANNEL];
        float prob_bg[COLORS_PER_CHANNEL][COLORS_PER_CHANNEL][COLORS_PER_CHANNEL];

        static float alpha_f;
        static float alpha_b;

        float eta_f;
        float eta_b;

        bool visited;

        float get_eta_f(const std::vector<std::pair<cv::Vec3b, float> > &color_heaviside);

        void scaleHistogram();

    public:
        Histogram2();

        void update(const std::vector<std::pair<cv::Vec3b, float> > &color_heaviside);

        float voteColor(const cv::Vec3b &color, float eta_f, float eta_b) const;
    };

}