#ifndef HISTOGRAMS_HISTOGRAM_H
#define HISTOGRAMS_HISTOGRAM_H

#include <opencv2/core/hal/interface.h>

namespace histograms
{
    class Histogram
    {
        static const unsigned int COLORS_PER_CHANNEL = 32;
        const unsigned int radius;
        float prob_fg[COLORS_PER_CHANNEL][COLORS_PER_CHANNEL][COLORS_PER_CHANNEL];
        float prob_bg[COLORS_PER_CHANNEL][COLORS_PER_CHANNEL][COLORS_PER_CHANNEL];
        bool first_visit;
        bool visited;

        float num_foreground;
        float num_background;

        float eta_f;
        float eta_b;

        static float alpha_f;
        static float alpha_b;

    public:
        explicit Histogram(unsigned int radius);

        void setForegroundBackground(float foreground, float background);

        void incrementColor(uchar blue, uchar green, uchar red, float heaviside);

        void resetCurrentForegroundBackground();

        bool isVisited() const;

        float voteForeground(uchar blue, uchar green, uchar red) const;

    };
}

#endif //HISTOGRAMS_HISTOGRAM_H
