#include "histogram.h"

namespace histograms
{
    float Histogram::alpha_f = 0.1;
    float Histogram::alpha_b = 0.2;

    Histogram::Histogram(unsigned int radius) : first_visit(true),
                                                prob_fg(),
                                                prob_bg(),
                                                radius(radius),
                                                num_foreground(0),
                                                num_background(0),
                                                eta_f(0),
                                                eta_b(0),
                                                visited(false)
    {
    }

    void Histogram::setForegroundBackground(float foreground, float background)
    {
        visited = true;
        eta_f = 0;
        eta_b = 0;
        num_foreground = foreground;
        num_background = background;
        for (int i = 0; i < COLORS_PER_CHANNEL; ++i)
        {
            for (int j = 0; j < COLORS_PER_CHANNEL; ++j)
            {
                for (int k = 0; k < COLORS_PER_CHANNEL; ++k)
                {
                    prob_bg[i][j][k] *= (1 - Histogram::alpha_b);
                    prob_fg[i][j][k] *= (1 - Histogram::alpha_f);
                }
            }
        }
    }

    void Histogram::incrementColor(uchar blue, uchar green, uchar red, float heaviside)
    {
        bool foreground = heaviside > 0.5f;
        if (foreground && num_foreground)
        {
            eta_f += heaviside;
            prob_fg[blue][green][red] += !first_visit ? Histogram::alpha_f / num_foreground : 1.0f / num_foreground;
            return;
        }
        if (num_background)
        {
            eta_b += (1 - heaviside);
            prob_bg[blue][green][red] += !first_visit ? Histogram::alpha_b / num_background : 1.0f / num_background;
        }
    }

    void Histogram::resetCurrentForegroundBackground()
    {
        first_visit = false;
        num_foreground = 0;
        num_background = 0;
    }

    bool Histogram::isVisited() const
    {
        return visited;
    }

    float Histogram::voteForeground(uchar blue, uchar green, uchar red) const
    {
        float for_fore = eta_f * prob_fg[blue][green][red];
        float for_back = eta_b * prob_bg[blue][green][red];
        if (for_fore > 0 || for_back > 0)
        {
            return for_fore / (for_fore + for_back);
        }
        return 0.5f;
        //return -1;
    }

}