#include "histogram2.h"

namespace histograms
{
    float Histogram2::alpha_f = 0.1;
    float Histogram2::alpha_b = 0.2;
 
    float Histogram2::get_eta_f(const std::vector<std::pair<cv::Vec3b, float>>& color_heaviside)
    {
        float eta_f = 0;
        for (int i = 0; i < color_heaviside.size(); ++i)
        {
            eta_f += color_heaviside[i].second;
        }
        return eta_f;
    }

    Histogram2::Histogram2() : prob_fg(),
                               prob_bg(),
                               eta_f(0),
                               eta_b(0),
                               visited(false)
    {

    }

    void Histogram2::scaleHistogram()
    {
        for (int i = 0; i < COLORS_PER_CHANNEL; ++i)
        {
            for (int j = 0; j < COLORS_PER_CHANNEL; ++j)
            {
                for (int k = 0; k < COLORS_PER_CHANNEL; ++k)
                {
                    prob_bg[i][j][k] *= (1 - alpha_b);
                    prob_fg[i][j][k] *= (1 - alpha_f);
                }
            }
        }
        skill *= (1 - alpha_f);
    }

    void Histogram2::update(const std::vector<std::pair<cv::Vec3b, float> > &color_heaviside)
    {
        float eta_f = get_eta_f(color_heaviside);
        size_t num_pts = color_heaviside.size();
        float eta_b = num_pts - eta_f;

        scaleHistogram();

        for (int i = 0; i < num_pts; ++i)
        {
            const cv::Vec3b& color = color_heaviside[i].first;
            int blue = color[0] / bin_size;
            int green = color[1] / bin_size;
            int red = color[2] / bin_size;
            float heaviside_value = color_heaviside[i].second;
            prob_fg[blue][green][red] += visited ?
                heaviside_value * alpha_f / eta_f :
                heaviside_value / eta_f;
            prob_bg[blue][green][red] += visited ?
                (1 - heaviside_value) * alpha_b / eta_b :
                (1 - heaviside_value) / eta_b;
        }
        skill += num_pts;
        visited = true;
    }

    float Histogram2::voteColor(const cv::Vec3b & color, float eta_f, float eta_b) const
    {
        int blue = color[0] / bin_size;
        int green = color[1] / bin_size;
        int red = color[2] / bin_size;
        float for_fore, for_back;
        if (eta_b * 5 >= eta_f && eta_f * 5 >= eta_b)
        {
            for_fore = eta_f * prob_fg[blue][green][red];
            for_back = eta_b * prob_bg[blue][green][red];
        }
        else
        {
            for_fore = prob_fg[blue][green][red];
            for_back = prob_bg[blue][green][red];
        }

        if (for_fore == 0 && for_back == 0)
            return 0.5f;
        return for_fore / (for_fore + for_back);
    }

    float Histogram2::getSkill() const
    {
        return skill;
    }

}