#include <maps.h>
#include "histogram.h"

namespace histograms
{
    float Histogram::alpha_f = 0.1;
    float Histogram::alpha_b = 0.2;
    bool Histogram::window_mask_initialized = false;
    cv::Mat1b Histogram::window_mask;

    Histogram::Histogram() : prob_fg(),
                             prob_bg(),
                             visited(false)
    {
        if (!window_mask_initialized)
        {
            Histogram::window_mask = CircleWindow(radius).getMask();
            window_mask_initialized = true;
        }
    }

    void Histogram::update(const Maps &local_square, int center_x, int center_y) {
        std::pair<float, float> eta_f_eta_b = get_eta_f_eta_b(local_square, center_x, center_y);
        int col_offset = static_cast<int>(radius) - center_x;
        int row_offset = static_cast<int>(radius) - center_y;
        float num_foreground = eta_f_eta_b.first;
        float num_background = eta_f_eta_b.second;

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

        for (int row = 0; row < local_square.mask.rows; ++row) {
            for (int col = 0; col < local_square.mask.cols; ++col) {
                if (window_mask(row + row_offset, col + col_offset)) {
                    int binSize = ceil(256.0 / COLORS_PER_CHANNEL);
                    uchar blue = local_square.color_map(row, col)[0] / binSize;
                    uchar green = local_square.color_map(row, col)[1] / binSize;
                    uchar red = local_square.color_map(row, col)[2] / binSize;
                    prob_fg[blue][green][red] += visited ?
                            local_square.heaviside(row, col) * Histogram::alpha_f / num_foreground :
                            local_square.heaviside(row, col) / num_foreground;
                    prob_bg[blue][green][red] += visited ?
                            (1 - local_square.heaviside(row, col)) * Histogram::alpha_b / num_background :
                            (1 - local_square.heaviside(row, col)) / num_background;
                }
            }
        }
        visited = true;

    }

    bool Histogram::isVisited() const
    {
        return visited;
    }

    std::pair<float, float> Histogram::get_eta_f_eta_b(const Maps &local_square, int center_x, int center_y)
    {
        int col_offset = static_cast<int>(radius) - center_x;
        int row_offset = static_cast<int>(radius) - center_y;
        float eta_f = 0;
        float eta_b = 0;
        for (int row = 0; row < local_square.mask.rows; ++row) {
            for (int col = 0; col < local_square.mask.cols; ++col) {
                if (window_mask(row + row_offset, col + col_offset)) {
                    eta_f += local_square.heaviside(row, col);
                    eta_b += 1 - local_square.heaviside(row, col);
                }
            }
        }
        return std::pair<float, float>(eta_f, eta_b);
    }

    float Histogram::voteColor(uchar blue, uchar green, uchar red, float eta_f, float eta_b) const
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

    void
    Histogram::votePatch(const Maps &local_square, int center_x, int center_y, cv::Mat1f &votes, cv::Mat1i &numVoters) const
    {
        std::pair<float, float> eta_f_eta_b = get_eta_f_eta_b(local_square, center_x, center_y);
        const cv::Mat3b& color_map = local_square.color_map;
        float eta_f = eta_f_eta_b.first;
        float eta_b = eta_f_eta_b.second;
        int col_offset = static_cast<int>(radius) - center_x;
        int row_offset = static_cast<int>(radius) - center_y;
        int bin_size = ceil(256.0 / COLORS_PER_CHANNEL);
        for (int row = 0; row < local_square.mask.rows; ++row) {
            for (int col = 0; col < local_square.mask.cols; ++col) {
                if (window_mask(row + row_offset, col + col_offset)) {
                    uchar blue = color_map(row, col)[0] / bin_size;
                    uchar green = color_map(row, col)[1] / bin_size;
                    uchar red = color_map(row, col)[2] / bin_size;
                    votes(row, col) += voteColor(blue, green, red, eta_f, eta_b);
                    ++numVoters(row, col);
                }
            }
        }
    }


}