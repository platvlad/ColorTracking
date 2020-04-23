#include <projection.h>
#include "histogram.h"
#include <iostream>

namespace histograms
{
    float Histogram::alpha_f = 0.1;
    float Histogram::alpha_b = 0.2;
    std::map<int, const Mask> Histogram::window_masks;

    Histogram::Histogram() : prob_fg(),
                             prob_bg(),
                             visited(false)
    {
        if (window_masks.find(radius) == window_masks.end())
        {
            window_masks.insert(std::make_pair(radius, CircleWindow(radius).getMask()));
        }
    }

    void Histogram::update(const Projection &local_square, int center_x, int center_y) {
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
                    prob_bg[i][j][k] *= (1 - alpha_b);
                    prob_fg[i][j][k] *= (1 - alpha_f);
                }
            }
        }
        bool strict_classification = false;
        const Mask& window_mask = window_masks.at(radius);

        for (size_t i = 0; i < window_mask.first.size(); ++i) {
            int row = window_mask.first[i];
            int col = window_mask.second[i];
            int row_on_local_square = row - row_offset;
            int col_on_local_square = col - col_offset;
            int binSize = ceil(256.0 / COLORS_PER_CHANNEL);
            if (row_on_local_square >= 0 && row_on_local_square < local_square.mask.rows &&
                col_on_local_square >= 0 && col_on_local_square < local_square.mask.cols)
            {
                uchar blue = local_square.color_map(row_on_local_square, col_on_local_square)[0] / binSize;
                uchar green = local_square.color_map(row_on_local_square, col_on_local_square)[1] / binSize;
                uchar red = local_square.color_map(row_on_local_square, col_on_local_square)[2] / binSize;
                float acc_blue = local_square.color_map(row_on_local_square, col_on_local_square)[0] / static_cast<float>(binSize);
                float acc_green = local_square.color_map(row_on_local_square, col_on_local_square)[1] / static_cast<float>(binSize);
                float acc_red = local_square.color_map(row_on_local_square, col_on_local_square)[2] / static_cast<float>(binSize);
                float frac_blue = acc_blue - blue;
                float frac_green = acc_green - green;
                float frac_red = acc_red - red;
                uchar alt_blue = blue;
                uchar alt_green = green;
                uchar alt_red = red;
                float blue_weight = 0;
                float green_weight = 0;
                float red_weight = 0;
                if (frac_blue > 0.5f && blue < COLORS_PER_CHANNEL - 1)
                {
                    alt_blue = blue + 1;
                    blue_weight = (frac_blue - 0.5) / (1.5 - frac_blue);
                }
                else if (frac_blue <= 0.5f && blue > 0)
                {
                    alt_blue = blue - 1;
                    blue_weight = (0.5 - frac_blue) / (0.5 + frac_blue);
                }
                if (frac_green > 0.5f && green < COLORS_PER_CHANNEL - 1)
                {
                    alt_green = green + 1;
                    green_weight = (frac_green - 0.5) / (1.5 - frac_green);
                }
                else if (frac_green <= 0.5f && green > 0)
                {
                    alt_green = green - 1;
                    green_weight = (0.5 - frac_green) / (0.5 + frac_green);
                }
                if (frac_red > 0.5f && red < COLORS_PER_CHANNEL - 1)
                {
                    alt_red = red + 1;
                    red_weight = (frac_red - 0.5) / (1.5 - frac_red);
                }
                else if (frac_red <= 0.5f && red > 0)
                {
                    alt_red = red - 1;
                    red_weight = (0.5 - frac_red) / (0.5 + frac_red);
                }
                float weight = 1 + blue_weight + green_weight + red_weight;
                
                
                if (strict_classification) {
                    if (local_square.heaviside(row_on_local_square, col_on_local_square) > 0.5)
                    {
                        prob_fg[blue][green][red] += visited ?
                                                     alpha_f / num_foreground :
                                                     1 / num_foreground;
                    }
                    else
                    {
                        prob_bg[blue][green][red] += visited ?
                                                     alpha_b / num_background :
                                                     1 / num_background;
                    }
                }
                else
                {
                    prob_fg[blue][green][red] += visited ?
                            local_square.heaviside(row_on_local_square, col_on_local_square) * alpha_f /
                            (num_foreground * weight) :
                            local_square.heaviside(row_on_local_square, col_on_local_square) / (num_foreground * weight);
                    prob_bg[blue][green][red] += visited ?
                            (1 - local_square.heaviside(row_on_local_square, col_on_local_square)) * alpha_b /
                            (num_background * weight) :
                            (1 - local_square.heaviside(row_on_local_square, col_on_local_square)) / (num_background * weight);

                    prob_fg[alt_blue][green][red] += visited ?
                        blue_weight * local_square.heaviside(row_on_local_square, col_on_local_square) * alpha_f /
                        (num_foreground * weight) :
                        blue_weight * local_square.heaviside(row_on_local_square, col_on_local_square) / (num_foreground * weight);
                    prob_bg[alt_blue][green][red] += visited ?
                        blue_weight * (1 - local_square.heaviside(row_on_local_square, col_on_local_square)) * alpha_b /
                        (num_background * weight) :
                        blue_weight * (1 - local_square.heaviside(row_on_local_square, col_on_local_square)) / (num_background * weight);

                    prob_fg[blue][alt_green][red] += visited ?
                        green_weight * local_square.heaviside(row_on_local_square, col_on_local_square) * alpha_f /
                        (num_foreground * weight) :
                        green_weight * local_square.heaviside(row_on_local_square, col_on_local_square) / (num_foreground * weight);
                    prob_bg[blue][alt_green][red] += visited ?
                        green_weight * (1 - local_square.heaviside(row_on_local_square, col_on_local_square)) * alpha_b /
                        (num_background * weight) :
                        green_weight * (1 - local_square.heaviside(row_on_local_square, col_on_local_square)) / (num_background * weight);

                    prob_fg[blue][green][alt_red] += visited ?
                        red_weight * local_square.heaviside(row_on_local_square, col_on_local_square) * alpha_f /
                        (num_foreground * weight) :
                        red_weight * local_square.heaviside(row_on_local_square, col_on_local_square) / (num_foreground * weight);
                    prob_bg[blue][green][alt_red] += visited ?
                        red_weight * (1 - local_square.heaviside(row_on_local_square, col_on_local_square)) * alpha_b /
                        (num_background * weight) :
                        red_weight * (1 - local_square.heaviside(row_on_local_square, col_on_local_square)) / (num_background * weight);
                }
            }
        }

//        for (int row = 0; row < local_square.mask.rows; ++row) {
//            for (int col = 0; col < local_square.mask.cols; ++col) {
//                if (window_mask(row + row_offset, col + col_offset)) {
//                    int binSize = ceil(256.0 / COLORS_PER_CHANNEL);
//                    uchar blue = local_square.color_map(row, col)[0] / binSize;
//                    uchar green = local_square.color_map(row, col)[1] / binSize;
//                    uchar red = local_square.color_map(row, col)[2] / binSize;
//                    if (strict_classification) {
//                        if (local_square.heaviside(row, col) > 0.5)
//                        {
//                            prob_fg[blue][green][red] += visited ?
//                                                         Histogram::alpha_f / num_foreground :
//                                                         1 / num_foreground;
//                        }
//                        else
//                            {
//                            prob_bg[blue][green][red] += visited ?
//                                                         Histogram::alpha_b / num_background :
//                                                         1 / num_background;
//                        }
//
//                    }
//                    else
//                    {
//                        prob_fg[blue][green][red] += visited ?
//                                                     local_square.heaviside(row, col) * Histogram::alpha_f /
//                                                     num_foreground :
//                                                     local_square.heaviside(row, col) / num_foreground;
//                        prob_bg[blue][green][red] += visited ?
//                                                     (1 - local_square.heaviside(row, col)) * Histogram::alpha_b /
//                                                     num_background :
//                                                     (1 - local_square.heaviside(row, col)) / num_background;
//                    }
//                }
//            }
//        }
        visited = true;

    }

    bool Histogram::isVisited() const
    {
        return visited;
    }

    std::pair<float, float> Histogram::get_eta_f_eta_b(const Projection &local_square, int center_x, int center_y, int circle_radius)
    {

        int col_offset = static_cast<int>(circle_radius) - center_x;
        int row_offset = static_cast<int>(circle_radius) - center_y;
        float eta_f = 0;
        float eta_b = 0;
        std::map<int, const Mask>::const_iterator window_mask_iter = window_masks.find(circle_radius);
        if (window_mask_iter == window_masks.end())
        {
            window_mask_iter = window_masks.insert(std::make_pair(circle_radius, CircleWindow(circle_radius).getMask())).first;
        }
        const Mask& window_mask = window_mask_iter->second;

        for (size_t i = 0; i < window_mask.first.size(); ++i)
        {
            int row = window_mask.first[i];
            int col = window_mask.second[i];
            int row_on_local_square = row - row_offset;
            int col_on_local_square = col - col_offset;
            if (row_on_local_square >= 0 && row_on_local_square < local_square.mask.rows &&
                col_on_local_square >= 0 && col_on_local_square < local_square.mask.cols)
            {
                eta_f += local_square.heaviside(row_on_local_square, col_on_local_square);
                eta_b += 1 - local_square.heaviside(row_on_local_square, col_on_local_square);
            }
        }

//        for (int row = 0; row < local_square.mask.rows; ++row) {
//            for (int col = 0; col < local_square.mask.cols; ++col) {
//                if (window_mask(row + row_offset, col + col_offset)) {
//                    eta_f += local_square.heaviside(row, col);
//                    eta_b += 1 - local_square.heaviside(row, col);
//                }
//            }
//        }
        return std::pair<float, float>(eta_f, eta_b);
    }

    float Histogram::voteColor(uchar blue, uchar green, uchar red, float eta_f, float eta_b) const
    {
        float for_fore, for_back;
        if (eta_f <= 5 * eta_b && eta_b <= 5 * eta_f)
        {
            for_fore = eta_f * prob_fg[blue][green][red];
            for_back = eta_b * prob_bg[blue][green][red];
            
        }
        else
        {
            for_fore = prob_fg[blue][green][red];
            for_back = prob_bg[blue][green][red];
        }
        if (for_fore > 0 || for_back > 0)
        {
            return for_fore / (for_fore + for_back);
        }
        return 0.5f;
        //return -1;
    }

    void
    Histogram::votePatch(const Projection &local_square,
                         int center_x,
                         int center_y,
                         cv::Mat1f &votes,
                         cv::Mat1i &numVoters,
                         int circle_radius) const
    {
        std::pair<float, float> eta_f_eta_b = get_eta_f_eta_b(local_square, center_x, center_y, circle_radius);
        const cv::Mat3b& color_map = local_square.color_map;
        float eta_f = eta_f_eta_b.first;
        float eta_b = eta_f_eta_b.second;
        int col_offset = static_cast<int>(circle_radius) - center_x;
        int row_offset = static_cast<int>(circle_radius) - center_y;
        int bin_size = ceil(256.0 / COLORS_PER_CHANNEL);

        std::map<int, const Mask>::const_iterator window_mask_iter = window_masks.find(circle_radius);
        if (window_mask_iter == window_masks.end())
        {
            window_mask_iter = window_masks.insert(std::make_pair(circle_radius, CircleWindow(circle_radius).getMask())).first;
        }
        const Mask& window_mask = window_mask_iter->second;

        for (size_t i = 0; i < window_mask.first.size(); ++i)
        {
            int row = window_mask.first[i];
            int col = window_mask.second[i];
            int row_on_local_square = row - row_offset;
            int col_on_local_square = col - col_offset;
            if (row_on_local_square >= 0 && row_on_local_square < local_square.mask.rows &&
                col_on_local_square >= 0 && col_on_local_square < local_square.mask.cols)
            {
                uchar blue = color_map(row_on_local_square, col_on_local_square)[0] / bin_size;
                uchar green = color_map(row_on_local_square, col_on_local_square)[1] / bin_size;
                uchar red = color_map(row_on_local_square, col_on_local_square)[2] / bin_size;

                float acc_blue = local_square.color_map(row_on_local_square, col_on_local_square)[0] / static_cast<float>(bin_size);
                float acc_green = local_square.color_map(row_on_local_square, col_on_local_square)[1] / static_cast<float>(bin_size);
                float acc_red = local_square.color_map(row_on_local_square, col_on_local_square)[2] / static_cast<float>(bin_size);
                float frac_blue = acc_blue - blue;
                float frac_green = acc_green - green;
                float frac_red = acc_red - red;
                uchar alt_blue = blue;
                uchar alt_green = green;
                uchar alt_red = red;
                float blue_weight = 0;
                float green_weight = 0;
                float red_weight = 0;
                if (frac_blue > 0.5f && blue < COLORS_PER_CHANNEL - 1)
                {
                    alt_blue = blue + 1;
                    blue_weight = (frac_blue - 0.5) / (1.5 - frac_blue);
                }
                else if (frac_blue <= 0.5f && blue > 0)
                {
                    alt_blue = blue - 1;
                    blue_weight = (0.5 - frac_blue) / (0.5 + frac_blue);
                }
                if (frac_green > 0.5f && green < COLORS_PER_CHANNEL - 1)
                {
                    alt_green = green + 1;
                    green_weight = (frac_green - 0.5) / (1.5 - frac_green);
                }
                else if (frac_green <= 0.5f && green > 0)
                {
                    alt_green = green - 1;
                    green_weight = (0.5 - frac_green) / (0.5 + frac_green);
                }
                if (frac_red > 0.5f && red < COLORS_PER_CHANNEL - 1)
                {
                    alt_red = red + 1;
                    red_weight = (frac_red - 0.5) / (1.5 - frac_red);
                }
                else if (frac_red <= 0.5f && red > 0)
                {
                    alt_red = red - 1;
                    red_weight = (0.5 - frac_red) / (0.5 + frac_red);
                }
                float weight = 1 + blue_weight + green_weight + red_weight;
                weight = 1;

                votes(row_on_local_square, col_on_local_square) += voteColor(blue, green, red, eta_f, eta_b) / weight;
                //votes(row_on_local_square, col_on_local_square) += blue_weight * voteColor(alt_blue, green, red, eta_f, eta_b) / weight;
                //votes(row_on_local_square, col_on_local_square) += green_weight * voteColor(blue, alt_green, red, eta_f, eta_b) / weight;
                //votes(row_on_local_square, col_on_local_square) += red_weight * voteColor(blue, green, alt_red, eta_f, eta_b) / weight;

                ++numVoters(row_on_local_square, col_on_local_square);
            }
        }

//        for (int row = 0; row < local_square.mask.rows; ++row) {
//            for (int col = 0; col < local_square.mask.cols; ++col) {
//                if (window_mask(row + row_offset, col + col_offset)) {
//                    uchar blue = color_map(row, col)[0] / bin_size;
//                    uchar green = color_map(row, col)[1] / bin_size;
//                    uchar red = color_map(row, col)[2] / bin_size;
//                    votes(row, col) += voteColor(blue, green, red, eta_f, eta_b);
//                    ++numVoters(row, col);
//                }
//            }
//        }
    }


}