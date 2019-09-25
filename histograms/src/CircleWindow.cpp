#include "CircleWindow.h"

namespace histograms
{
    CircleWindow::CircleWindow(unsigned int radius) : radius(radius),
                                                      area(0)
    {
        window_size = 2 * radius + 1;
        mask = cv::Mat1b(cv::Size(window_size, window_size));
        edge_curve = std::vector<int>(window_size);
        rel_rows = std::vector<int>();
        rel_cols = std::vector<int>();
        fillCircleWindow();
    }

    void CircleWindow::fillCircleWindow()
    {
        unsigned int sqr_radius = radius * radius;
        for (int i = 0; i < window_size; ++i)
        {
            for (int j = 0; j < window_size; ++j)
            {
                unsigned int sqr_dist = (radius - i) * (radius - i) +
                                        (radius - j) * (radius - j);
                mask(i, j) = sqr_dist <= sqr_radius ? 1 : 0;
                if (mask(i, j))
                {
                    ++area;
                    rel_rows.push_back(radius - i);
                    rel_cols.push_back(radius - j);
                }
                if (j > 0)
                {
                    if (!mask(i, j - 1) && mask(i, j))
                    {
                        edge_curve[i] = radius - j;
                    }
                }
                else
                {
                    if (mask(i, j))
                    {
                        edge_curve[i] = radius;
                    }
                }
            }
        }
    }

    unsigned int CircleWindow::getWindowSize() const
    {
        return window_size;
    }

    const std::vector<int> &CircleWindow::getEdgeCurve() const
    {
        return edge_curve;
    }

    unsigned int CircleWindow::getArea() const
    {
        return area;
    }

    const std::vector<int> &CircleWindow::getRelRows() const
    {
        return rel_rows;
    }

    const std::vector<int> &CircleWindow::getRelCols() const
    {
        return rel_cols;
    }

    const cv::Mat1b &CircleWindow::getMask() const
    {
        return mask;
    }
}