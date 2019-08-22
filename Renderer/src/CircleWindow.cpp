#include "CircleWindow.h"

namespace histograms
{
    CircleWindow::CircleWindow(unsigned int radius) : radius(radius),
                                                      area(0)
    {
        window_size = 2 * radius + 1;
        mask = std::vector<std::vector<bool> >(window_size);
        for (unsigned int i = 0; i < window_size; ++i)
        {
            mask[i] = std::vector<bool>(window_size);
        }
        edge_curve = std::vector<int>(window_size);
        fillCircleWindow();
    }

    void CircleWindow::fillCircleWindow()
    {
        unsigned int sqr_radius = radius * radius;
        for (unsigned int i = 0; i < window_size; ++i)
        {
            for (unsigned int j = 0; j < window_size; ++j)
            {
                unsigned int sqr_dist = (radius - i) * (radius - i) +
                                        (radius - j) * (radius - j);
                mask[i][j] = (sqr_dist <= sqr_radius);
                if (mask[i][j])
                {
                    ++area;
                }
                if (j > 0)
                {
                    if (!mask[i][j - 1] && mask[i][j])
                    {
                        edge_curve[i] = radius - j;
                    }
                }
                else
                {
                    if (mask[i][j])
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
}