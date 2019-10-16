#include "CircleWindow.h"

namespace histograms
{
    CircleWindow::CircleWindow(unsigned int radius) : radius(static_cast<int>(radius))
    {
        window_size = 2 * static_cast<int>(radius) + 1;
        mask = std::pair<std::vector<int>, std::vector<int> >();
        mask.first = std::vector<int>();
        mask.second = std::vector<int>();
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
                if (sqr_dist <= sqr_radius)
                {
                    mask.first.push_back(i);
                    mask.second.push_back(j);
                }
            }
        }
    }

    const Mask &CircleWindow::getMask() const
    {
        return mask;
    }
}