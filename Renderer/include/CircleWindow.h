#ifndef HISTOGRAMS_CIRCLEWINDOW_H
#define HISTOGRAMS_CIRCLEWINDOW_H

#include <vector>

namespace histograms
{
    class CircleWindow
    {
        std::vector<std::vector<bool> > mask;
        unsigned int radius;
        unsigned int window_size;
        std::vector<int> edge_curve;
        unsigned int area;

        void fillCircleWindow();

    public:
        explicit CircleWindow(unsigned int radius = 0);

        unsigned int getWindowSize() const;

        unsigned int getArea() const;

        const std::vector<int> &getEdgeCurve() const;
    };
}
#endif //HISTOGRAMS_CIRCLEWINDOW_H
