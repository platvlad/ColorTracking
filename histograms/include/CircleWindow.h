#ifndef HISTOGRAMS_CIRCLEWINDOW_H
#define HISTOGRAMS_CIRCLEWINDOW_H

#include <vector>
//#include <opencv2/core/mat.hpp>

typedef std::pair<std::vector<int>, std::vector<int> > Mask;

namespace histograms
{
    class CircleWindow
    {
        Mask mask;
        int radius;
        int window_size;


        void fillCircleWindow();

    public:
        explicit CircleWindow(unsigned int radius = 0);

        const Mask &getMask() const;
    };
}
#endif //HISTOGRAMS_CIRCLEWINDOW_H
