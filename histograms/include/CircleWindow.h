#ifndef HISTOGRAMS_CIRCLEWINDOW_H
#define HISTOGRAMS_CIRCLEWINDOW_H

#include <vector>
#include <opencv2/core/mat.hpp>

namespace histograms
{
    class CircleWindow
    {
        cv::Mat1b mask;

    private:
        int radius;
        int window_size;
        std::vector<int> edge_curve;
        unsigned int area;

        std::vector<int> rel_rows;
        std::vector<int> rel_cols;

        void fillCircleWindow();

    public:
        explicit CircleWindow(unsigned int radius = 0);

        unsigned int getWindowSize() const;

        unsigned int getArea() const;

        const std::vector<int> &getEdgeCurve() const;

        const std::vector<int> &getRelRows() const;

        const std::vector<int> &getRelCols() const;

        const cv::Mat1b &getMask() const;
    };
}
#endif //HISTOGRAMS_CIRCLEWINDOW_H
