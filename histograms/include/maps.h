#ifndef RENDERER_MAPS_H
#define RENDERER_MAPS_H


#include <opencv2/core/mat.hpp>

class Maps
{
    const int width;
    const int height;

public:
    Maps(int width, int height);

    cv::Mat1f depth_map;
    cv::Mat1i color_map;
    cv::Mat1b mask;
    cv::Mat1f signed_distance;
    cv::Mat1f heaviside;
    cv::Rect roi;

    cv::Rect getExtendedROI(int offset = 0) const;

};


#endif //RENDERER_MAPS_H
