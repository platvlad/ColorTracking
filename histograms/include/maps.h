#ifndef RENDERER_MAPS_H
#define RENDERER_MAPS_H


#include <opencv2/core/mat.hpp>

class Maps
{
    const int width;
    const int height;

public:
    Maps(int width, int height);

    explicit Maps(const cv::Mat3b &color_frame);

    explicit Maps(const cv::Size& size);

    cv::Mat1f depth_map;
    const cv::Mat3b color_map;
    cv::Mat1b mask;
    cv::Mat1f signed_distance;
    cv::Mat1f heaviside;
    cv::Rect roi;

    cv::Rect getExtendedROI(int offset = 0) const;

    Maps operator()(const cv::Rect &req_roi) const;

    cv::Rect getPatchSquare(int center_x, int center_y, int radius);

    bool hasEmptyProjection() const;

    bool isEmpty() const;

};


#endif //RENDERER_MAPS_H
