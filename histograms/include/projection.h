#ifndef RENDERER_MAPS_H
#define RENDERER_MAPS_H


#include <opencv2/core/mat.hpp>

class Projection
{
    int width;
    int height;

public:
    Projection(int width, int height);

    explicit Projection(const cv::Mat3b &color_frame);

    explicit Projection(const cv::Size& size);

    cv::Mat3f depth_map;
    cv::Mat3b color_map;
    cv::Mat1b mask;
    cv::Mat1f signed_distance;
    cv::Mat1f heaviside;
    cv::Rect roi;
    cv::Mat1i nearest_labels;

    cv::Rect getExtendedROI(int offset = 0) const;

    Projection operator()(const cv::Rect &req_roi) const;

    cv::Rect getPatchSquare(int center_x, int center_y, int radius);

    bool hasEmptyProjection() const;


};


#endif //RENDERER_MAPS_H
