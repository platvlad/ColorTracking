#include "maps.h"

Maps::Maps(int width, int height) : width(width), height(height), color_map(cv::Mat3b(height, width))
{
    cv::Size frame_size(width, height);
    depth_map = cv::Mat(frame_size, depth_map.type(), cv::Scalar(std::numeric_limits<float>::max()));
    signed_distance = cv::Mat1f(height, width);
    heaviside = cv::Mat1f(height, width);
    mask = cv::Mat::zeros(frame_size, mask.type());
    roi = cv::Rect();
}

Maps::Maps(const cv::Mat3b &color_frame) : color_map(color_frame),
                                           height(color_frame.rows),
                                           width(color_frame.cols)
{
    cv::Size frame_size = color_frame.size();
    depth_map = cv::Mat(frame_size, depth_map.type(), cv::Scalar(std::numeric_limits<float>::max()));
    signed_distance = cv::Mat1f(height, width);
    heaviside = cv::Mat1f(height, width);
    mask = cv::Mat::zeros(frame_size, mask.type());
    roi = cv::Rect();
}

Maps::Maps(const cv::Size &size) : color_map(cv::Mat3b::zeros(size)),
                                   height(size.height),
                                   width(size.width)
{
    depth_map = cv::Mat(size, depth_map.type(), cv::Scalar(std::numeric_limits<float>::max()));
    signed_distance = cv::Mat1f(size);
    heaviside = cv::Mat1f(size);
    mask = cv::Mat1b::zeros(size);
    roi = cv::Rect();
}


cv::Rect2i Maps::getExtendedROI(int offset) const
{
    if (roi.empty())
    {
        return roi;
    }
    int left = std::max(0, roi.x - offset);
    int right = std::min(width, roi.x + roi.width + offset);
    int top = std::max(0, roi.y - offset);
    int bottom = std::min(height, roi.y + roi.height + offset);
    return cv::Rect2i(left, top, right - left, bottom - top);
}

Maps Maps::operator()(const cv::Rect &req_roi) const
{
    Maps maps_on_roi =  Maps(color_map(req_roi));
    maps_on_roi.depth_map = depth_map(req_roi);
    maps_on_roi.mask = mask(req_roi);
    maps_on_roi.signed_distance = signed_distance(req_roi);
    maps_on_roi.heaviside = heaviside(req_roi);
    int roi_left = std::max(0, roi.x - req_roi.x);
    int roi_right = std::min(req_roi.width, roi.x + roi.width - req_roi.x);
    int roi_up = std::max(0, roi.y - req_roi.y);
    int roi_down = std::min(req_roi.height, roi.y + roi.height - req_roi.y);
    maps_on_roi.roi = cv::Rect(roi_left, roi_up, roi_right - roi_left, roi_down - roi_up);
    return maps_on_roi;
}

bool Maps::hasEmptyProjection() const
{
    return roi.empty();
}


bool Maps::isEmpty() const
{
    return !width && !height;
}

cv::Rect Maps::getPatchSquare(int center_x, int center_y, int radius)
{
    int left = std::max(0, center_x - radius);
    int up = std::max(0, center_y - radius);
    int right = std::min(center_x + radius + 1, width);
    int down = std::min(center_y + radius + 1, height);
    return cv::Rect(left, up, right - left, down - up);
}

