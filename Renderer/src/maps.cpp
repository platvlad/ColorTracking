#include "maps.h"

Maps::Maps(int width, int height) : width(width), height(height)
{
    cv::Size frame_size(width, height);
    depth_map = cv::Mat(frame_size, depth_map.type(), cv::Scalar(std::numeric_limits<float>::max()));
    color_map = cv::Mat::zeros(frame_size, color_map.type());
    signed_distance = cv::Mat1f(height, width);
    heaviside = cv::Mat1f(height, width);
    mask = cv::Mat::zeros(frame_size, mask.type());
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