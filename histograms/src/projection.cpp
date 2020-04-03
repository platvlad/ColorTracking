#include <opencv2/imgproc.hpp>
#include "projection.h"

#define _USE_MATH_DEFINES
#include <math.h>

const float Projection::s_heaviside = 1.2f;

Projection::Projection() : width(0), height(0), frame_offset(0), color_map(cv::Mat3b(0, 0))
{
    cv::Size frame_size(width, height);
    cv::Mat3f(frame_size, cv::Vec3f(0, 0, std::numeric_limits<float>::max()));
    depth_map = cv::Mat3f(frame_size, cv::Vec3f(0, 0, std::numeric_limits<float>::max()));
    signed_distance = cv::Mat1f(height, width);
    heaviside = cv::Mat1f(height, width);
    mask = cv::Mat::zeros(frame_size, mask.type());
    roi = cv::Rect();
    nearest_labels = cv::Mat1i(height, width);
}

Projection::Projection(const cv::Mat3b &color_frame, int frame_offset) : color_map(color_frame),
                                                                         height(color_frame.rows),
                                                                         width(color_frame.cols),
                                                                         frame_offset(frame_offset)
{
    cv::Size frame_size = color_frame.size();
    depth_map = cv::Mat3f(frame_size, cv::Vec3f(0, 0, std::numeric_limits<float>::max()));
    signed_distance = cv::Mat1f(height, width);
    heaviside = cv::Mat1f(height, width);
    mask = cv::Mat::zeros(frame_size, mask.type());
    roi = cv::Rect();
    nearest_labels = cv::Mat1i(height, width);
}

Projection::Projection(const cv::Size &size) : color_map(cv::Mat3b::zeros(size)),
                                               height(size.height),
                                               width(size.width)
{
    depth_map = cv::Mat3f(size, cv::Vec3f(0, 0, std::numeric_limits<float>::max()));
    signed_distance = cv::Mat1f(size);
    heaviside = cv::Mat1f(size);
    mask = cv::Mat1b::zeros(size);
    roi = cv::Rect();
    nearest_labels = cv::Mat1i(height, width);
}


cv::Rect Projection::getExtendedROI(int offset) const
{
    if (roi.width == 0 && roi.height == 0)
    {
        return roi;
    }
    int left = std::max(0, roi.x - offset);
    int right = std::min(width, roi.x + roi.width + offset);
    int top = std::max(0, roi.y - offset);
    int bottom = std::min(height, roi.y + roi.height + offset);
    return cv::Rect(left, top, right - left, bottom - top);
}

Projection Projection::operator()(const cv::Rect &req_roi) const
{
    Projection maps_on_roi = Projection(color_map(req_roi), frame_offset);
    maps_on_roi.depth_map = depth_map(req_roi);
    maps_on_roi.mask = mask(req_roi);
    maps_on_roi.signed_distance = signed_distance(req_roi);
    maps_on_roi.heaviside = heaviside(req_roi);
    maps_on_roi.nearest_labels = nearest_labels(req_roi);
    int roi_left = std::max(0, roi.x - req_roi.x);
    int roi_right = std::min(req_roi.width, roi.x + roi.width - req_roi.x);
    int roi_up = std::max(0, roi.y - req_roi.y);
    int roi_down = std::min(req_roi.height, roi.y + roi.height - req_roi.y);
    maps_on_roi.roi = cv::Rect(roi_left, roi_up, roi_right - roi_left, roi_down - roi_up);
    return maps_on_roi;
}

bool Projection::hasEmptyProjection() const
{
    return roi.width == 0 && roi.height == 0;
}

cv::Rect Projection::getPatchSquare(int center_x, int center_y)
{
    int left = std::max(0, center_x - frame_offset);
    int up = std::max(0, center_y - frame_offset);
    int right = std::min(center_x + frame_offset + 1, width);
    int down = std::min(center_y + frame_offset + 1, height);
    return cv::Rect(left, up, right - left, down - up);
}

void Projection::trimToExtendedROI(){
    cv::Rect extendedROI = getExtendedROI(frame_offset);
    for (int i = 0; i < vertex_projections.size(); ++i)
    {
        glm::vec3& pixel = vertex_projections[i];
        pixel = glm::vec3(pixel.x - extendedROI.x, pixel.y - extendedROI.y, pixel.z);
    }
    width = extendedROI.width;
    height = extendedROI.height;
    depth_map = depth_map(extendedROI);
    color_map = color_map(extendedROI);
    mask = mask(extendedROI);
    signed_distance = signed_distance(extendedROI);
    heaviside = heaviside(extendedROI);
    roi = cv::Rect(roi.x - extendedROI.x, roi.y - extendedROI.y, roi.width, roi.height);
    nearest_labels = nearest_labels(extendedROI);
}

void Projection::invertMask()
{
    uchar* mask_ptr = mask.ptr<uchar>();
    int i = 0;
    size_t elems_to_jump = mask.step1() - mask.cols;
    for (size_t row = 0; row < signed_distance.rows; ++row)
    {
        for (size_t col = 0; col < signed_distance.cols; ++col)
        {
            mask_ptr[i] = mask_ptr[i] ? 0 : 1;
            ++i;
        }
        i += elems_to_jump;
    }
//    size_t pixel_count = mask.rows * mask.cols;
//    for (size_t i = 0; i < pixel_count; ++i)
//    {
//        mask_ptr[i] = mask_ptr[i] ? 0 : 1;
//    }
}

void Projection::computeSignedDistance()
{
    cv::Size projection_size = cv::Size(width, height);
    if (projection_size.width == 0 && projection_size.height == 0)
    {
        return;
    }
    cv::Mat1f internal_signed_distance = cv::Mat1f::zeros(projection_size);
    cv::Mat1f external_signed_distance = cv::Mat1f::zeros(projection_size);
    cv::distanceTransform(mask,
                          internal_signed_distance,
                          CV_DIST_L2,
                          CV_DIST_MASK_PRECISE);
    internal_signed_distance = -internal_signed_distance;
    invertMask();

    cv::distanceTransform(mask,
                          external_signed_distance,
                          CV_DIST_L2,
                          CV_DIST_MASK_PRECISE);

    cv::Mat1b contour = (internal_signed_distance < -1.5 | internal_signed_distance >= 0);
    cv::Mat1f dist_to_contour = cv::Mat1f::zeros(internal_signed_distance.size());
    cv::distanceTransform(contour,
                          dist_to_contour,
                          nearest_labels,
                          CV_DIST_L2,
                          CV_DIST_MASK_PRECISE,
                          cv::DIST_LABEL_PIXEL);


    invertMask();
    signed_distance = internal_signed_distance + external_signed_distance;

}

float heaviside_parametrized(float x, float s)
{
    return (0.5 -atan(x * s) / M_PI);
}

void Projection::computeHeaviside()
{
    float* signed_distance_ptr = signed_distance.ptr<float>();
    float* heaviside_ptr = heaviside.ptr<float>();
    int i = 0;
    size_t elems_to_jump = heaviside.step1() - signed_distance.cols;
    for (size_t row = 0; row < signed_distance.rows; ++row)
    {
        for (size_t col = 0; col < signed_distance.cols; ++col)
        {
            heaviside_ptr[i] = heaviside_parametrized(signed_distance_ptr[i], Projection::s_heaviside);
            ++i;
        }
        i += elems_to_jump;
    }
}

cv::Size Projection::getSize() const
{
    return cv::Size(width, height);
}
