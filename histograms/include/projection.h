#ifndef RENDERER_MAPS_H
#define RENDERER_MAPS_H


#include <opencv2/core.hpp>
#include <glm/vec3.hpp>

class Projection
{


    int width;
    int height;
    int frame_offset;

    void invertMask();

public:
    static const float s_heaviside;

    Projection();

    explicit Projection(const cv::Mat3b &color_frame, int frame_offset);

    explicit Projection(const cv::Size& size);

    cv::Mat3f depth_map;
    cv::Mat3b color_map;
    cv::Mat1b mask;
    cv::Mat1f signed_distance;
    cv::Mat1f heaviside;
    cv::Rect roi;
    cv::Mat1i nearest_labels;
    std::vector<glm::vec3> vertex_projections;

    cv::Rect getExtendedROI(int offset = 0) const;

    Projection operator()(const cv::Rect &req_roi) const;

    cv::Rect getPatchSquare(int center_x, int center_y);

    bool hasEmptyProjection() const;

    void trimToExtendedROI();

    void computeSignedDistance();

    void computeHeaviside();

    cv::Size getSize() const;
};


#endif //RENDERER_MAPS_H
