#include<opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>

#include "renderer.h"

const float Renderer::s_heaviside = 1.2f;

Renderer::Renderer(const glm::mat4 &camera_matrix, float z_near, float z_far, size_t width, size_t height) :
                      camera_matrix(camera_matrix),
                      z_near(z_near),
                      z_far(z_far),
                      width(width),
                      height(height)
{

}

Renderer::Renderer()
{
    camera_matrix = glm::mat4();
    z_near = 0;
    z_far = 0;
    width = 0;
    height = 0;
}


Maps Renderer::projectMesh(const histograms::Mesh& mesh, const glm::mat4& pose) const
{
    Maps maps(width, height);
    const std::vector<glm::uvec3>& faces = mesh.getFaces();
    const std::vector<glm::vec3>& vertices = mesh.getVertices();
    for (size_t i = 0; i < faces.size(); ++i)
    {
        glm::uvec3 face = faces[i];
        glm::vec3 v0 = vertices[face[0]];
        glm::vec3 v1 = vertices[face[1]];
        glm::vec3 v2 = vertices[face[2]];
        glm::vec3 p0 = projectVertex(v0, pose);
        glm::vec3 p1 = projectVertex(v1, pose);
        glm::vec3 p2 = projectVertex(v2, pose);

        if (p0 != glm::vec3() && p1 != glm::vec3() && p2 != glm::vec3())
        {
            renderTriangle(maps, i, p0, p1, p2);
        }

    }
    computeSignedDistance(maps.mask, maps.signed_distance);
    computeHeaviside(maps.signed_distance, maps.heaviside);
    return maps;
}

glm::vec3 Renderer::projectVertex(const glm::vec3& vertex, const glm::mat4& pose) const
{
    glm::vec4 v_hom = glm::vec4(vertex, 1);
    v_hom = pose * v_hom;
    glm::vec4 p_hom = camera_matrix * v_hom;
    if (p_hom[3] != 0.0f)
    {
        return glm::vec3(p_hom[0] / p_hom[3], -2 * camera_matrix[2][1] - (p_hom[1] / p_hom[3]), p_hom[3]);
    }
    return glm::vec3();
}

//const cv::Mat1f& Renderer::getDepthMap() const
//{
//    return depth_map;
//}
//
//const cv::Mat1f& Renderer::getSignedDistance() const
//{
//    return signed_distance;
//}
//
//const cv::Mat1f& Renderer::getHeaviside() const
//{
//    return heaviside;
//}
//
//const cv::Mat1b& Renderer::getMask() const
//{
//    return mask;
//}
//
//cv::Rect2i Renderer::getROI(int frame_size) const
//{
//    if (roi.empty())
//    {
//        return roi;
//    }
//    int left = std::max(0, roi.x - frame_size);
//    int right = std::min(width, roi.x + roi.width + frame_size);
//    int top = std::max(0, roi.y - frame_size);
//    int bottom = std::min(height, roi.y + roi.height + frame_size);
//    return cv::Rect2i(left, top, right - left, bottom - top);
//}

size_t Renderer::getWidth() const
{
    return width;
}

void Renderer::roundXY(glm::vec3& point)
{
    point.x = round(point.x);
    point.y = round(point.y);
}

// z -- depth
// x, y are in screen coordinates
// before: depth[:, :] = std::numeric_limits<float>::max()
void Renderer::renderTriangle(Maps& maps, int const color, glm::vec3& p0, glm::vec3& p1, glm::vec3& p2) const
{
    roundXY(p0);  // inplace
    roundXY(p1);
    roundXY(p2);
    if (p0.y == p1.y && p0.y == p2.y) return;

    if (p0.y > p1.y) std::swap(p0, p1);
    if (p0.y > p2.y) std::swap(p0, p2);
    if (p1.y > p2.y) std::swap(p1, p2);

    int const yMin = static_cast<int>(p0.y);
    bool const secondHalfPrecalc = p1.y == p0.y;
    float const totalHeight = p2.y - p0.y;
    for (int i = 0; i < totalHeight; ++i) {
        int const y = yMin + i;
        bool const secondHalf = secondHalfPrecalc || i > p1.y - p0.y;
        float const segmentHeight = secondHalf ? p2.y - p1.y : p1.y - p0.y;

        float const alpha = i / totalHeight;
        float const beta = (i - (secondHalf ? p1.y - p0.y : 0))
                           / segmentHeight;

        glm::vec3 a(p0 + (p2 - p0) * alpha);
        glm::vec3 b(secondHalf ? (p1 + (p2 - p1) * beta)
                               : (p0 + (p1 - p0) * beta));
        if (a.x > b.x) std::swap(a, b);

        int const jMin = static_cast<int>(ceil(a.x));
        int const jMax = static_cast<int>(floor(b.x));
        bool const thinLine = jMin == jMax;
        for (int j = jMin; j <= jMax; ++j) {
            int const x = j;
            float const phi = thinLine ? 1.0f : (j - a.x) / (b.x - a.x);
            glm::vec3 const p(a * 1.0f + (b - a) * phi);

            if (x < 0 || y < 0
                || static_cast<size_t>(x) >= width
                || static_cast<size_t>(y) >= height) {
                continue;
            }
            if (maps.depth_map.at<float>(y, x) > p.z) {  // cv::Mat1f
                maps.depth_map.at<float>(y, x) = p.z;
                maps.color_map.at<float>(y, x) = color;  // cv::Mat1i
                maps.mask.at<uchar>(y, x) = 255;
                updateROI(maps.roi, x, y);
            }
        }
    }
}

void Renderer::updateROI(cv::Rect& roi, int x, int y)
{
    if (roi.empty())
    {
        roi.x = x;
        roi.y = y;
        roi.width = 1;
        roi.height = 1;
    }
    int x_diff = roi.x - x;
    int y_diff = roi.y - y;
    if (x_diff > 0)
    {
        roi.x -= x_diff;
        roi.width += x_diff;
    }
    if (y_diff > 0)
    {
        roi.y -= y_diff;
        roi.height += y_diff;
    }
    if (roi.width <= x - roi.x)
    {
        roi.width = x - roi.x + 1;
    }
    if (roi.height <= y - roi.y)
    {
        roi.height = y - roi.y + 1;
    }
}

//cv::Rect2i Renderer::computeROI()
//{
//    size_t left = width - 1;
//    size_t right = 0;
//    size_t top = height - 1;
//    size_t bottom = 0;
//
//    const float* depth_ptr = depth_map.ptr<float>();
//    size_t row = 0;
//    size_t column = 0;
//    size_t numPixels = width * height;
//    for (size_t i = 0; i < numPixels; ++i)
//    {
//        if (depth_ptr[i] != std::numeric_limits<float>::max())
//        {
//            if (column < left) left = column;
//            if (column > right) right = column;
//            if (row < top) top = row;
//            if (row > bottom) bottom = row;
//        }
//    }
//    if (right < left)
//    {
//        return cv::Rect2i();
//    }
//    left = (left > 8) ? left - 8 : 0;
//    right = (right + 8 < width) ? right + 8 : width - 1;
//    top = (top > 8) ? top - 8 : 0;
//    bottom = (bottom + 8 < height) ? bottom + 8 : height - 1;
//
//    return cv::Rect2i(left, top, right - left + 1, bottom - top + 1);
//}

void Renderer::invertMask(cv::Mat1b& mask)
{
    uchar* mask_ptr = mask.ptr<uchar>();
    size_t pixel_count = mask.rows * mask.cols;
    for (size_t i = 0; i < pixel_count; ++i)
    {
        mask_ptr[i] = mask_ptr[i] ? 0 : 1;
    }
}

void Renderer::computeSignedDistance(cv::Mat1b& mask, cv::Mat1f& signed_distance)
{
    cv::Mat1f internal_signed_distance = cv::Mat::zeros(signed_distance.size(), signed_distance.type());
    cv::Mat1f external_signed_distance = cv::Mat::zeros(signed_distance.size(), signed_distance.type());
    cv::distanceTransform(mask,
                          internal_signed_distance,
                          cv::DIST_L2,
                          cv::DIST_MASK_PRECISE,
                           CV_32F);
    internal_signed_distance = -internal_signed_distance;
    invertMask(mask);
    cv::distanceTransform(mask,
                          external_signed_distance,
                          cv::DIST_L2,
                          cv::DIST_MASK_PRECISE,
                          CV_32F);
    invertMask(mask);
    signed_distance = internal_signed_distance + external_signed_distance;

}

float heaviside_parametrized(float x, float s)
{
    return (0.5 -atan(x * s) / M_PI);
}

void Renderer::computeHeaviside(cv::Mat1f& signed_distance, cv::Mat1f& heaviside)
{
    size_t num_pixels = signed_distance.rows * signed_distance.cols;
    float* signed_distance_ptr = signed_distance.ptr<float>();
    float* heaviside_ptr = heaviside.ptr<float>();
    for (size_t i = 0; i < num_pixels; ++i)
    {
        heaviside_ptr[i] = heaviside_parametrized(signed_distance_ptr[i], Renderer::s_heaviside);
    }

}

