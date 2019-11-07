#include<opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <iostream>

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

Renderer::Renderer(const Renderer &other, const cv::Size &size)
{
    float width_scale = static_cast<float>(size.width) / static_cast<float>(other.width);
    float height_scale = static_cast<float>(size.height) / static_cast<float>(other.height);
    const glm::mat4& s_cam = other.camera_matrix;
    camera_matrix = glm::mat4(s_cam[0][0] * width_scale, s_cam[0][1] * height_scale, s_cam[0][2], s_cam[0][3],
                              s_cam[1][0] * width_scale, s_cam[1][1] * height_scale, s_cam[1][2], s_cam[1][3],
                              s_cam[2][0] * width_scale, s_cam[2][1] * height_scale, s_cam[2][2], s_cam[2][3],
                              s_cam[3][0], s_cam[3][1], s_cam[3][2], s_cam[3][3]);
    z_near = other.z_near;
    z_far = other.z_far;
    width = size.width;
    height = size.height;
}


void Renderer::projectMesh(const histograms::Mesh& mesh, const glm::mat4& pose, Maps &maps) const
{

    //Maps maps(width, height);
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
            renderTriangle(maps, p0, p1, p2);
        }

    }
    computeSignedDistance(maps.mask, maps.signed_distance);
    computeHeaviside(maps.signed_distance, maps.heaviside);
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

size_t Renderer::getWidth() const
{
    return width;
}

void Renderer::roundXY(glm::vec3& point)
{
    point.x = floor(point.x);
    point.y = floor(point.y);
}

// z -- depth
// x, y are in screen coordinates
// before: depth[:, :] = std::numeric_limits<float>::max()
void Renderer::renderTriangle(Maps& maps, glm::vec3& p0, glm::vec3& p1, glm::vec3& p2) const
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
                maps.mask.at<uchar>(y, x) = 255;
                updateROI(maps.roi, x, y);
            }
        }
    }
}

void Renderer::renderMesh(const histograms::Mesh &mesh, cv::Mat3b& frame, const glm::mat4 &pose) const
{
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
            renderTriangleWireframe(frame, p0, p1, p2);
        }

    }
}

void Renderer::renderTriangleWireframe(cv::Mat3b& color_map, glm::vec3 &p0, glm::vec3 &p1, glm::vec3 &p2)
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

        if (jMin >= 0 && y >= 0
            && static_cast<size_t>(jMin) < color_map.cols
            && static_cast<size_t>(y) < color_map.rows)
        {
            color_map.at<cv::Vec3b>(y, jMin) = cv::Vec3b(0, 128, 0);  // cv::Mat1i
        }

        if (jMax >= 0 && y >= 0
            && static_cast<size_t>(jMax) < color_map.cols
            && static_cast<size_t>(y) < color_map.rows)
        {
            color_map.at<cv::Vec3b>(y, jMax) = cv::Vec3b(0, 128, 0);
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

void Renderer::invertMask(cv::Mat1b& mask)
{
    uchar* mask_ptr = mask.ptr<uchar>();
    size_t pixel_count = mask.rows * mask.cols;
    for (size_t i = 0; i < pixel_count; ++i)
    {
        mask_ptr[i] = mask_ptr[i] ? 0 : 1;
    }
}


void Renderer::computePixelBoardsMask(cv::Mat1b &origin_mask, cv::Mat1b &board_mask)
{
//    for (int row = 0; row < board_mask.rows; ++row)
//    {
//        for (int col = 0; col < board_mask.cols; ++col)
//        {
//            int half_row = row / 2;
//            int half_col = col / 2;
//            bool on_foreground = false;
//            if (origin_mask( half_row, half_col) > 0)
//            {
//                on_foreground = true;
//            }
//            else
//            {
//                bool horizontal_border = row % 2 == 0 && half_row > 0;
//                if (horizontal_border)
//                {
//                    if (origin_mask(half_row - 1, half_col))
//                    {
//                        on_foreground = true;
//                    }
//                    else if (col % 2 == 0 && half_col > 0)   // vertical border
//                    {
//                        on_foreground = origin_mask(half_row, half_col - 1) || origin_mask(half_row - 1, half_col - 1);
//                    }
//                }
//                else if (col % 2 == 0 && half_col > 0)    // vertical border
//                {
//                    on_foreground = origin_mask(half_row, half_col - 1);
//                }
//            }
//            board_mask = (on_foreground) ? 1 : 0;
//        }
//    }
    for (int row = 0; row < origin_mask.rows; ++row)
    {
        for (int col = 0;  col < origin_mask.cols; ++col)
        {
            if (origin_mask(row, col))
            {
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        board_mask(2 * row + i, 2 * col + j) = 1;
                    }
                }
            }
        }
    }
}

void Renderer::computeDistanceOnUpsampled(cv::Mat1b &mask, cv::Mat1f &distance)
{
    cv::Size double_size = cv::Size(distance.size().width * 2 + 1, distance.size().height * 2 + 1);
    cv::Mat1f pixel_boards_distance = cv::Mat1f::zeros(double_size);
    cv::Mat1b pixel_boards_mask = cv::Mat1b::zeros(double_size);
    computePixelBoardsMask(mask, pixel_boards_mask);
    cv::distanceTransform(pixel_boards_mask,
                          pixel_boards_distance,
                          cv::DIST_L2,
                          cv::DIST_MASK_PRECISE,
                          CV_32F);
    for (int row = 0; row < distance.rows; ++row)
    {
        for (int col = 0; col < distance.cols; ++col)
        {
            distance(row, col) = pixel_boards_distance(2 * row + 1, 2 * col + 1);
        }
    }
}



void Renderer::computeSignedDistance(cv::Mat1b& mask, cv::Mat1f& signed_distance)
{
    clock_t start = clock();
//    computeDistanceOnUpsampled(mask, signed_distance);
//    cv::Mat1f internal_signed_distance = cv::Mat::zeros(signed_distance.size(), signed_distance.type());
//    cv::Mat1f external_signed_distance = cv::Mat::zeros(signed_distance.size(), signed_distance.type());
//    computeDistanceOnUpsampled(mask, internal_signed_distance);
//    internal_signed_distance = -internal_signed_distance;
//    invertMask(mask);
//    computeDistanceOnUpsampled(mask, external_signed_distance);
//    invertMask(mask);
//    signed_distance = internal_signed_distance + external_signed_distance;
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
    clock_t finish = clock();
    std::cout << (double) (finish - start) / CLOCKS_PER_SEC << std::endl;
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

cv::Size Renderer::getSize() const
{
    return cv::Size(width, height);
}
