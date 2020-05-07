#include <set>

#include<opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>

#include "renderer.h"

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


Projection Renderer::projectMesh(const histograms::Mesh& mesh,
                           const glm::mat4& pose,
                           const cv::Mat3b &frame,
                           int frame_offset,
                           bool compute_signed_distance) const
{
    Projection maps(frame, frame_offset);
    const std::vector<glm::uvec3>& faces = mesh.getFaces();
    const std::vector<glm::vec3>& vertices = mesh.getVertices();
    maps.vertex_projections = std::vector<glm::vec3>(vertices.size());
    std::vector<glm::vec4> vertex_transforms = std::vector<glm::vec4>(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        vertex_transforms[i] = transformVertex(vertices[i], pose);
        maps.vertex_projections[i] = projectTransformedVertex(vertex_transforms[i]);
        roundXY(maps.vertex_projections[i]);  // inplace
    }

    for (size_t i = 0; i < faces.size(); ++i)
    {
        glm::uvec3 face = faces[i];
        renderTriangle(maps, vertex_transforms, face);
    }
    if (frame_offset >= 0)
    {
        maps.trimToExtendedROI();
    }
    if (compute_signed_distance)
    {
        maps.computeSignedDistance();
        maps.computeHeaviside();
    }
    return maps;
}

// version for flipped frames
Projection Renderer::projectMesh2(const histograms::Mesh& mesh,
    const glm::mat4& pose,
    const cv::Mat3b &frame,
    int frame_offset,
    bool compute_signed_distance) const
{
    Projection maps(frame, frame_offset);
    const std::vector<glm::uvec3>& faces = mesh.getFaces();
    const std::vector<glm::vec3>& vertices = mesh.getVertices();
    maps.vertex_projections = std::vector<glm::vec3>(vertices.size());
    std::vector<glm::vec4> vertex_transforms(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        vertex_transforms[i] = transformVertex(vertices[i], pose);
        maps.vertex_projections[i] = projectTransformedVertex2(vertex_transforms[i]);
        roundXY(maps.vertex_projections[i]);  // inplace
    }

    for (size_t i = 0; i < faces.size(); ++i)
    {
        glm::uvec3 face = faces[i];
        renderTriangle(maps, vertex_transforms, face);
    }
    if (frame_offset >= 0)
    {
        maps.trimToExtendedROI();
    }
    if (compute_signed_distance)
    {
        maps.computeSignedDistance();
        maps.computeHeaviside();
    }
    return maps;
}


glm::vec4 Renderer::transformVertex(const glm::vec3 &vertex, const glm::mat4 &pose) const
{
    glm::vec4 v_hom = glm::vec4(vertex, 1);
    return pose * v_hom;
}

glm::vec3 Renderer::projectTransformedVertex(const glm::vec4& vertex) const
{
    glm::vec4 p_hom = camera_matrix * vertex;
    if (p_hom[3] != 0.0f)
    {
        return glm::vec3(p_hom[0] / p_hom[3], -2 * camera_matrix[2][1] - (p_hom[1] / p_hom[3]), p_hom[3]);
    }
    return glm::vec3();
}

glm::vec3 Renderer::projectTransformedVertex2(const glm::vec4& vertex) const
{
    glm::vec4 p_hom = camera_matrix * vertex;
    if (p_hom[3] != 0.0f)
    {
        return glm::vec3(p_hom[0] / p_hom[3], (p_hom[1] / p_hom[3]), p_hom[3]);
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
// x, y, z -- transformed point before projection to camera
// before: depth[:, :] = std::numeric_limits<float>::max()
void Renderer::renderTriangle(Projection& maps, const std::vector<glm::vec4> &vertex_transforms, glm::uvec3& face) const
{
    int i0 = face[0];
    int i1 = face[1];
    int i2 = face[2];
    glm::vec3 p0 = maps.vertex_projections[i0];
    glm::vec3 p1 = maps.vertex_projections[i1];
    glm::vec3 p2 = maps.vertex_projections[i2];
    if (p0.z == 0.0 || p1.z == 0 || p2.z == 0)
    {
        return;
    }
    glm::vec4 tx0 = vertex_transforms[i0];
    glm::vec4 tx1 = vertex_transforms[i1];
    glm::vec4 tx2 = vertex_transforms[i2];

    if (p0.y == p1.y && p0.y == p2.y) return;

    if (p0.y > p1.y)
    {
        std::swap(p0, p1);
        std::swap(tx0, tx1);
    }
    if (p0.y > p2.y)
    {
        std::swap(p0, p2);
        std::swap(tx0, tx2);
    }
    if (p1.y > p2.y)
    {
        std::swap(p1, p2);
        std::swap(tx1, tx2);
    }

    int const yMin = static_cast<int>(p0.y);
    bool const secondHalfPrecalc = p1.y == p0.y;
    float const totalHeight = p2.y - p0.y;
    for (int i = 0; i < totalHeight; ++i) {
        int const y = yMin + i;
        if (y >= height)
            break;
        if (y < 0)
            continue;
        bool const secondHalf = secondHalfPrecalc || i > p1.y - p0.y;
        float const segmentHeight = secondHalf ? p2.y - p1.y : p1.y - p0.y;

        float const alpha = i / totalHeight;
        float const beta = (i - (secondHalf ? p1.y - p0.y : 0))
                           / segmentHeight;

        glm::vec3 a(p0 + (p2 - p0) * alpha);
        glm::vec4 tx_a(tx0 + (tx2 - tx0) * alpha);
        glm::vec3 b(secondHalf ? (p1 + (p2 - p1) * beta)
                               : (p0 + (p1 - p0) * beta));
        glm::vec4 tx_b(secondHalf ? (tx1 + (tx2 - tx1) * beta)
                                  : (tx0 + (tx1 - tx0) * beta));
        if (a.x > b.x) {
            std::swap(a, b);
            std::swap(tx_a, tx_b);
        }

        int const jMin = static_cast<int>(ceil(a.x));
        int const jMax = static_cast<int>(floor(b.x));
        bool const thinLine = jMin == jMax;
        for (int j = jMin; j <= jMax; ++j) {
            int const x = j;
            if (x >= width)
                break;
            if (x < 0)
                continue;
            float const phi = thinLine ? 1.0f : (j - a.x) / (b.x - a.x);
            glm::vec3 const p(a * 1.0f + (b - a) * phi);
            glm::vec4 const tx_p(tx_a * 1.0f + (tx_b - tx_a) * phi);

            if (x < 0 || y < 0
                || static_cast<size_t>(x) >= width
                || static_cast<size_t>(y) >= height) {
                continue;
            }
            if (maps.depth_map.at<cv::Vec3f>(y, x)[2] > p.z) {  // cv::Mat1f
                maps.depth_map.at<cv::Vec3f>(y, x)[0] = tx_p.x;
                maps.depth_map.at<cv::Vec3f>(y, x)[1] = tx_p.y;
                maps.depth_map.at<cv::Vec3f>(y, x)[2] = p.z;
                maps.mask.at<uchar>(y, x) = 255;
                updateROI(maps.roi, x, y);
            }
        }
    }
}

std::set<int> getFaceSet(cv::Mat1i &face_ids)
{
    std::set<int> result;
    for (int row = 0; row < face_ids.rows; ++row)
    {
        for (int col = 0; col < face_ids.cols; ++col)
        {
            int face_id = face_ids(row, col);
            if (face_id >= 0)
            {
                result.insert(face_id);
            }
        }
    }
    return result;
}

void Renderer::renderMesh(const histograms::Mesh &mesh, cv::Mat3b& frame, const glm::mat4 &pose) const
{
    const std::vector<glm::uvec3>& faces = mesh.getFaces();
    const std::vector<glm::vec3>& vertices = mesh.getVertices();

    std::vector<glm::vec3> vertex_projections(vertices.size());
    std::vector<glm::vec4> vertex_transforms(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        vertex_transforms[i] = transformVertex(vertices[i], pose);
        vertex_projections[i] = projectTransformedVertex2(vertex_transforms[i]);
    }

    cv::Size frame_size = frame.size();
    cv::Mat1f depth_map(frame_size, std::numeric_limits<float>::max());
    cv::Rect roi;
    cv::Mat1i face_ids(frame_size, -1);

    for (size_t i = 0; i < faces.size(); ++i)
    {
        glm::uvec3 face = faces[i];
        
        glm::vec3 p0 = vertex_projections[face[0]];
        glm::vec3 p1 = vertex_projections[face[1]];
        glm::vec3 p2 = vertex_projections[face[2]];
        renderTriangleFaceId(depth_map, roi, face_ids, i, p0, p1, p2);
    }
    std::set<int> face_set = getFaceSet(face_ids(roi));

    for (std::set<int>::iterator face_it = face_set.begin(); face_it != face_set.end(); ++face_it)
    {
        glm::uvec3 face = faces[*face_it];

        glm::vec3& p0 = vertex_projections[face[0]];
        glm::vec3& p1 = vertex_projections[face[1]];
        glm::vec3& p2 = vertex_projections[face[2]];
        std::vector<cv::Point> img_pts(3);
        img_pts[0] = cv::Point(p0[0], p0[1]);
        img_pts[1] = cv::Point(p1[0], p1[1]);
        img_pts[2] = cv::Point(p2[0], p2[1]);

        if (p0 != glm::vec3() && p1 != glm::vec3() && p2 != glm::vec3())
        {
            cv::polylines(frame, img_pts, true, cv::Scalar(0, 127, 0, 0.5));
        }
    }
    
}

void Renderer::renderTriangleFaceId(cv::Mat1f& depth_map, cv::Rect& roi, cv::Mat1i& face_ids, 
    int face_num, glm::vec3 &p0, glm::vec3 &p1, glm::vec3 &p2) const
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
            if (depth_map(y, x) > p.z) {  // cv::Mat1f
                depth_map(y, x) = p.z;
                face_ids(y, x) = face_num;
                updateROI(roi, x, y);
            }
        }
    }
}

void Renderer::updateROI(cv::Rect& roi, int x, int y)
{
    if (roi.width == 0 && roi.height == 0)
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

cv::Size Renderer::getSize() const
{
    return cv::Size(width, height);
}

glm::vec2 Renderer::getFocal() const
{
    return glm::vec2(camera_matrix[0][0], camera_matrix[1][1]);
}

const glm::mat4& Renderer::getCameraMatrix() const 
{
    return camera_matrix;
}