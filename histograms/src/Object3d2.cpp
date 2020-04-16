#include <set>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "mesh.h"
#include "Object3d2.h"

namespace histograms
{
    const float Object3d2::lambda = 0.1;

    Object3d2::Object3d2(const Mesh& mesh, const Renderer& renderer) : 
        frame_offset(40),
        mesh(mesh),
        renderer(renderer),
        histograms(32, Histogram2())
    {
        common_histogram = new Histogram2();
    }

    Object3d2::Object3d2(const Object3d2 & other)
    {
        frame_offset = other.frame_offset;
        renderer = other.renderer;
        mesh = other.mesh;
        histograms = other.histograms;
        common_histogram = new Histogram2(*(other.common_histogram));
    }

    Object3d2 & Object3d2::operator=(const Object3d2 & other)
    {
        delete common_histogram;
        frame_offset = other.frame_offset;
        renderer = other.renderer;
        mesh = other.mesh;
        histograms = other.histograms;
        common_histogram = new Histogram2(*(other.common_histogram));
        return *this;
    }

    int Object3d2::getHistogram(cv::Vec3f transformed_pt, const glm::mat4 & pose_inv) const
    {
        glm::vec4 transformed_hom(transformed_pt[0], transformed_pt[1], transformed_pt[2], 1.0f);
        glm::vec4 obj_pt_3d_hom = pose_inv * transformed_hom;
        glm::vec3 obj_pt(obj_pt_3d_hom);
        obj_pt = obj_pt - mesh.getCenter();
        obj_pt = glm::normalize(obj_pt);
        int histo_class = obj_pt.x > 0 ? 1 : 0;
        histo_class *= 2;
        histo_class += (obj_pt.y > 0 ? 1 : 0);
        histo_class *= 2;
        histo_class += (obj_pt.z > 0 ? 1 : 0);
        histo_class *= 2;
        histo_class += (abs(obj_pt.x) > abs(obj_pt.y) ? 1 : 0);
        histo_class *= 2;
        histo_class += (abs(obj_pt.z) > 0.5 ? 1 : 0);
        return histo_class;
    }

    cv::Mat3f Object3d2::getTransformedBorderPoints(const Projection &projection)
    {
        cv::Size maps_size = projection.getSize();
        std::map<int, cv::Vec3f> labels_transformed_pts;
        cv::Mat3f transformed_pts(projection.depth_map.size());

        for (int row = 0; row < maps_size.height; ++row)
        {
            for (int col = 0; col < maps_size.width; ++col)
            {
                float signed_distance_value = projection.signed_distance(row, col);
                if (signed_distance_value < 0)
                {
                    cv::Vec3f depth_value = projection.depth_map(row, col);
                    cv::Vec3f transformed_pt = depth_value;
                    transformed_pt[2] = -transformed_pt[2];
                    transformed_pts(row, col) = transformed_pt;
                    if (signed_distance_value >= -1.5)
                    {
                        int label = projection.nearest_labels(row, col);
                        labels_transformed_pts[label] = transformed_pt;
                    }
                }
                
            }
        }
        for (int row = 0; row < maps_size.height; ++row)
        {
            for (int col = 0; col < maps_size.width; ++col)
            {
                if (projection.signed_distance(row, col) >= 0)
                {
                    int label = projection.nearest_labels(row, col);
                    transformed_pts(row, col) = labels_transformed_pts[label];
                }
            }
        }
        return transformed_pts;
    }

    void Object3d2::updateHistograms(const cv::Mat3b& frame, const glm::mat4& pose, bool debug_info)
    {
        Projection projection = renderer.projectMesh2(mesh, pose, frame, frame_offset);

        if (projection.hasEmptyProjection())
        {
            return;
        }
        cv::Mat3f transformed_pts = getTransformedBorderPoints(projection);

        cv::Mat1f& signed_distance = projection.signed_distance;
        cv::Mat1b& mask = projection.mask;
        cv::Mat1f& heaviside = projection.heaviside;
        cv::Size projection_size = projection.getSize();
        glm::mat4 pose_inv = glm::inverse(pose);

        std::vector<std::pair<cv::Vec3b, float> > color_heavisides[32];
        std::vector<std::pair<cv::Vec3b, float> > common_color_heavisides;

        cv::Mat3b histo_nums = cv::Mat3b::zeros(projection_size);

        for (int row = 0; row < projection_size.height; ++row)
        {
            for (int col = 0; col < projection_size.width; ++col)
            {
                if (abs(signed_distance(row, col)) <= frame_offset)
                {
                    cv::Vec3f transformed_pt = transformed_pts(row, col);
                    int histo_num = getHistogram(transformed_pt, pose_inv);
                    cv::Vec3b& color_value = projection.color_map(row, col);
                    float heaviside_value = projection.heaviside(row, col);
                    std::pair<cv::Vec3b, float> color_heaviside_pair(color_value, heaviside_value);
                    color_heavisides[histo_num].push_back(color_heaviside_pair);
                    common_color_heavisides.push_back(color_heaviside_pair);
                    if (debug_info)
                    {
                        histo_nums(row, col) = cv::Vec3b(histo_num * 8, (histo_num + 1) * 16 % 255, (histo_num + 2) * 24 % 255);
                    }
                }
            }
        }

        if (debug_info)
        {
            cv::imwrite("C:\\MyProjects\\repo\\VSProjects\\ColorTracking\\ColorTracking\\build\\data\\debug_frames\\histo_nums.png", histo_nums);
        }

        for (int i = 0; i < 32; ++i)
        {
            histograms[i].update(color_heavisides[i]);
        }
        common_histogram->update(common_color_heavisides);

    }

    const Mesh& Object3d2::getMesh() const
    {
        return mesh;
    }

    int Object3d2::getFrameOffset() const
    {
        return frame_offset;
    }

    const Renderer& Object3d2::getRenderer() const
    {
        return renderer;
    }

    cv::Mat1f Object3d2::findColorForeground(const Projection &projection, const cv::Mat3b & frame, const glm::mat4 & pose) const
    {
        cv::Size projection_size = projection.getSize();
        cv::Mat1f votes_fg = cv::Mat1f::zeros(projection_size);
        const cv::Mat1f& signed_distance = projection.signed_distance;

        cv::Mat3f transformed_pts = getTransformedBorderPoints(projection);
        glm::mat4 pose_inv = glm::inverse(pose);
        float eta_f[32] = { 0 };
        float eta_b[32] = { 0 };
        float common_eta_f = 0;
        float common_eta_b = 0;

        cv::Mat1b histo_nums = cv::Mat1b::zeros(projection_size);

        for (int row = 0; row < projection_size.height; ++row)
        {
            for (int col = 0; col < projection_size.width; ++col)
            {
                if (abs(signed_distance(row, col)) <= frame_offset)
                {
                    cv::Vec3f transformed_pt = transformed_pts(row, col);
                    int histo_num = getHistogram(transformed_pt, pose_inv);
                    const cv::Vec3b& color_value = projection.color_map(row, col);
                    float heaviside_value = projection.heaviside(row, col);
                    eta_f[histo_num] += heaviside_value;
                    eta_b[histo_num] += 1 - heaviside_value;
                    common_eta_f += heaviside_value;
                    common_eta_b += 1 - heaviside_value;
                    histo_nums(row, col) = histo_num;
                }
            }
        }

        float histo_skills[32];
        for (size_t i = 0; i < 32; ++i)
        {
            histo_skills[i] = histograms[i].getSkill();
        }
        float common_histo_skill = common_histogram->getSkill();
        float sufficient_skill = common_histo_skill / 32;

        for (int row = 0; row < projection_size.height; ++row)
        {
            for (int col = 0; col < projection_size.width; ++col)
            {
                if (abs(signed_distance(row, col)) <= frame_offset)
                {
                    float heaviside_value = projection.heaviside(row, col);
                    int histo_num = histo_nums(row, col);
                    cv::Vec3b color = projection.color_map(row, col);
                    float histo_vote = histograms[histo_num].voteColor(color, eta_f[histo_num], eta_b[histo_num]);
                    float histo_skill = histo_skills[histo_num];
                    if (histo_skill >= sufficient_skill)
                    {
                        votes_fg(row, col) = histo_vote;
                    }
                    else
                    {
                        float common_vote = common_histogram->voteColor(color, common_eta_f, common_eta_b);
                        votes_fg(row, col) = (histo_vote * histo_skill + common_vote * (sufficient_skill - histo_skill)) / sufficient_skill;
                    }
                }
            }
        }

        return votes_fg;
    }

    Object3d2::~Object3d2()
    {
        delete common_histogram;
    }

}
