#include "LktTracker.h"

#include <iostream>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>

void LktTracker::plotEnergy(const LkPoseGetter& f_tracker, const glm::mat4& pose, int frame_number)
{
    histograms::Object3d2& object3d = data.object3D2;
    std::string directory_name = data.directory_name;
    int num_points = 100;
    float max_rotation = 0.1f;
    float max_translation = 0.1f * object3d.getMesh().getBBDiameter();

    float rotation_step = max_rotation / static_cast<float>(num_points);
    float translation_step = max_translation / static_cast<float>(num_points);
    std::string base_file_name =
        directory_name + "/plots/" + std::to_string(frame_number);
    std::string tr_x_file = base_file_name + "tr_x.yml";
    std::vector<float> offsets(2 * num_points + 1);
    for (int pt = -num_points; pt <= num_points; ++pt)
    {
        int pose_number = pt + num_points;
        offsets[pose_number] = translation_step * pt;
    }
    glm::vec3 axis(1.0, 0.0, 0.0);
    std::ofstream fout_tr(tr_x_file);
    fout_tr << "frames:" << std::endl;
    for (int i = 0; i < offsets.size(); ++i)
    {
        glm::vec3 offset_vector(axis.x * offsets[i], axis.y * offsets[i], axis.z * offsets[i]);
        glm::mat4 transform = glm::translate(pose, offset_vector);
        fout_tr << "  - frame: " << i + 1 << std::endl;
        fout_tr << "    error: " << f_tracker.getAvgReprojectionError(transform) << std::endl;
    }
    fout_tr.close();
}

void LktTracker::run()
{
    histograms::Object3d2& object3D = data.object3D2;
    cv::Mat3b frame = getFrame();
    int frame_number = 1;
    glm::mat4 pose = data.getPose(1);
    const Renderer& renderer = object3D.getRenderer();
    const glm::mat4& camera_matrix = renderer.getCameraMatrix();
    LkPoseGetter f_tracker(object3D.getMesh(), pose, camera_matrix, frame.size());
    while (!frame.empty())
    {
        std::cout << "frame_number = " << frame_number << std::endl;
        pose = f_tracker.handleFrame(frame);
        data.estimated_poses[frame_number] = pose;
        //data.writePng(frame, frame_number);
        f_tracker.addNewFeatures(pose);
        frame = getFrame();
        if (frame.empty())
        {
            break;
        }
        ++frame_number;
    }
    data.writePositions("output.yml");
}