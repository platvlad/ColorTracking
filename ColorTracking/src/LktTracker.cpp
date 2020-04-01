#include "LktTracker.h"

#include "LkPoseGetter.h"

void LktTracker::run()
{
    histograms::Object3d& object3D = data.object3D;
    cv::Mat3b frame = getFrame();
    int frame_number = 1;
    glm::mat4 pose = data.getPose(1);
    const Renderer& renderer = object3D.getRenderer();
    const glm::mat4& camera_matrix = renderer.getCameraMatrix();
    LkPoseGetter f_tracker(object3D.getMesh(), pose, camera_matrix, frame.size());
    while (!frame.empty())
    {
        pose = f_tracker.handleFrame(frame);
        data.estimated_poses[frame_number] = pose;
        data.writePng(frame, frame_number);
        f_tracker.addNewFeatures(pose);
        frame = getFrame();
        if (frame.empty())
        {
            break;
        }
        ++frame_number;
    }
    data.writePositions();
}