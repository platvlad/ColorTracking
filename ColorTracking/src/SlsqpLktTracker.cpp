#include "SlsqpLktTracker.h"

#include <iostream>
#include "SlsqpLktPoseGetter.h"

void SlsqpLktTracker::run()
{
    histograms::Object3d& object3D = data.object3D;
    glm::mat4 pose = data.getPose(1);
    glm::mat4 prev_pose = pose;
    int frame_number = 1;
    cv::Mat3b frame = getFrame();
    SlsqpLktPoseGetter pose_getter(&object3D, pose, frame);

    while (!frame.empty())
    {
        object3D.updateHistograms(frame, pose);
        data.estimated_poses[frame_number] = pose;
        data.writePng(frame, frame_number);

        frame = getFrame();
        if (frame.empty())
            break;
        ++frame_number;
        bool plot_energy = false;
        if (plot_energy)
        {
            pose = pose_getter.getPose(frame, 0, data.directory_name, frame_number);
        }
        else
        {
            pose = pose_getter.getPose(frame, 0);
        }

        histograms::PoseEstimator estimator;
        std::cout << frame_number << ' ' << estimator.estimateEnergy(object3D, frame, pose, 10).first << std::endl;
    }
    data.writePositions();
}
