#include "tracker/GroundTruthTracker.h"

#include <iostream>
#include "PoseEstimator.h"
#include "pose_getter/GroundTruthPoseGetter.h"

void GroundTruthTracker::run()
{
    histograms::Object3d& object3D = data.object3D;
    glm::mat4 pose = data.getPose(1);
    GroundTruthPoseGetter pose_getter(data);

    int frame_number = 1;
    cv::Mat3b frame = getFrame();
    
    pose_getter.getPose(frame, 0);

    while (!frame.empty())
    {
        object3D.updateHistograms(frame, pose);
        data.estimated_poses[frame_number] = pose;
        data.writePng(frame, frame_number);

        frame = getFrame();
        if (frame.empty())
            break;
        ++frame_number;
        pose = pose_getter.getPose(frame, 0);
        histograms::PoseEstimator estimator;
        std::cout << frame_number << ' ' << estimator.estimateEnergy(object3D, frame, pose, 10).first << std::endl;
        bool plot_energy = false;
        if (plot_energy)
        {
            //plotRodriguesDirection(object3D, frame, pose, real_pose, directory_name + "/plot/" + std::to_string(frame_number));
            plotEnergy(object3D, frame, pose, frame_number, data.directory_name);
            //data.writePlots(frame, frame_number, pose);
        }
    }
    data.writePositions();

}