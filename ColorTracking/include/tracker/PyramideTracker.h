#pragma once

#include "tracker/Tracker.h"

template<class Optimizer>
class PyramideTracker: public Tracker
{
    size_t pyramide_levels;
public:
    PyramideTracker(const std::string &directory_name, size_t pyramide_levels = 2) :
        Tracker(directory_name),
        pyramide_levels(pyramide_levels)
    {
    };

    void run() override
    {
        histograms::Object3d& object3D = data.object3D;
        glm::mat4 pose = data.getPose(1);
        Optimizer pose_getter(&object3D, pose);

        int frame_number = 1;
        cv::Mat3b frame = getFrame();
        cv::Mat3b processed_frame = processFrame(frame);
        glm::mat4 prev_pose = pose;
        while (!frame.empty())
        {
            object3D.updateHistograms(processed_frame, pose);
            data.estimated_poses[frame_number] = pose;
            data.writePng(processed_frame, frame_number);

            frame = getFrame();
            if (frame.empty())
                break;
            processed_frame = processFrame(frame);
            ++frame_number;

            if (frame_number > 2)
            {
                glm::mat4 optimization_init = extrapolate(prev_pose, pose);
                //glm::mat4 optimization_init = pose;
                pose_getter.setInitialPose(optimization_init);
            }

            prev_pose = pose;
            pose = getPoseOnPyramide(processed_frame, pose_getter, pyramide_levels);
            histograms::PoseEstimator estimator;
            std::cout << frame_number << ' ' << estimator.estimateEnergy(object3D, processed_frame, pose, 10).first << std::endl;
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
};
