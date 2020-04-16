#pragma once

#include "Tracker.h"
#include "Tracker2.h"

template<class Optimizer>
class PyramideTracker: public Tracker
{
    size_t pyramide_levels;
public:
    PyramideTracker(const std::string &directory_name, size_t pyramide_levels = 3) :
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
        glm::mat4 prev_pose = pose;
        while (!frame.empty())
        {
            object3D.updateHistograms(frame, pose);
            data.estimated_poses[frame_number] = pose;
            data.writePng(frame, frame_number);

            frame = getFrame();
            if (frame.empty())
                break;
            ++frame_number;

            if (frame_number > 2)
            {
                glm::mat4 optimization_init = extrapolate(prev_pose, pose);
                //glm::mat4 optimization_init = pose;
                pose_getter.setInitialPose(optimization_init);
            }

            prev_pose = pose;
            pose = getPoseOnPyramide(frame, pose_getter, pyramide_levels);
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
};

template<>
class PyramideTracker<SLSQPPoseGetter>: public Tracker2
{
    size_t pyramide_levels;
public:
    PyramideTracker(const std::string &directory_name, size_t pyramide_levels = 3) :
        Tracker2(directory_name),
        pyramide_levels(pyramide_levels)
    {
    };

    void run() override
    {
        histograms::Object3d2& object3D = data.object3D2;
        glm::mat4 pose = data.getPose(1);
        SLSQPPoseGetter pose_getter(&object3D, pose);

        int frame_number = 1;
        cv::Mat3b frame = getFrame();
        glm::mat4 prev_pose = pose;

        
        while (!frame.empty())
        {
            object3D.updateHistograms(frame, pose, frame_number == 4);
            data.estimated_poses[frame_number] = pose;
            data.writePng(frame, frame_number);

            frame = getFrame();
            if (frame.empty())
                break;
            ++frame_number;

            if (frame_number > 2)
            {
                glm::mat4 optimization_init = extrapolate(prev_pose, pose);
                //glm::mat4 optimization_init = pose;
                pose_getter.setInitialPose(optimization_init);
            }

            prev_pose = pose;
            pose = getPoseOnPyramide(frame, pose_getter, pyramide_levels);
            histograms::PoseEstimator2 estimator;
            std::cout << frame_number << ' ' << estimator.estimateEnergy(object3D, frame, pose, 10).first << std::endl;
            if (frame_number == 4)
            {
                glm::mat4 real_pose = data.getPose(frame_number);
                std::cout << frame_number << " real " << estimator.estimateEnergy(object3D, frame, real_pose, 10, true).first << std::endl;
            }
            bool plot_energy = false;
            if (plot_energy)
            {
                ////plotRodriguesDirection(object3D, frame, pose, real_pose, directory_name + "/plot/" + std::to_string(frame_number));
                //plotEnergy(object3D, frame, pose, frame_number, data.directory_name);
                ////data.writePlots(frame, frame_number, pose);
            }
        }
        data.writePositions();
    }
};