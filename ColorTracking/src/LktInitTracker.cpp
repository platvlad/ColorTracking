#include "LktInitTracker.h"

#include <iostream>
#include "SLSQPPoseGetter.h"
#include "LkPoseGetter.h"

void LktInitTracker::run()
{
    histograms::Object3d2& object3D = data.object3D2;
    glm::mat4 pose = data.getPose(1);
    glm::mat4 prev_pose = pose;
    glm::mat4 prev_prev_pose = prev_pose;
    int frame_number = 1;
    cv::Mat3b frame = getFrame();
    cv::Mat3b processed_frame;
    equalizeHSV(frame, processed_frame);
    SLSQPPoseGetter color_pose_getter(&object3D, pose);

    const Renderer& renderer = object3D.getRenderer();
    const glm::mat4& camera_matrix = renderer.getCameraMatrix();
    LkPoseGetter f_tracker(object3D.getMesh(), pose, camera_matrix, frame.size());
    f_tracker.handleFrame(frame);

    while (!frame.empty())
    {
        std::cout << "frame number = " << frame_number << std::endl;
        object3D.updateHistograms(processed_frame, pose);
        data.estimated_poses[frame_number] = pose;
        data.writePng(frame, frame_number);
        f_tracker.setPrevModel(pose);
        
        frame = getFrame();
        if (frame.empty())
            break;
        equalizeHSV(frame, processed_frame);
        ++frame_number;
        glm::mat4 feat_pose = f_tracker.handleFrame(frame);
        histograms::PoseEstimator2 estimator;

        float feat_pose_error = estimator.estimateEnergy(object3D, processed_frame, feat_pose, 100).first;
        float old_pose_error = estimator.estimateEnergy(object3D, processed_frame, pose, 100).first;
        if (frame_number > 2)
        //if (false)
        {
            glm::mat4 extrapolated = extrapolate(prev_pose, pose);
            
            extrapolated = prev_pose;
            float extrapolated_pose_error = estimator.estimateEnergy(object3D, processed_frame, extrapolated, 100).first;
            //std::cout << "feat pose error = " << feat_pose_error << std::endl;
            //std::cout << "extrapolated pose error = " << extrapolated_pose_error << std::endl;
            //if (extrapolated_pose_error < feat_pose_error && extrapolated_pose_error < old_pose_error)
            if (extrapolated_pose_error < feat_pose_error)
            {
                color_pose_getter.setInitialPose(extrapolated);
            }
            //else if (feat_pose_error < old_pose_error)
            else
            {
                color_pose_getter.setInitialPose(feat_pose);
            }
            //else
            //{
              //  color_pose_getter.setInitialPose(pose);
            //}
        }
        else
        {
            //if (feat_pose_error < old_pose_error)
            //{
                color_pose_getter.setInitialPose(feat_pose);
            //}
            //else
            //{
                //color_pose_getter.setInitialPose(pose);
            //}
        }

        prev_prev_pose = prev_pose;
        prev_pose = pose;
        pose = color_pose_getter.getPose(processed_frame, 0);
        //std::cout << frame_number << ' ' << estimator.estimateEnergy(object3D, processed_frame, pose, 10).first << std::endl;
        f_tracker.addNewFeatures(pose);

        bool plot_energy = false;
        if (plot_energy)
        {
            ////plotRodriguesDirection(object3D, frame, pose, real_pose, directory_name + "/plot/" + std::to_string(frame_number));
            //plotEnergy(object3D, frame, pose, frame_number, data.directory_name);
            ////data.writePlots(frame, frame_number, pose);
        }
        
    }
    data.writePositions("output_lkt_init_rgb_no_reproj.yml");
}
