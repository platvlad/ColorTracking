//#include "LktInitTracker.h"
//
//#include <iostream>
//#include "SLSQPPoseGetter.h"
//#include "LkPoseGetter.h"
//
//void LktInitTracker::run()
//{
//    histograms::Object3d& object3D = data.object3D;
//    glm::mat4 pose = data.getPose(1);
//    glm::mat4 prev_pose = pose;
//    int frame_number = 1;
//    cv::Mat3b frame = getFrame();
//    SLSQPPoseGetter color_pose_getter(&object3D, pose);
//
//    const Renderer& renderer = object3D.getRenderer();
//    const glm::mat4& camera_matrix = renderer.getCameraMatrix();
//    LkPoseGetter f_tracker(object3D.getMesh(), pose, camera_matrix, frame.size());
//    f_tracker.handleFrame(frame);
//
//    while (!frame.empty())
//    {
//        object3D.updateHistograms(frame, pose);
//        data.estimated_poses[frame_number] = pose;
//        data.writePng(frame, frame_number);
//        f_tracker.setPrevModel(pose);
//        
//        frame = getFrame();
//        if (frame.empty())
//            break;
//        ++frame_number;
//        glm::mat4 feat_pose = f_tracker.handleFrame(frame);
//        color_pose_getter.setInitialPose(feat_pose);
//        histograms::PoseEstimator estimator;
//        std::cout << frame_number << ' ' << estimator.estimateEnergy(object3D, frame, feat_pose, 10).first << std::endl;
//
//        pose = color_pose_getter.getPose(frame, 0);
//        std::cout << frame_number << ' ' << estimator.estimateEnergy(object3D, frame, pose, 10).first << std::endl;
//        f_tracker.addNewFeatures(pose);
//
//        bool plot_energy = false;
//        if (plot_energy)
//        {
//            //plotRodriguesDirection(object3D, frame, pose, real_pose, directory_name + "/plot/" + std::to_string(frame_number));
//            plotEnergy(object3D, frame, pose, frame_number, data.directory_name);
//            //data.writePlots(frame, frame_number, pose);
//        }
//        
//    }
//    data.writePositions();
//}
