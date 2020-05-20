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
            //std::cout << frame_number << ' ' << estimator.estimateEnergy(object3D, frame, pose, 10).first << std::endl;
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

    void plotColorTranslation(
        const cv::Mat3b &frame,
        const glm::mat4& pose,
        const std::vector<float> &offsets,
        const glm::vec3& axis,
        std::string file_name)
    {
        std::ofstream fout_tr(file_name);
        fout_tr << "frames:" << std::endl;
        for (int i = 0; i < offsets.size(); ++i)
        {
            glm::vec3 offset_vector(axis.x * offsets[i], axis.y * offsets[i], axis.z * offsets[i]);
            glm::mat4 transform = glm::translate(pose, offset_vector);
            histograms::PoseEstimator2 estimator;
            fout_tr << "  - frame: " << i + 1 << std::endl;
            fout_tr << "    error: " << estimator.estimateEnergy(data.object3D2, frame, transform).first << std::endl;
        }
        fout_tr.close();
    }

    void plotEnergy(const cv::Mat3b& frame, const glm::mat4& pose, int frame_number)
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
        std::string rot_x_file = base_file_name + "rot_x.yml";
        std::string rot_y_file = base_file_name + "rot_y.yml";
        std::string rot_z_file = base_file_name + "rot_z.yml";
        std::string tr_x_file = base_file_name + "tr_x.yml";
        std::string tr_y_file = base_file_name + "tr_y.yml";
        std::string tr_z_file = base_file_name + "tr_z.yml";
        std::vector<float> angles(2 * num_points + 1);
        std::vector<float> offsets(2 * num_points + 1);
        for (int pt = -num_points; pt <= num_points; ++pt)
        {
            int pose_number = pt + num_points;
            angles[pose_number] = rotation_step * pt;
            offsets[pose_number] = translation_step * pt;
        }
        //plotColorRotation(object3d, frame, pose, angles, glm::vec3(1.0, 0.0, 0.0), rot_x_file);
        //plotColorRotation(object3d, frame, pose, angles, glm::vec3(0.0, 1.0, 0.0), rot_y_file);
        //plotColorRotation(object3d, frame, pose, angles, glm::vec3(0.0, 0.0, 1.0), rot_z_file);
        plotColorTranslation(frame, pose, offsets, glm::vec3(1.0, 0.0, 0.0), tr_x_file);
        //plotColorTranslation(object3d, frame, pose, offsets, glm::vec3(0.0, 1.0, 0.0), tr_x_file);
        //plotColorTranslation(object3d, frame, pose, offsets, glm::vec3(0.0, 0.0, 1.0), tr_x_file);

    }

    void run() override
    {
        histograms::Object3d2& object3D = data.object3D2;
        glm::mat4 pose = data.getPose(1);
        SLSQPPoseGetter pose_getter(&object3D, pose);

        int frame_number = 1;
        cv::Mat3b frame = getFrame();
        cv::Mat3b processed_frame;
        equalizeHSV(frame, processed_frame);
        glm::mat4 prev_pose = pose;
        glm::mat4 prev_prev_pose = prev_pose;
        
        while (!frame.empty())
        {
            std::cout << frame_number << std::endl;
            if (frame_number == 122)
            {
                bool for_debug = true;
            }
            object3D.updateHistograms(processed_frame, pose);
            data.estimated_poses[frame_number] = pose;
            //data.writePng(rgb_frame, frame_number);

            frame = getFrame();
            if (frame.empty())
                break;
            equalizeHSV(frame, processed_frame);
            ++frame_number;
            histograms::PoseEstimator2 estimator;
            if (frame_number > 2)
            //if (false)
            {
                glm::mat4 optimization_init;
                //if (frame_number > 3)
                if (false)
                {
                    optimization_init = extrapolate(prev_prev_pose, prev_pose, pose);
                }
                else
                {
                    optimization_init = extrapolate(prev_pose, pose);
                }
                //glm::mat4 optimization_init = pose;
                if (estimator.estimateEnergy(object3D, processed_frame, optimization_init).first < 
                    estimator.estimateEnergy(object3D, processed_frame, pose).first)
                {
                    pose_getter.setInitialPose(optimization_init);
                }
                else
                {
                    //pose_getter.setInitialPose(pose);
                    pose_getter.setInitialPose(optimization_init);
                }
            }

            if (frame_number == 12)
            {
                bool for_debug = true;
            }

            prev_prev_pose = prev_pose;
            prev_pose = pose;
            pose = getPoseOnPyramide(processed_frame, pose_getter, pyramide_levels);
            
            //std::cout << frame_number << ' ' << estimator.estimateEnergy(object3D, processed_frame, pose, 10).first << std::endl;
            /*if (frame_number == 12)
            {
                glm::mat4 real_pose = data.getPose(frame_number);
                std::cout << frame_number << " real " << estimator.estimateEnergy(object3D, processed_frame, real_pose, 10, 1).first << std::endl; 
                std::cout << frame_number << " estimated " << estimator.estimateEnergy(object3D, processed_frame, pose, 10, 2).first << std::endl;
            }*/
            bool plot_energy = false;
            if (plot_energy)
            {
                ////plotRodriguesDirection(object3D, frame, pose, real_pose, directory_name + "/plot/" + std::to_string(frame_number));
                //plotEnergy(object3D, frame, pose, frame_number, data.directory_name);
                ////data.writePlots(frame, frame_number, pose);
            }
        }
        data.writePositions("output.yml");
    }
};