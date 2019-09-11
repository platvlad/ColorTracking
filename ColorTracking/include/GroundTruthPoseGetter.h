#ifndef COLORTRACKING_GROUNDTRUTHPOSEGETTER_H
#define COLORTRACKING_GROUNDTRUTHPOSEGETTER_H
#include <glm/mat4x4.hpp>
#include <boost/filesystem/path.hpp>
#include "PoseGetter.h"

class GroundTruthPoseGetter : public PoseGetter
{
    int frame_number;
    int current_frame;
    const boost::filesystem::path path;
public:
    explicit GroundTruthPoseGetter(const boost::filesystem::path& ground_truth_path);

    glm::mat4 getPose(const cv::Mat& frame);

    glm::mat4 getPose(int frame_index);
};
#endif //COLORTRACKING_GROUNDTRUTHPOSEGETTER_H
