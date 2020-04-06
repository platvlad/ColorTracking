#ifndef COLORTRACKING_GROUNDTRUTHPOSEGETTER_H
#define COLORTRACKING_GROUNDTRUTHPOSEGETTER_H
#include <glm/mat4x4.hpp>
#include <boost/filesystem/path.hpp>
#include "pose_getter/PoseGetter.h"
#include "DataIO.h"

class GroundTruthPoseGetter : public PoseGetter
{
    int frame_number;
    int current_frame;
    const DataIO* dataIO;
public:
    explicit GroundTruthPoseGetter(const DataIO& dataIO);

    glm::mat4 getPose(const cv::Mat& frame);

    glm::mat4 getPose(const cv::Mat& frame, int mode);

    glm::mat4 getPose(int frame_index);

    virtual void setInitialPose(const glm::mat4 &pose) {}

    ~GroundTruthPoseGetter() {}
};
#endif //COLORTRACKING_GROUNDTRUTHPOSEGETTER_H
