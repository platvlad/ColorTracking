#ifndef COLORTRACKING_GROUNDTRUTHPOSEGETTER_H
#define COLORTRACKING_GROUNDTRUTHPOSEGETTER_H
#include <glm/mat4x4.hpp>
#include <boost/filesystem/path.hpp>
#include "PoseGetter.h"
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

    ~GroundTruthPoseGetter() {}
};
#endif //COLORTRACKING_GROUNDTRUTHPOSEGETTER_H
