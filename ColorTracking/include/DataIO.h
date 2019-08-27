#ifndef COLORTRACKING_DATAIO_H
#define COLORTRACKING_DATAIO_H

#include <map>

#include <glm/mat4x4.hpp>
#include <boost/filesystem/path.hpp>
#include "mesh.h"
#include <opencv2/videoio.hpp>
#include <Object3d.h>

#include "test_runner/io.hpp"

class DataIO
{
public:
    histograms::Object3d object3D;
    cv::VideoCapture videoCapture;
    std::string directory_name;
    boost::filesystem::path ground_truth_path;
    std::map<int, testrunner::Pose> estimated_poses;

    explicit DataIO(const std::string& directory_name);

    void writePositions();

    void writePng(cv::Mat3b frame, int frame_number);

    static histograms::Mesh getMesh(const boost::filesystem::path& path);

    static glm::mat4 getCamera(const boost::filesystem::path& path, int height = 512);

    static glm::mat4 getPose(const boost::filesystem::path& path, int frame_number=1);

    static int getNumFrames(const boost::filesystem::path& path);

    static cv::VideoCapture getVideo(const boost::filesystem::path& filePath);

    static float getZNear(const glm::mat4& pose);
};

#endif //COLORTRACKING_DATAIO_H
