#ifndef COLORTRACKING_INPUTREADER_H
#define COLORTRACKING_INPUTREADER_H

#include <glm/mat4x4.hpp>
#include <boost/filesystem/path.hpp>
#include "mesh.h"
#include <opencv2/videoio.hpp>
#include <Object3d.h>

class InputReader
{
public:
    histograms::Object3d object3D;
    cv::VideoCapture videoCapture;
    boost::filesystem::path ground_truth_path;

    explicit InputReader(const std::string& directory_name);

    static histograms::Mesh getMesh(const boost::filesystem::path& path);

    static glm::mat4 getCamera(const boost::filesystem::path& path, int height = 512);

    static glm::mat4 getPose(const boost::filesystem::path& path, int frame_number=1);

    static int getNumFrames(const boost::filesystem::path& path);

    static cv::VideoCapture getVideo(const boost::filesystem::path& filePath);

    static float getZNear(const glm::mat4& pose);
};

#endif //COLORTRACKING_INPUTREADER_H
