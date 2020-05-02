#ifndef COLORTRACKING_DATAIO_H
#define COLORTRACKING_DATAIO_H

#include <map>

#include <glm/mat4x4.hpp>
#include <boost/filesystem/path.hpp>
#include "mesh.h"
#include <opencv2/highgui.hpp>
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

    void writePositions(const std::string &file_name = "output.yml");

    void writePng(cv::Mat3b frame, int frame_number);

    void writePlots(const cv::Mat3b &frame, int frame_number, const glm::mat4 &pose);

    glm::mat4 getPose(int frame_number) const;

    int getNumFrames() const;

    static histograms::Mesh getMesh(const boost::filesystem::path& path);

private:
    float mesh_scale_factor;


    static glm::mat4 getCamera(const boost::filesystem::path& path, int height = 512);

    static cv::VideoCapture getVideo(const boost::filesystem::path& filePath);

    static float getZNear(const glm::mat4& pose);
};

#endif //COLORTRACKING_DATAIO_H
