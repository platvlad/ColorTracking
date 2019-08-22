#ifndef TESTRUNNER_IO_HPP
#define TESTRUNNER_IO_HPP

#include "data.hpp"

#include <map>

#include <boost/filesystem/path.hpp>

#include <opencv2/core/core.hpp>

namespace testrunner
{

glm::mat4 readCamera(boost::filesystem::path const &configPath);

std::map<int, Pose> readPoses(boost::filesystem::path const &configPath);

Fixations readFixations(boost::filesystem::path const &configPath);

float readDiameter(boost::filesystem::path const &path);

std::map<int, boost::filesystem::path> listFrames(
        boost::filesystem::path const &dir);

cv::Mat3f readRgbImage(boost::filesystem::path const &path);

cv::Mat3f readGrayscaleImage(boost::filesystem::path const &path);

void writePoses(std::map<int, Pose> const &poses,
                boost::filesystem::path const &dstPath);

void writeErrors(std::map<int, float> const &errors,
                 boost::filesystem::path const &dstPath);

void writeDiameter(float diameter,
                   boost::filesystem::path const &dstPath);

}
// namespace testrunner

#endif // TESTRUNNER_IO_HPP
