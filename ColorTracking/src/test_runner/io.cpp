#include <test_runner/io.hpp>

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/lexical_cast.hpp>

#include <glm/mat3x3.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <yaml-cpp/yaml.h>

namespace testrunner
{

namespace
{

    template<class T>
    T readMat(YAML::Node const &node)
    {
        T mat;
        int const rowCount = static_cast<int>(node.size());
        for (int rowIdx = 0; rowIdx < rowCount; ++rowIdx) {
            YAML::Node const &row = node[rowIdx];
            int const colCount = static_cast<int>(row.size());
            for (int colIdx = 0; colIdx < colCount; ++colIdx) {
                mat[colIdx][rowIdx] = row[colIdx].as<T::value_type>();
            }
        }
        return mat;
    }


    template<class T>
    T readVec(YAML::Node const &node)
    {
        T vec;
        int const nodeSize = static_cast<int>(node.size());
        for (int i = 0; i < nodeSize; ++i) {
            vec[i] = node[i].as<T::value_type>();
        }
        return vec;
    }

    void operator >>(YAML::Node const &node, Fixation::State &state)
    {
        std::string str = node.as<std::string>();
        if (str == "free") {
            state = Fixation::FREE;
        } else if (str == "fixed") {
            state = Fixation::FIXED;
        } else {
            assert(false);
        }
    }

    glm::mat4 readPosition(YAML::Node const &node)
    {
        glm::vec3 const t = readVec<glm::vec3>(node["t"]);
        glm::mat3 const r = readMat<glm::mat3>(node["R"]);
        return glm::mat4(
                glm::vec4(r[0], 0.0f),
                glm::vec4(r[1], 0.0f),
                glm::vec4(r[2], 0.0f),
                glm::vec4(t, 1.0f));
    }


    void operator >>(YAML::Node const &node, glm::mat4 &pose)
    {
        pose = readPosition(node);
    }

    template<class T>
    T readElem(YAML::Node const &node)
    {
        return node.as<T>();
    }

    template<>
    glm::mat4 readElem<glm::mat4>(YAML::Node const &node)
    {
        return readPosition(node);
    }

    template<>
    Fixation::State readElem<Fixation::State>(YAML::Node const &node)
    {
        Fixation::State state;
        node >> state;
        return state;
    }

    template<class T>
    std::vector<T> readVector(YAML::Node const &node)
    {
        std::vector<T> vec(node.size());
        int const nodeSize = static_cast<int>(node.size());
        for (int i = 0; i < nodeSize; ++i) {
            vec[i] = readElem<T>(node[i]);
        }
        return vec;
    }

    Fixation::FixationVec readFixationVec(YAML::Node const &node)
    {
        return readVector<Fixation::State>(node);
    }


    Fixation readFixation(YAML::Node const &node)
    {
        Fixation const fixation = {
                readFixationVec(node["euler"]),
                readFixationVec(node["t"])};
        return fixation;
    }

    std::pair<int, Pose> readFrameAndPose(YAML::Node const &node)
    {
        std::pair<int, Pose> frameAndPose;

        frameAndPose.first = node["frame"].as<int>();

        frameAndPose.second.pose = readPosition(node["pose"]);

        if (YAML::Node const bonesNode = node["bones"]) {
            frameAndPose.second.bones =
                    readVector<glm::mat4>(bonesNode);
        }

        if (YAML::Node const facemodsNode = node["facemods"]) {
            frameAndPose.second.facemods =
                    readVector<double>(facemodsNode);
        }

        return frameAndPose;
    }
}


glm::mat4 readCamera(boost::filesystem::path const &configPath)
{
    boost::filesystem::ifstream configStream(configPath);
    YAML::Node docNode = YAML::Load(configStream);
    YAML::Node const &matNode = docNode["matrix"];
    return readMat<glm::mat4>(matNode);
}


std::map<int, Pose> readPoses(boost::filesystem::path const &configPath)
{
    boost::filesystem::ifstream configStream(configPath);
    YAML::Node docNode = YAML::Load(configStream);

    YAML::Node const &framesNode = docNode["frames"];
    std::map<int, Pose> allPoses;
    int nodeSize = static_cast<int>(framesNode.size());
    for (int i = 0; i < nodeSize; ++i) {
        allPoses.insert(readFrameAndPose(framesNode[i]));
    }

    return allPoses;
}


Fixations readFixations(boost::filesystem::path const &configPath)
{
    boost::filesystem::ifstream configStream(configPath);
    YAML::Node docNode = YAML::Load(configStream);

    Fixations fixations;

    if (YAML::Node const poseNode = docNode["pose"]) {
        fixations.pose = readFixation(poseNode);
    }

    if (YAML::Node const bonesNode = docNode["bones"]) {
        int const nodeSize = static_cast<int>(bonesNode.size());
        fixations.bones.reserve(nodeSize);
        for (int i = 0; i < nodeSize; ++i) {
            fixations.bones.push_back(readFixation(bonesNode[i]));
        }
    }

    return fixations;
}


float readDiameter(boost::filesystem::path const &path)
{
    boost::filesystem::ifstream stream(path);
    YAML::Node docNode = YAML::Load(stream);

    float diameter = docNode["diameter"].as<float>();

    return diameter;
}


std::map<int, boost::filesystem::path> listFrames(
        boost::filesystem::path const &dir)
{
    typedef boost::filesystem::directory_iterator Iter;
    Iter const end;

    std::map<int, boost::filesystem::path> dst;
    for (Iter it(dir); it != end; ++it) {
        boost::filesystem::path const path = it->path();
        int const frame = boost::lexical_cast<int>(path.stem().string());
        dst.insert(std::make_pair(frame, it->path()));
    }

    return dst;
}


cv::Mat3f readRgbImage(boost::filesystem::path const &path)
{
    cv::Mat3b bgr(cv::imread(path.string().c_str(), cv::IMREAD_COLOR));
    cv::Mat3b rgb;
    cv::cvtColor(bgr, rgb, CV_BGR2RGB);
    cv::Mat3f floatRgb;
    rgb.convertTo(floatRgb, CV_32FC3, 1.0 / 255);
    cv::Mat3f flippedRgb;
    cv::flip(floatRgb, flippedRgb, 0);
    return flippedRgb;
}


cv::Mat3f readGrayscaleImage(boost::filesystem::path const &path)
{
    cv::Mat3f rgb(readRgbImage(path));
    cv::Mat1f gray;
    cv::cvtColor(rgb, gray, CV_RGB2GRAY);
    return gray;
}

}
// namespace testrunner


namespace
{
    std::string protectInf(float const x)
    {
        if (glm::isinf(x)) {
            return (x > 0) ? ".inf" : "-.inf";
        }
        return boost::lexical_cast<std::string>(x);
    }
}


namespace YAML
{

void operator <<(YAML::Emitter &emitter, glm::mat4 const &pose)
{
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "R" << YAML::Value;
    {
        emitter << YAML::BeginSeq;
        for (int row = 0; row < 3; ++row) {
            emitter << YAML::Flow;
            emitter << YAML::BeginSeq;
            emitter << protectInf(pose[0][row])
                    << protectInf(pose[1][row])
                    << protectInf(pose[2][row]);
            emitter << YAML::EndSeq;
        }
        emitter << YAML::EndSeq;
    }
    emitter << YAML::Key << "t" << YAML::Value;
    {
        emitter << YAML::Flow;
        emitter << YAML::BeginSeq;
        emitter << protectInf(pose[3][0])
                << protectInf(pose[3][1])
                << protectInf(pose[3][2]);
        emitter << YAML::EndSeq;
    }
    emitter << YAML::EndMap;
}

}
// namespace YAML


namespace testrunner
{

void writePoses(std::map<int, Pose> const &poses,
                boost::filesystem::path const &dstPath)
{
    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "frames";
    emitter << YAML::Value;
    {
        emitter << YAML::BeginSeq;
        typedef std::pair<int, Pose> FrameAndPose;
        BOOST_FOREACH (FrameAndPose const &frameAndPose, poses) {
            emitter << YAML::BeginMap;
            emitter << YAML::Key << "frame"
                    << YAML::Value << frameAndPose.first;
            emitter << YAML::Key << "pose"
                    << YAML::Value << frameAndPose.second.pose;
            if (!frameAndPose.second.bones.empty()) {
                emitter << YAML::Key << "bones"
                        << YAML::Value << frameAndPose.second.bones;
            }
            if (!frameAndPose.second.facemods.empty()) {
                emitter << YAML::Key << "facemods"
                        << YAML::Value << frameAndPose.second.facemods;
            }
            emitter << YAML::EndMap;
        }
        emitter << YAML::EndSeq;
    }
    emitter << YAML::EndMap;

    boost::filesystem::ofstream dstStream(dstPath);
    dstStream << emitter.c_str();
}


void writeErrors(
        std::map<int, float> const &errors,
        boost::filesystem::path const &dstPath)
{
    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "frames";
    emitter << YAML::Value;
    {
        emitter << YAML::BeginSeq;
        typedef std::pair<int, float> FrameAndError;
        BOOST_FOREACH (FrameAndError const &frameAndError, errors) {
            emitter << YAML::BeginMap;
            emitter << YAML::Key << "frame"
                    << YAML::Value << frameAndError.first;
            emitter << YAML::Key << "error"
                    << YAML::Value << protectInf(frameAndError.second);
            emitter << YAML::EndMap;
        }
        emitter << YAML::EndSeq;
    }
    emitter << YAML::EndMap;

    boost::filesystem::ofstream dstStream(dstPath);
    dstStream << emitter.c_str();
}


void writeDiameter(float diameter, boost::filesystem::path const &dstPath)
{
    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "diameter";
    emitter << YAML::Value << diameter;
    emitter << YAML::EndMap;

    boost::filesystem::ofstream dstStream(dstPath);
    dstStream << emitter.c_str();
}

}
// namespace testrunner
