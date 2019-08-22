#ifndef TESTRUNNER_DATA_HPP
#define TESTRUNNER_DATA_HPP

#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace testrunner
{

struct Pose
{
    glm::mat4 pose;
    std::vector<glm::mat4> bones;
    std::vector<double> facemods;

    Pose(glm::mat4 const &pose = glm::mat4(),
         std::vector<glm::mat4> const &bones = std::vector<glm::mat4>(),
         std::vector<double> const &facemods = std::vector<double>());
};


struct Fixation
{
    enum State
    {
        FREE = 0,
        FIXED = 1
    };

    typedef std::vector<State> FixationVec;

    FixationVec euler;
    FixationVec t;
};


struct Fixations
{
    Fixation pose;
    std::vector<Fixation> bones;
};

}

#endif //TESTRUNNER_DATA_HPP
