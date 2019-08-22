#include <test_runner/data.hpp>

namespace testrunner
{

Pose::Pose(
        glm::mat4 const &pose,
        std::vector<glm::mat4> const &bones,
        std::vector<double> const &facemods):
    pose(pose),
    bones(bones),
    facemods(facemods)
{
}

}
// namespace testrunner
