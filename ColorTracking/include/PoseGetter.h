#ifndef COLORTRACKING_POSEGETTER_H
#define COLORTRACKING_POSEGETTER_H

#include <glm/mat4x4.hpp>

class PoseGetter
{
public:
    PoseGetter() {};
    virtual glm::mat4 getPose() = 0;
};

#endif //COLORTRACKING_POSEGETTER_H
