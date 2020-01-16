#ifndef COLORTRACKING_GEOMUTILS_H
#define COLORTRACKING_GEOMUTILS_H

#include <glm/ext/matrix_float4x4.hpp>
#include <Object3d.h>

glm::mat4 applyResultToPose(const glm::mat4& matr, const double* params);

#endif //COLORTRACKING_GEOMUTILS_H
