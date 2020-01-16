//#include <fstream>
//#include <glm/gtc/matrix_transform.hpp>
//#include <opencv/cv.hpp>
//#include "GeomUtils.h"
//
//glm::mat4 applyResultToPose(const glm::mat4& matr, const double* params)
//{
//    cv::Matx31f rot_vec = cv::Matx31f(params[0], params[1], params[2]);
//    cv::Matx33f rot_matr = cv::Matx33f();
//    cv::Rodrigues(rot_vec, rot_matr);
//    glm::mat4 difference =
//            glm::mat4(rot_matr(0, 0), rot_matr(1, 0), rot_matr(2, 0), 0,
//                      rot_matr(0, 1), rot_matr(1, 1), rot_matr(2, 1), 0,
//                      rot_matr(0, 2), rot_matr(1, 2), rot_matr(2, 2), 0,
//                      params[3], params[4], params[5], 1);
//    return matr * difference;
//}