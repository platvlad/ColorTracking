#ifndef COLORTRACKING_TESTS_H
#define COLORTRACKING_TESTS_H

class Tests
{
    static Maps projectPyramid();

    static histograms::Mesh getPyramidMesh();

    static glm::mat4 getPyramidCameraMatrix();

    static glm::mat4 getPyramidPose();

public:

    static void testProjection();

    static void testGetROI();

    static void testHeaviside();

    static void testUpdateHistograms();

    static void testEvaluateEnergy();

    static void runTests();
};

#endif //COLORTRACKING_TESTS_H
