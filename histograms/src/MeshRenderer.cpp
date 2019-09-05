//
//#include <MeshRenderer.h>
//#include <OpenGL/OpenGL.h>
//#include <OpenGL/glext.h>
//
//
//histograms::MeshRenderer::MeshRenderer(const histograms::Mesh &mesh, const glm::mat4 &cameraMatrix,
//                                       float zNear, float zFar, int width, int height) : mesh(mesh),
//                                                                                         camera_matrix(cameraMatrix),
//                                                                                         z_near(zNear), z_far(zFar),
//                                                                                         width(width), height(height)
//{
//    std::vector<unsigned int> indices;
//    GLuint elementbuffer;
//    glGenBuffersARB(1, &elementbuffer);
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
//    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
//
//}
//
