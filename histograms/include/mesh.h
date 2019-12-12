#ifndef HISTOGRAMS_MESH_H
#define HISTOGRAMS_MESH_H

#include <vector>
#include <opencv2/core/mat.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>


namespace histograms
{
    class Mesh
    {
        std::vector<glm::vec3> vertices;
        std::vector<glm::uvec3> faces;

        float bb_diameter;

    public:
        Mesh(const std::vector<glm::vec3>& vertices, const std::vector<glm::uvec3>& faces);

        Mesh();

        const std::vector<glm::vec3>& getVertices() const;

        const std::vector<glm::uvec3>& getFaces() const;

        float getBBDiameter() const;

        void fitDiameterToOne();

        void exportWithPose(const std::string &file_name, const glm::mat4 &pose) const;

    };
}
#endif //HISTOGRAMS_MESH_H
