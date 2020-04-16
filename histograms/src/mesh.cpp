#include "mesh.h"

#include <fstream>

namespace histograms
{
    Mesh::Mesh(const std::vector<glm::vec3> &vertices, const std::vector<glm::uvec3>& faces) :
                  vertices(vertices),
                  faces(faces)
    {
        float x_low, x_up, y_low, y_up, z_low, z_up;
        if (!vertices.empty())
        {
            x_low = vertices[0].x;
            x_up = vertices[0].x;
            y_low = vertices[0].y;
            y_up = vertices[0].y;
            z_low = vertices[0].z;
            z_up = vertices[0].z;
            for (size_t i = 0; i < vertices.size(); ++i)
            {
                const glm::vec3& vertex = vertices[i];
                if (vertex.x < x_low)
                {
                    x_low = vertex.x;
                }
                if (vertex.x > x_up)
                {
                    x_up = vertex.x;
                }
                if (vertex.y < y_low)
                {
                    y_low = vertex.y;
                }
                if (vertex.y > y_up)
                {
                    y_up = vertex.y;
                }
                if (vertex.z < z_low)
                {
                    z_low = vertex.z;
                }
                if (vertex.z > z_up)
                {
                    z_up = vertex.z;
                }
            }
            float x_dim = x_up - x_low;
            float y_dim = y_up - y_low;
            float z_dim = z_up - z_low;
            center = glm::vec3((x_low + x_up) / 2, (y_low + y_up) / 2, (z_low + z_up) / 2);
            bb_diameter = sqrt(x_dim * x_dim + y_dim * y_dim + z_dim * z_dim);
        }

    }

    Mesh::Mesh()
    {
        vertices = std::vector<glm::vec3>();
        faces = std::vector<glm::uvec3>();
        bb_diameter = 0;
    }

    const std::vector<glm::vec3>& Mesh::getVertices() const
    {
        return vertices;
    }

    const std::vector<glm::uvec3>& Mesh::getFaces() const
    {
        return faces;
    }

    float Mesh::getBBDiameter() const
    {
        return bb_diameter;
    }

    void Mesh::exportWithPose(const std::string &file_name, const glm::mat4 &pose) const
    {
        std::ofstream output_file(file_name);
        for (size_t i = 0; i < vertices.size(); ++i)
        {
            glm::vec4 vertex = glm::vec4(vertices[i], 1);
            glm::vec4 transformed_vertex = pose * vertex;
            output_file << "v " << transformed_vertex[0] << ' ' <<
                                   transformed_vertex[1] << ' ' <<
                                   transformed_vertex[2] << ' ' <<
                                   std::endl;

        }
        for (size_t i = 0; i < faces.size(); ++i)
        {
            output_file << " f " << faces[i][0] + 1 << ' ' << faces[i][1] + 1 << ' ' << faces[i][2] + 1 << std::endl;
        }
        output_file.close();
    }

    const glm::vec3 & Mesh::getCenter() const
    {
        return center;
    }

    void Mesh::fitDiameterToFive()
    {
        if (bb_diameter == 0)
        {
            return;
        }
        float scale_factor = bb_diameter / 5;
        for (int i = 0; i < vertices.size(); ++i)
        {
            vertices[i] /= scale_factor;
        }
        center /= scale_factor;
        bb_diameter = 5;
    }


}
