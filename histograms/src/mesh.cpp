#include "mesh.h"

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


}
