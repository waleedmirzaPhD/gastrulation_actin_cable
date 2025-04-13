#include <cmath> // For mathematical operations
#include <fstream> // For file operations
#include <iostream> // For input and output
#include <map> // For map container
#include <vector> // For vector container
#include <iomanip> // For formatted output
#include <mpi.h> // For MPI functions
#include <chrono> // For time operations
#include <thread> // For threading operations
#include <utility> // For utility functions, e.g., std::pair
#include <set>
// Trilinos headers for numerical computations
#include <Teuchos_RCP.hpp>
#include <random>  
// Headers from hiperlife for high-performance life science simulations
#include "hl_TypeDefs.h"
#include "hl_Geometry.h"
#include "hl_StructMeshGenerator.h"
#include "hl_DistributedMesh.h"
#include "hl_Remesher.h"
#include "hl_FillStructure.h"
#include "hl_DOFsHandler.h"
#include "hl_HiPerProblem.h"
#include "hl_ConsistencyCheck.h"
#include <hl_LoadBalance.h>
#include <hl_LinearSolver_Direct_Amesos2.h>
#include <hl_NonlinearSolver_NewtonRaphson.h>
#include <hl_MeshLoader.h>
#include <hl_ConfigFile.h>
#include <hl_LinearSolver_Direct_MUMPS.h>
#include "hl_BasicMeshGenerator.h"
#include "HexagonalGrid.h"
#include "Auxvertex_model.h"
#include <vector>
#include <set>
#include <algorithm>
#include <vector>
#include <algorithm>
#include <unordered_set> // Using unordered_set for efficient look-up
#include "hl_ParamStructure.h"
#include <cstdlib>   // For rand and srand
#include <ctime>     // For time
#include <iostream>  // For I/O operations





// Function to calculate the area of a hexagon on a 3D surface
double calculateHexagonArea3D(const std::vector<std::vector<double>>& nodes3D) 
{
    if (nodes3D.size() != 6) {
        throw std::invalid_argument("A hexagon must have exactly 6 vertices.");
    }

    // Compute the center point of the hexagon
    std::vector<double> center(3, 0.0);
    for (const auto& node : nodes3D) {
        center[0] += node[0];
        center[1] += node[1];
        center[2] += node[2];
    }
    center[0] /= 6.0;
    center[1] /= 6.0;
    center[2] /= 6.0;

    double totalArea = 0.0;

    // Loop through each pair of adjacent vertices to form triangles
    for (size_t i = 0; i < nodes3D.size(); i++) {
        const auto& p1 = nodes3D[i];
        const auto& p2 = nodes3D[(i + 1) % nodes3D.size()];

        // Create vectors from the center to the two vertices
        std::vector<double> v1 = {p1[0] - center[0], p1[1] - center[1], p1[2] - center[2]};
        std::vector<double> v2 = {p2[0] - center[0], p2[1] - center[1], p2[2] - center[2]};

        // Compute the cross product of v1 and v2
        std::vector<double> crossProduct = {
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        };

        // Compute the magnitude of the cross product
        double area = 0.5 * std::sqrt(
            crossProduct[0] * crossProduct[0] +
            crossProduct[1] * crossProduct[1] +
            crossProduct[2] * crossProduct[2]
        );

        // Accumulate the area of this triangle
        totalArea += area;
    }

    return totalArea;
}

// Structure to represent a 3D vertex
struct Vertex3D 
{
    double x, y, z;
    Vertex3D(double x, double y, double z = 0) : x(x), y(y), z(z) {}
};


// Structure to represent a Hexagon
struct Hexagon {
    std::vector<Vertex3D> boundaryNodes;
};

// Class to represent a grid with hexagons (you should define this according to your needs)
class Grid {
public:
    std::vector<int> getVertexIndicesForHexagon(int i) {
        // Replace with your actual implementation
        return {};
    }
};



// Function to write the hexagonal mesh to a VTK file
void writeHexagonalMeshToVTK(const std::string& filename, const std::vector<Hexagon>& hexagons) {
    std::ofstream vtkFile(filename);
    if (!vtkFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 3.0\nHexagonal mesh\nASCII\n";
    vtkFile << "DATASET UNSTRUCTURED_GRID\n";

    // Count total points
    size_t totalPoints = 0;
    for (const auto& hexagon : hexagons) {
        totalPoints += hexagon.boundaryNodes.size();
    }

    vtkFile << "POINTS " << totalPoints << " float\n";

    // Write points
    for (const auto& hexagon : hexagons) {
        for (const auto& node : hexagon.boundaryNodes) {
            vtkFile << node.x << " " << node.y << " " << node.z << "\n";
        }
    }

    // Write cells (hexagons)
    vtkFile << "CELLS " << hexagons.size() << " " << hexagons.size() * 7 << "\n"; // 7 = 1 (size) + 6 (hexagon vertices)
    size_t pointIndex = 0;
    for (const auto& hexagon : hexagons) {
        vtkFile << "6"; // A hexagon has 6 vertices
        for (size_t i = 0; i < hexagon.boundaryNodes.size(); ++i) {
            vtkFile << " " << pointIndex++;
        }
        vtkFile << "\n";
    }

    // Write cell types for hexagons
    vtkFile << "CELL_TYPES " << hexagons.size() << "\n";
    for (size_t i = 0; i < hexagons.size(); ++i) {
        vtkFile << "7\n"; // VTK_POLYGON (polygon with 6 sides)
    }

    vtkFile.close();
}

// Function to project 2D coordinates to spherical coordinates (latitude, longitude)
std::pair<double, double> projectToHemisphere(double x, double y, double radius) {
    double r = std::sqrt(x * x + y * y);
    double theta = std::atan2(y, x);
    double phi = M_PI / 2.5 * (r / radius); // Map distance to [0, pi/2]
    return {phi, theta};
}

// Function to convert spherical coordinates to 3D Cartesian coordinates
std::tuple<double, double, double> sphericalToCartesian(double phi, double theta, double radius) {
    double x = radius * std::sin(phi) * std::cos(theta);
    double y = radius * std::sin(phi) * std::sin(theta);
    double z = radius * std::cos(phi);
    return {x, y, z};
}

// Function to transform circular grid coordinates to hemispherical coordinates
std::vector<Vertex3D> transformToHemisphericalGrid(const std::vector<std::pair<double, double>>& circularGrid, double radius) {
    std::vector<Vertex3D> hemisphericalGrid;

    for (const auto& point : circularGrid) 
    {
        auto [phi, theta] = projectToHemisphere(point.first, point.second, radius);
        auto [x, y, z] = sphericalToCartesian(phi, theta, radius);
        hemisphericalGrid.emplace_back(x, y, z);
    }

    return hemisphericalGrid;
}



// Function to generate a random double in a given range
double getRandomNumber(double min, double max) {
    // Generate a random number between 0 and RAND_MAX
    double fraction = static_cast<double>(rand()) / RAND_MAX;
    return min + fraction * (max - min);
}

struct Point {
    double x;
    double y;
};


// Function to calculate the centroid of a polygon
Point calculateCentroid(const std::vector<std::pair<double, double>>& vertices) {
    double centroidX = 0, centroidY = 0;
    for (const auto& vertex : vertices) {
        centroidX += vertex.first;
        centroidY += vertex.second;
    }
    centroidX /= vertices.size();
    centroidY /= vertices.size();
    return {centroidX, centroidY};
}

// Function to check if a set of vertices forms a convex polygon
bool isConvex(const std::vector<std::pair<double, double>>& vertices) {
    int n = vertices.size();
    bool isConvex = true;

    for (int i = 0; i < n; ++i) {
        auto& A = vertices[i];
        auto& B = vertices[(i + 1) % n];
        auto& C = vertices[(i + 2) % n];

        double crossProduct = (B.first - A.first) * (C.second - A.second) - (B.second - A.second) * (C.first - A.first);
        if (crossProduct < 0) {
            isConvex = false;
            break;
        }
    }

    return isConvex;
}





//     return transformedVertices;
// }
// Function to transform a hexagon into an irregular but convex shape with strong noise along the centroid-vertex axis
std::vector<std::pair<double, double>> transformHexagonToIrregularConvex(const std::vector<std::pair<double, double>>& vertices, double perturbationRange) {
    std::vector<std::pair<double, double>> transformedVertices = vertices;
    auto centroid = calculateCentroid(vertices); // Get the centroid of the hexagon

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0, perturbationRange);

    // Apply noise along the axis that connects the centroid to each vertex
    for (auto& vertex : transformedVertices) {
        double dx = vertex.first - centroid.x;
        double dy = vertex.second - centroid.y;
        double distance = std::sqrt(dx * dx + dy * dy);

        // Calculate the unit direction vector from the centroid to the vertex
        double unitDirX = dx / distance;
        double unitDirY = dy / distance;

        // Generate a random perturbation value along the centroid-vertex direction
        double perturbDistance = 1.5*distribution(generator); // This adds strong noise along the direction

        // Apply the perturbation along the centroid-vertex axis
        vertex.first += perturbDistance * unitDirX;
        vertex.second += perturbDistance * unitDirY;
    }

    // Ensure convexity
    if (!isConvex(transformedVertices)) {
        // If the shape becomes non-convex, reduce perturbations slightly and reapply
        for (auto& vertex : transformedVertices) {
            double dx = vertex.first - centroid.x;
            double dy = vertex.second - centroid.y;
            double distance = std::sqrt(dx * dx + dy * dy);

            // Calculate the unit direction vector from the centroid to the vertex
            double unitDirX = dx / distance;
            double unitDirY = dy / distance;

            // Apply smaller perturbations along the centroid-vertex axis
            double perturbDistance = distribution(generator); // Reduce the noise by half

            // Apply the perturbation along the centroid-vertex axis
            vertex.first += perturbDistance * unitDirX;
            vertex.second += perturbDistance * unitDirY;
        }
    }

    return transformedVertices;
}


// Function to change area of a shape based on perturbation range, ensuring convexity
std::vector<std::pair<double, double>> changeAreaBasedOnPerturbation(const std::vector<std::pair<double, double>>& vertices, double areaFactor) {
    std::vector<std::pair<double, double>> scaledVertices = vertices;
    Point centroid = calculateCentroid(vertices); // Get the centroid of the shape

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.8 * areaFactor, 1.2 * areaFactor); // Random area factor around the given range

    bool validTransformation = false;
    while (!validTransformation) {
        scaledVertices = vertices; // Reset to original vertices
        double randomAreaFactor = distribution(generator); // Get a random scaling factor

        // Scale each vertex away from or towards the centroid based on the random area factor
        for (auto& vertex : scaledVertices) {
            double dx = vertex.first - centroid.x;
            double dy = vertex.second - centroid.y;

            // Apply the scaling factor to increase or decrease the area
            vertex.first = centroid.x + dx * randomAreaFactor;
            vertex.second = centroid.y + dy * randomAreaFactor;
        }

        // Ensure the shape remains convex after transformation
        validTransformation = true ; //isConvex(scaledVertices);
    }

    return scaledVertices;
}
