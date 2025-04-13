
#include <cmath>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include "HexagonalGrid.h"
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <cmath>

void HexagonalGrid::add_vertex(double x, double y) {
    Vertex v(x, y);
    if (vertexMap.count(v) == 0) {
        vertexMap[v] = vertices.size();
        vertices.emplace_back(x, y);
    }
}

int HexagonalGrid::add_vertex_(double x, double y) {
    Vertex v(x, y);
    // Check if the vertex already exists in the map
    auto it = vertexMap.find(v);
    if (it == vertexMap.end()) {
        // If it doesn't exist, add it both to the map and the vertices vector
        int newIndex = vertices.size();
        vertices.emplace_back(x, y);  // Add the new vertex to the list of vertices
        vertexMap[v] = newIndex;      // Map the vertex to its index
        return newIndex;              // Return the new index
    } else {
        // If it already exists, return the existing index
        return it->second;
    }
}



void HexagonalGrid::add_hexagon(double center_x, double center_y, double radius) 
{
    int centerIndex = add_center_vertex(center_x, center_y);
    std::vector<int> vertexIndices;
    int hexagonId = hexagonIdCounter++; // Assign and then increment the hexagon ID

    for (int i = 0; i < 6; ++i) {
        double angle_rad = M_PI / 180.0 * (60 * i);
        double x = center_x + radius * cos(angle_rad);
        double y = center_y + radius * sin(angle_rad);
        add_vertex(x, y);
        vertexIndices.push_back(vertexMap[Vertex(x, y)]);
    }

    // Map hexagon ID to its vertices
    hexagonToVertices[hexagonId] = vertexIndices;
    // Ensure the barycenter index is associated with the hexagon ID
    hexagonIdToBarycenter[hexagonId] = centerIndex;     


    // Decompose hexagon into 6 triangles, including hexagon ID
    for (int i = 0; i < 6; ++i) 
    {
        bool flag = false;
        int nextIndex = (i + 1) % 6;
        if (i==0)
            flag=true;
        else
            flag=false;
        triangles.emplace_back(centerIndex, vertexIndices[i], vertexIndices[nextIndex], hexagonId,flag);
    }

}

double HexagonalGrid::generateRandomOffset(double range) 
{
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution(-range, range);
    return distribution(generator);
}

int HexagonalGrid::add_center_vertex(double center_x, double center_y) {
    // This method adds the center vertex and returns its index
    add_vertex(center_x, center_y);
    return vertexMap[Vertex(center_x, center_y)];
}

// Function to generate a random double in a given range
double _getRandomNumber(double min, double max) {
    // Generate a random number between 0 and RAND_MAX
    double fraction = static_cast<double>(rand()) / RAND_MAX;
    return min + fraction * (max - min);
}

// Function to generate a hexagonal grid
void HexagonalGrid::generate_hexagonal_grid(int grid_width, int grid_height, double radius) {
    double vert_spacing = std::sqrt(3) * radius;
    double horiz_spacing = 1.5 * radius;

    for (int row = 0; row < grid_height; ++row) {
        for (int col = 0; col < grid_width; ++col) {
            double center_x = col * horiz_spacing;
            double center_y = row * vert_spacing;
            if (col % 2 == 1) {
                center_y += vert_spacing / 2;
            }
            add_hexagon(center_x, center_y, radius);
        }
    }
    identifyBoundaryVertices();
    setFlagsForBoundaryHexagons();
}

void HexagonalGrid::write_to_vtk_(const std::string& filename) 
{
    std::ofstream vtkFile(filename);
    vtkFile << "# vtk DataFile Version 3.0\nHexagonal grid decomposed into triangles\nASCII\n";
    vtkFile << "DATASET UNSTRUCTURED_GRID\n";
    vtkFile << "POINTS " << vertices.size() << " float\n";
    for (const auto& vertex : vertices) {
        vtkFile << vertex.first << " " << vertex.second << " 0\n";
    }
    vtkFile << "CELLS " << triangles.size() << " " << triangles.size() * 4 << "\n";
    for (const auto& triangle : triangles) {
        vtkFile << "3 " << triangle.v1 << " " << triangle.v2 << " " << triangle.v3 << "\n";
    }
    vtkFile << "CELL_TYPES " << triangles.size() << "\n";
    for (size_t i = 0; i < triangles.size(); ++i) {
        vtkFile << "5\n"; // VTK_TRIANGLE
    }
    vtkFile.close();
}

void HexagonalGrid::generate_circular_hexagonal_grid(double radius, double hex_radius) {
    double vert_spacing = std::sqrt(3) * hex_radius;
    double horiz_spacing = 1.5 * hex_radius;

    int max_distance = std::ceil(radius / hex_radius);

    for (int row = -max_distance; row <= max_distance; ++row) {
        for (int col = -max_distance; col <= max_distance; ++col) {
            double center_x = col * horiz_spacing;
            double center_y = row * vert_spacing;
            if (col % 2 != 0) {
                center_y += vert_spacing / 2;
            }

            // Check if the center point is within the circular boundary
            if (std::sqrt(center_x * center_x + center_y * center_y) <= radius) {
                add_hexagon(center_x, center_y, hex_radius);
            }
        }
    }
    identifyBoundaryVertices();
    setFlagsForBoundaryHexagons();
}


void HexagonalGrid::write_to_vtk(const std::string& filename) 
{
    std::ofstream vtkFile(filename);
    if (!vtkFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    
    vtkFile << "# vtk DataFile Version 3.0\nHexagonal grid decomposed into triangles\nASCII\n";
    vtkFile << "DATASET UNSTRUCTURED_GRID\n";
    vtkFile << "POINTS " << vertices.size() << " float\n";
    for (const auto& vertex : vertices) {
        vtkFile << vertex.first << " " << vertex.second << " 0\n";
    }
    
    vtkFile << "CELLS " << triangles.size() << " " << triangles.size() * 4 << "\n";
    for (const auto& triangle : triangles) {
        vtkFile << "3 " << triangle.v1 << " " << triangle.v2 << " " << triangle.v3 << "\n";
    }
    
    vtkFile << "CELL_TYPES " << triangles.size() << "\n";
    for (size_t i = 0; i < triangles.size(); ++i) {
        vtkFile << "5\n"; // VTK_TRIANGLE
    }

    // Writing cell data: hexagon IDs for each triangle
    vtkFile << "CELL_DATA " << triangles.size() << "\n";
    vtkFile << "SCALARS hexagon_id int 1\n"; // Specify the data name, type, and number of components
    vtkFile << "LOOKUP_TABLE default\n"; // Use a default lookup table
    for (const auto& triangle : triangles) {
        vtkFile << triangle.hexagonId << "\n"; // Assume Triangle has a member hexagonId
    }

    vtkFile.close();
}

int HexagonalGrid::getHexagonIdForTriangle(int triangleId) 
{
    if (triangleId >= 0 && triangleId < triangles.size()) {
        return triangles[triangleId].hexagonId;
    } else {
        return -1; // Invalid triangleId
    }
}

int HexagonalGrid::getFlagForTriangle(int triangleId) 
{
    if (triangleId >= 0 && triangleId < triangles.size()) {
        return triangles[triangleId].flag_first_triangle;
    } else {
        return -1; // Invalid triangleId
    }
}

int HexagonalGrid::getBarycenterNodeIndexForHexagon(int hexagonId) 
{
    return hexagonIdToBarycenter[hexagonId];
}

std::vector<int> HexagonalGrid::getVertexIndicesForHexagon(int hexagonId) 
{
    return hexagonToVertices[hexagonId];

}

double HexagonalGrid::calculateHexagonPerimeter(int hexagonId) {
    const auto& vertexIndices = hexagonToVertices[hexagonId];
    double perimeter = 0.0;
    for (size_t i = 0; i < vertexIndices.size(); ++i) {
        size_t j = (i + 1) % vertexIndices.size();
        const auto& vi = vertices[vertexIndices[i]];
        const auto& vj = vertices[vertexIndices[j]];
        double dx = vi.first - vj.first;
        double dy = vi.second - vj.second;
        perimeter += std::sqrt(dx * dx + dy * dy);

    }
    return perimeter;
}

double HexagonalGrid::calculateHexagonArea(int hexagonId) {
    const auto& vertexIndices = hexagonToVertices[hexagonId];
    double area = 0.0;
    for (size_t i = 0; i < vertexIndices.size(); ++i) {
        size_t j = (i + 1) % vertexIndices.size();
        const auto& vi = vertices[vertexIndices[i]];
        const auto& vj = vertices[vertexIndices[j]];
        area += vi.first * vj.second;
        area -= vj.first * vi.second;
    }
    return std::abs(area) / 2.0;
}

void HexagonalGrid::identifyBoundaryVertices() {
    // Map to count how many hexagons share each vertex
    std::map<int, int> vertexShareCount;

    // Count the occurrences of each vertex in all hexagons
    for (const auto& hexagonVertices : hexagonToVertices) 
    {
        for (int vertexId : hexagonVertices.second) 
        {
            vertexShareCount[vertexId]++;
        }
    }

    // Identify vertices that are shared by less than 3 hexagons as boundary vertices
    for (const auto& entry : vertexShareCount) 
    {
        if (entry.second < 3) 
        { // This threshold might need adjustment based on your grid's specifics
            boundaryVertexIds.push_back(entry.first);
        }
    }

}

std::vector<int> HexagonalGrid::getBoundaryVertexIds()  {
    return boundaryVertexIds;
}


std::vector<int> HexagonalGrid::pickSpacedInnerHexagons(double minDistance) 
{
        identifyBoundaryVertices(); // Ensure boundary vertices are updated

        // Filter out boundary hexagons
        std::vector<int> innerHexagons;
        for (const auto& hexagon : hexagonToVertices) {
            int hexId = hexagon.first;
            bool isInner = true;
            for (int vId : hexagon.second) {
                if (std::find(boundaryVertexIds.begin(), boundaryVertexIds.end(), vId) != boundaryVertexIds.end()) {
                    isInner = false;
                    break;
                }
            }
            if (isInner) {
                innerHexagons.push_back(hexId);
            }
        }

        // Shuffle to randomize initial picks
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(innerHexagons.begin(), innerHexagons.end(), g);

        std::vector<int> selectedHexagons;
        std::unordered_set<int> selectedIndices;

        // Attempt to pick 10 spaced hexagons
        for (int currentHexId : innerHexagons) {
            bool isFarEnough = true;
            for (int selectedHexId : selectedHexagons) {
                if (distanceBetweenHexagons(currentHexId, selectedHexId) < minDistance) {
                    isFarEnough = false;
                    break;
                }
            }
            if (isFarEnough) {
                selectedHexagons.push_back(currentHexId);
                if (selectedHexagons.size() == 10) break;
            }
        }

        if (selectedHexagons.size() < 10) {
            std::cerr << "Not enough spaced inner hexagons to pick 10." << std::endl;
        }

        return selectedHexagons;
}

double HexagonalGrid::distanceBetweenHexagons(int hexId1, int hexId2) 
{
        auto& center1 = vertices[hexagonIdToBarycenter[hexId1]];
        auto& center2 = vertices[hexagonIdToBarycenter[hexId2]];
        return sqrt(pow(center1.first - center2.first, 2) + pow(center1.second - center2.second, 2));
}



void HexagonalGrid::generate_circular_hexagonal_grid_with_deleted_cell(double radius, double hex_radius, int delete_row, int delete_col) {
    double vert_spacing = std::sqrt(3) * hex_radius;
    double horiz_spacing = 1.5 * hex_radius;

    int max_distance = std::ceil(radius / hex_radius);

    for (int row = -max_distance; row <= max_distance; ++row) {
        for (int col = -max_distance; col <= max_distance; ++col) {
            // Skip the specified cell to delete
            if (row == delete_row && col == delete_col) {
                continue;
            }

            double center_x = col * horiz_spacing;
            double center_y = row * vert_spacing;
            if (col % 2 != 0) {
                center_y += vert_spacing / 2;
            }

            // Check if the center point is within the circular boundary
            if (std::sqrt(center_x * center_x + center_y * center_y) <= radius) {
                add_hexagon(center_x, center_y, hex_radius);
            }
        }
    }
    identifyBoundaryVertices();
}

void HexagonalGrid::generate_hexagonal_grid_with_hole(int grid_width, int grid_height, double radius) {
    double vert_spacing = std::sqrt(3) * radius;
    double horiz_spacing = 1.5 * radius;
    int centerRow = grid_height / 2;
    int centerCol = grid_width / 2;

    for (int row = 0; row < grid_height; ++row) {
        for (int col = 0; col < grid_width; ++col) {
            // Check if the current cell is the center cell; skip adding a hexagon here
            if (row == centerRow && col == centerCol)
            {
                continue;
            }
            double center_x = col * horiz_spacing;
            double center_y = row * vert_spacing;
            // Adjust vertical position for every second column
            if (col % 2 == 1) {
                center_y += vert_spacing / 2;
            }
            // Add a hexagon at the calculated center position
            add_hexagon(center_x, center_y, radius);
        }
    }
    // Identify boundary vertices after generating all hexagons
    identifyBoundaryVertices();

}

    // Calculate and update the grid dimensions for a new radius to maintain the same area
void HexagonalGrid::calculateNewDimensions(int original_nx, int original_ny, double original_r, double new_r, int &new_nx, int &new_ny) {
        // Calculate the area of a single hexagon with the original radius
        double original_area = (3 * sqrt(3) / 2) * original_r * original_r;

        // Calculate the total area of the original grid
        double total_area = original_nx * original_ny * original_area;

        // Calculate the area of a single hexagon with the new radius
        double new_area = (3 * sqrt(3) / 2) * new_r * new_r;

        // Calculate the total number of hexagons needed to maintain the same area
        int total_hexagons = static_cast<int>(round(total_area / new_area));

        // Assume the new grid dimensions should be as square-like as possible
        new_nx = static_cast<int>(round(sqrt(total_hexagons)));
        new_ny = (total_hexagons + new_nx - 1) / new_nx; // Adjust new_ny based on new_nx

        std::cout << "New dimensions: " << new_nx << "x" << new_ny << " for radius " << new_r << std::endl;
    }

void HexagonalGrid::setFlagsForBoundaryHexagons()
{
    // Identify the boundary vertices if they are not already identified
    identifyBoundaryVertices();
    
    // Iterate over all hexagons
    for (const auto& hexagonEntry : hexagonToVertices)
    {
        int hexagonId = hexagonEntry.first;
        const std::vector<int>& hexagonVertices = hexagonEntry.second;

        // Check if the hexagon is a boundary hexagon
        bool isBoundaryHexagon = false;
        for (const int& vertex : hexagonVertices)
        {
            if (std::find(boundaryVertexIds.begin(), boundaryVertexIds.end(), vertex) != boundaryVertexIds.end())
            {
                isBoundaryHexagon = true;
                break;
            }
        }

        // If it's a boundary hexagon, proceed to update flags for its triangles
        if (isBoundaryHexagon)
        {
            bool flagSet = false;

            // Iterate over triangles belonging to this hexagon
            for (int triangleIndex = 0; triangleIndex < triangles.size(); ++triangleIndex)
            {
                if (triangles[triangleIndex].hexagonId == hexagonId)
                {
                    const Triangle& triangle = triangles[triangleIndex];

                    // Check if the triangle has any boundary vertices
                    bool hasBoundaryVertex = false;
                    for (int vertexId : {triangle.v1, triangle.v2, triangle.v3})
                    {
                        if (std::find(boundaryVertexIds.begin(), boundaryVertexIds.end(), vertexId) != boundaryVertexIds.end())
                        {
                            hasBoundaryVertex = true;
                            break;
                        }
                    }

                    // If the triangle has no boundary vertices, set its flag to true
                    if (!hasBoundaryVertex && !flagSet)
                    {
                        triangles[triangleIndex].flag_first_triangle = true;
                        flagSet = true;  // Ensure only one flag is set per hexagon
                    }
                    else
                    {
                        triangles[triangleIndex].flag_first_triangle = false;  // Ensure others are false
                    }
                }
            }
        }
    }

    // Set all remaining triangle flags to false
    for (auto& triangle : triangles)
    {
        if (!triangle.flag_first_triangle) // Only set if not already true
        {
            triangle.flag_first_triangle = false;
        }
    }



}


void HexagonalGrid::generate_circular_mesh_with_boundary_nodes(double boundary_radius, double hex_radius) 
{
    double vert_spacing = std::sqrt(3) * hex_radius;
    double horiz_spacing = 1.5 * hex_radius;

    // Calculate maximum distance for the mesh grid
    int max_distance = std::ceil(boundary_radius / hex_radius);

    // Generate hexagons within the circular boundary
    for (int row = -max_distance; row <= max_distance; ++row) {
        for (int col = -max_distance; col <= max_distance; ++col) {
            double center_x = col * horiz_spacing;
            double center_y = row * vert_spacing;
            if (col % 2 != 0) {
                center_y += vert_spacing / 2;
            }

            // Check if the center point is within the circular boundary
            if (std::sqrt(center_x * center_x + center_y * center_y) <= boundary_radius) {
                add_hexagon(center_x, center_y, hex_radius);
            }
        }
    }

    // Identify boundary vertices after generating all hexagons
    identifyBoundaryVertices();

    // Adjust boundary vertices to be at the same radius
    for (int vertexId : boundaryVertexIds) {
        auto& vertex = vertices[vertexId];
        double current_radius = std::sqrt(vertex.first * vertex.first + vertex.second * vertex.second);
        if (current_radius != 0) {  // Avoid division by zero for the origin
            double scaling_factor = boundary_radius / current_radius;
            vertex.first *= scaling_factor;
            vertex.second *= scaling_factor;
        }
    }

    // Update boundary hexagons and their flags
    setFlagsForBoundaryHexagons();
}




void HexagonalGrid::generate_circular_hexagonal_grid_with_deleted_cell2(double boundary_radius, double hex_radius, int delete_row, int delete_col) 
{
    double vert_spacing = std::sqrt(3) * hex_radius;
    double horiz_spacing = 1.5 * hex_radius;

    // Calculate maximum distance for the mesh grid
    int max_distance = std::ceil(boundary_radius / hex_radius);

    // Generate hexagons within the circular boundary
    for (int row = -max_distance; row <= max_distance; ++row) {
        for (int col = -max_distance; col <= max_distance; ++col) {
            if (row == delete_row && col == delete_col) {
                continue;
            }
            double center_x = col * horiz_spacing;
            double center_y = row * vert_spacing;
            if (col % 2 != 0) {
                center_y += vert_spacing / 2;
            }

            // Check if the center point is within the circular boundary
            if (std::sqrt(center_x * center_x + center_y * center_y) <= boundary_radius) {
                add_hexagon(center_x, center_y, hex_radius);
            }
        }
    }

    // Identify boundary vertices after generating all hexagons
    identifyBoundaryVertices();

    // Adjust boundary vertices to be at the same radius
    for (int vertexId : boundaryVertexIds) {
        auto& vertex = vertices[vertexId];
        double current_radius = std::sqrt(vertex.first * vertex.first + vertex.second * vertex.second);
        if (current_radius != 0 and current_radius>0.95*boundary_radius ) {  // Avoid division by zero for the origin
            double scaling_factor = boundary_radius / current_radius;
            vertex.first *= scaling_factor;
            vertex.second *= scaling_factor;
        }
    }

    // Update boundary hexagons and their flags
    setFlagsForBoundaryHexagons();
}



void HexagonalGrid::generate_circular_hexagonal_grid_with_deleted_closest_cell(double boundary_radius, double hex_radius, double target_x, double target_y) 
{
    double vert_spacing = std::sqrt(3) * hex_radius;
    double horiz_spacing = 1.5 * hex_radius;

    // Calculate maximum distance for the mesh grid
    int max_distance = std::ceil(boundary_radius / hex_radius);

    // Variables to keep track of the closest cell
    int closest_row = 0;
    int closest_col = 0;
    double min_distance = std::numeric_limits<double>::max();

    // Generate hexagons within the circular boundary and find the closest cell
    for (int row = -max_distance; row <= max_distance; ++row) {
        for (int col = -max_distance; col <= max_distance; ++col) {
            double center_x = col * horiz_spacing;
            double center_y = row * vert_spacing;
            if (col % 2 != 0) {
                center_y += vert_spacing / 2;
            }

            // Check if the center point is within the circular boundary
            if (std::sqrt(center_x * center_x + center_y * center_y) <= boundary_radius) {
                double distance = std::sqrt((center_x - target_x) * (center_x - target_x) + (center_y - target_y) * (center_y - target_y));
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_row = row;
                    closest_col = col;
                }
            }
        }
    }

    // Generate hexagons within the circular boundary, skipping the closest cell
    for (int row = -max_distance; row <= max_distance; ++row) {
        for (int col = -max_distance; col <= max_distance; ++col) {
            if (row == closest_row && col == closest_col) {
                continue;
            }
            double center_x = col * horiz_spacing;
            double center_y = row * vert_spacing;
            if (col % 2 != 0) {
                center_y += vert_spacing / 2;
            }

            // Check if the center point is within the circular boundary
            if (std::sqrt(center_x * center_x + center_y * center_y) <= boundary_radius) {
                add_hexagon(center_x, center_y, hex_radius);
            }
        }
    }

    // Identify boundary vertices after generating all hexagons
    identifyBoundaryVertices();

    // Adjust boundary vertices to be at the same radius
    for (int vertexId : boundaryVertexIds) {
        auto& vertex = vertices[vertexId];
        double current_radius = std::sqrt(vertex.first * vertex.first + vertex.second * vertex.second);
        if (current_radius != 0 && current_radius > 0.75 * boundary_radius) {  // Avoid division by zero for the origin
            double scaling_factor = boundary_radius / current_radius;
            vertex.first *= scaling_factor;
            vertex.second *= scaling_factor;
        }
    }

    // Update boundary hexagons and their flags
    setFlagsForBoundaryHexagons();
}
