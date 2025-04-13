#ifndef HEXAGONALGRID_H
#define HEXAGONALGRID_H

#include <vector>
#include <map>
#include <utility> // For std::pair
#include <string>
#include <fstream> // For std::ofstream
#include <cmath> // For std::pow, std::round, std::cos, std::sin

struct Vertex {
    double x, y,z;

    Vertex(double x, double y) : x(round(x, 4)), y(round(y, 4)) {}

    static double round(double value, int decimalPlaces) {
        double factor = std::pow(10.0, decimalPlaces);
        return std::round(value * factor) / factor;
    }

    bool operator<(const Vertex& other) const {
        return x < other.x || (!(other.x < x) && y < other.y);
    }
};

struct Triangle {
    int v1, v2, v3;
    int hexagonId; // ID of the parent hexagon
    int flag_first_triangle;
    Triangle(int v1, int v2, int v3, int hexId, bool flag=false) : v1(v1), v2(v2), v3(v3), hexagonId(hexId),flag_first_triangle(flag) {}
};




class HexagonalGrid 
{
    public:
        std::vector<std::pair<double, double>> vertices;
        std::map<Vertex, int> vertexMap;
        std::vector<Triangle> triangles;
        std::map<int, int> hexagonIdToBarycenter;
        std::map<int, std::vector<int>> hexagonToVertices;
        std::vector<int> boundaryVertexIds;
        int hexagonIdCounter = 0;

        void add_vertex(double x, double y);
        int add_vertex_(double x, double y);
        void add_hexagon(double center_x, double center_y, double radius);
        int add_center_vertex(double center_x, double center_y);
        void generate_hexagonal_grid(int grid_width, int grid_height, double radius);
        void write_to_vtk_(const std::string& filename);
        void write_to_vtk(const std::string& filename);
        int getHexagonIdForTriangle(int triangleId);
        void setFlagsForBoundaryHexagons();
        void generate_circular_mesh_with_boundary_nodes(double boundary_radius, double hex_radius); 
        void generate_circular_hexagonal_grid_with_deleted_cell2(double radius, double hex_radius, int delete_row, int delete_col);
        int getFlagForTriangle(int triangleId);
        int getBarycenterNodeIndexForHexagon(int hexagonId);
        std::vector<int> getVertexIndicesForHexagon(int hexagonId);
        double  calculateHexagonPerimeter(int hexagonId);
        double  calculateHexagonArea(int hexagonId);
        void identifyBoundaryVertices();
        std::vector<int> getBoundaryVertexIds();
        std::vector<int> pickSpacedInnerHexagons(double minDistance) ;
        double distanceBetweenHexagons(int hexId1, int hexId2);
        void generate_hexagonal_grid_with_hole(int grid_width, int grid_height, double radius);
        void calculateNewDimensions(int original_nx, int original_ny, double original_r, double new_r, int &new_nx, int &new_ny);
        void generate_circular_hexagonal_grid_with_deleted_closest_cell(double boundary_radius, double hex_radius, double target_x, double target_y) ;
        void mapToHalfSphere(double radius);
        double generateRandomOffset(double range) ;
        void generate_circular_hexagonal_grid(double radius, double hex_radius);
        void generate_circular_hexagonal_grid_with_deleted_cell(double radius, double hex_radius, int delete_row, int delete_col);
};

#endif // HEXAGONALGRID_H
    // grid.generate_circular_hexagonal_grid_with_deleted_cell(15,2,2,2);
    // //grid.generate_circular_hexagonal_grxid_with_deleted_cell(16,1.52,2,2); --> thsi is for column 1 
