#ifndef MISCELLANEOUS_ROUTINES_H
#define MISCELLANEOUS_ROUTINES_H

#include <vector>
#include <tuple>
#include <utility>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <stdexcept>

// Structures
struct Vertex3D {
    double x, y, z;
    Vertex3D(double x, double y, double z = 0) : x(x), y(y), z(z) {}
};

struct Point {
    double x, y;
};

struct Hexagon {
    std::vector<Vertex3D> boundaryNodes;
};

// Function declarations
double calculateHexagonArea3D(const std::vector<std::vector<double>>& nodes3D);
void writeHexagonalMeshToVTK(const std::string& filename, const std::vector<Hexagon>& hexagons);
std::pair<double, double> projectToHemisphere(double x, double y, double radius);
std::tuple<double, double, double> sphericalToCartesian(double phi, double theta, double radius);
std::vector<Vertex3D> transformToHemisphericalGrid(const std::vector<std::pair<double, double>>& circularGrid, double radius);
Point calculateCentroid(const std::vector<std::pair<double, double>>& vertices);
bool isConvex(const std::vector<std::pair<double, double>>& vertices);
std::vector<std::pair<double, double>> transformHexagonToIrregularConvex(const std::vector<std::pair<double, double>>& vertices, double perturbationRange);
std::vector<std::pair<double, double>> changeAreaBasedOnPerturbation(const std::vector<std::pair<double, double>>& vertices, double areaFactor);
double getRandomNumber(double min, double max);

#endif // MISCELLANEOUS_ROUTINES_H
