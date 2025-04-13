#ifndef MISCELLANEOUS2_H
#define MISCELLANEOUS2_H

#include <array>
#include <cmath>

#include "Amesos.h"
/// hiperlife headers
#include "hl_FillStructure.h"
#include "hl_Geometry.h"
#include "hl_SurfLagrParam.h"
#include "hl_Tensor.h"
#include "hl_LinearSolver_Direct_Amesos2.h"
#include <hl_LinearSolver_Direct_MUMPS.h>
#include <fstream>

/// Header to auxiliary functions
#include "Auxvertex_model.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <iomanip>
// Including necessary C++ and MPI headers
#include <iostream>
#include <mpi.h>

// Including necessary Trilinos headers for numerical computations
#include <Teuchos_RCP.hpp>
#include <math.h>

// Including necessary headers from hiperlife for high-performance life science simulations
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
#include <chrono>
#include <thread>
#include "hl_BasicMeshGenerator.h"
#include <utility> // For std::pair


/// hiperlife headers
#include "hl_Core.h"
#include "hl_Parser.h"
#include "hl_TypeDefs.h"
#include "hl_GlobalBasisFunctions.h"
#include "hl_StructMeshGenerator.h"
#include "hl_DistributedMesh.h"
#include "hl_FillStructure.h"
#include "hl_DOFsHandler.h"
#include "hl_HiPerProblem.h"
#include "hl_LinearSolver_Direct_MUMPS.h"
#include "hl_NonlinearSolver_NewtonRaphson.h"
#include <ad.h>
#include "hl_Remesher.h"
#include "hl_MeshLoader.h"
#include "hl_LinearSolver_Iterative_AztecOO.h"


#include <cmath>
#include <iostream>


using namespace hiperlife::Tensor;

namespace miscellaneous2
{

    // Function to compute the cross product of two vectors
    template <typename Type>
    hiperlife::Tensor::tensor<Type, 1> crossProduct(hiperlife::Tensor::tensor<Type, 1>& vec1, hiperlife::Tensor::tensor<Type, 1>& vec2) 
    {
        return hiperlife::Tensor::tensor<Type, 1>({
            vec1(1) * vec2(2) - vec1(2) * vec2(1),
            vec1(2) * vec2(0) - vec1(0) * vec2(2),
            vec1(0) * vec2(1) - vec1(1) * vec2(0)
        });
    }


    // Function to compute the magnitude of a vector
    template <typename Type>
    Type magnitude( tensor<Type, 1>& vec) 
    {
        using std::sqrt; 
        return sqrt(vec(0) * vec(0) + vec(1) * vec(1) + vec(2) * vec(2));
    }


    template <typename Type>
    Type distance3D( tensor<Type, 1> p1,  tensor<Type, 1> p2) 
    {
        using std::sqrt;
        return sqrt(pow(p2(0) - p1(0), 2) + pow(p2(1) - p1(1), 2) + pow(p2(2) - p1(2), 2));
    }


    // Function to calculate the perimeter of a 3D hexagon
    template <typename Type>
    Type calculatePerimeter( tensor<Type, 2>& hexagon) 
    {
        Type perimeter = 0.0;
        int n = 6;

        for (int i = 0; i < n; ++i) {
            // Compute distance between consecutive vertices
            tensor<Type, 1> p1(3); 
            p1(0)= hexagon(i, 0);
            p1(1)= hexagon(i, 1);
            p1(2)= hexagon(i, 2);

            tensor<Type, 1> p2(3); 
            p2(0)= hexagon((i + 1) % n, 0);
            p2(1)= hexagon((i + 1) % n, 1);
            p2(2)= hexagon((i + 1) % n, 2);
            perimeter += distance3D(p1, p2); // Wrap around to the first vertex
        }

        return perimeter;
    }




    // Function to compute the area of a 3D hexagon
    template <typename Type>
    Type calculate3DHexagonArea( tensor<Type, 2>& vertices) 
    {
        Type totalArea = 0.0;
        tensor<Type, 1> reference(3); 
        reference(0) =  vertices(0, 0); 
        reference(1) =  vertices(0, 1); 
        reference(2) =  vertices(0,2);

        // Iterate through the vertices and compute the area of each triangle
        for (int i = 1; i < 5; i++) {
            // Create two vectors representing two edges of the triangle
            tensor<Type, 1> edge1({
                vertices(i, 0) - reference(0),
                vertices(i, 1) - reference(1),
                vertices(i, 2) - reference(2)
            });

            tensor<Type, 1> edge2({
                vertices(i + 1, 0) - reference(0),
                vertices(i + 1, 1) - reference(1),
                vertices(i + 1, 2) - reference(2)
            });

            // Compute the cross product of the two edge vectors
            tensor<Type, 1> cross = crossProduct(edge1, edge2);

            // Compute the area of the triangle
            Type triangleArea = static_cast<Type>(0.5) * magnitude(cross);
            totalArea += triangleArea;
        }

        return totalArea;
    }



} // namespace miscellaneous2

#endif // MISCELLANEOUS2_H


    // // Function to compute the cross product of two vectors
    // tensor<double, 1> crossProduct( tensor<double, 1> vec1,  tensor<double, 1> vec2) 
    // {

    //     tensor<double,1> a(3); 
    //     a(0) = vec1(1) * vec2(2) - vec1(2) * vec2(1); 
    //     a(1) = vec1(2) * vec2(0) - vec1(0) * vec2(2); 
    //     a(2) = vec1(0) * vec2(1) - vec1(1) * vec2(0);
    //     return a;
    // }

    // // Function to compute the magnitude of a vector
    //  double magnitude(  tensor<double, 1> vec) 
    //  {
    //      return std::sqrt(vec(0) * vec(0) + vec(1) * vec(1) + vec(2) * vec(2));
    //  }

    // // Function to compute the area of a 3D hexagon
    // double calculate3DHexagonArea(tensor<double, 2> vertices) 
    // {
    //     double totalArea = 0.0;
    //     tensor<double, 1> reference = vertices(0,all);

    //     // Iterate through the vertices and compute the area of each triangle
    //     for (int i = 1; i < 5; i++) {
    //         // Create two vectors representing two edges of the triangle
    //         tensor<double, 1> edge1({
    //             vertices(i, 0) - reference(0),
    //             vertices(i, 1) - reference(1),
    //             vertices(i, 2) - reference(2)
    //         });

    //         tensor<double, 1> edge2({
    //             vertices(i + 1, 0) - reference(0),
    //             vertices(i + 1, 1) - reference(1),
    //             vertices(i + 1, 2) - reference(2)
    //         });

    //         // Compute the cross product of the two edge vectors
    //         tensor<double, 1> cross = crossProduct(edge1, edge2);

    //         // Compute the area of the triangle
    //         double triangleArea = 0.5 * std::sqrt(product(cross,cross,{{0,0}}));
    //         totalArea += triangleArea;
    //     }

    //     return totalArea;
    // }



    // Function to compute the cross product of two vectors




//Function to calculate the distance between two 3D points
// double distance3D(const std::vector<double>& p1, const std::vector<double>& p2) 
// {
//     return sqrt(pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2) + pow(p2[2] - p1[2], 2));
// }

// // Function to calculate the perimeter of a 3D hexagon
// double calculatePerimeter(const std::vector<std::vector<double>>& hexagon)
// {
//     double perimeter = 0.0;
//     int n = hexagon.size();

//     for (int i = 0; i < n; ++i) {
//         // Compute distance between consecutive vertices
//         perimeter += distance3D(hexagon[i], hexagon[(i + 1) % n]); // Wrap around to the first vertex
//     }

//     return perimeter;
// }
