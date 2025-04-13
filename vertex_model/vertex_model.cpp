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

#include "MiscellaneousRoutines.h"
#include <iostream>


int main(int argc, char *argv[]) 
{
    // Use the necessary namespaces to simplify the code
    using namespace std;
    using namespace hiperlife;
    using Teuchos::rcp;
    using Teuchos::RCP;
    using namespace hiperlife::Tensor;


    // Ensure enough arguments are passed
    if (argc < 4) 
    {
        std::cerr << "Usage: " << argv[0] << " <ratio> <r> <restart> <mesh_file>" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    double ratio_ = std::stod(argv[1]);
    double r = std::stod(argv[2]);
    bool restart = (std::stoi(argv[3]) != 0); // Use 0 or 1 as the third argument for restart

    // The mesh file name
    std::string file_name = argv[4];
    // **************************************************************//
    // *****                 INITIALIZATION                     *****//
    // **************************************************************//
    hiperlife::Init(argc, argv);
    const int myRank = hiperlife::MyRank();
    const int numProcs = hiperlife::NumProcs();

    HexagonalGrid grid,grid2;
    double radius = 10.0; 
    grid.generate_circular_mesh_with_boundary_nodes(16, r) ;
    //grid.generate_circular_hexagonal_grid_with_deleted_cell2(16,r,2,1);
    grid2.generate_circular_hexagonal_grid_with_deleted_closest_cell(16, r, 0., 0) ;
    grid.write_to_vtk("hexagonal_grid.vtk");
    grid2.write_to_vtk("hexagonal_grid2.vtk"); 
       
    int _nDim = 3;
    int totalTriangles = grid.hexagonIdCounter*6; // This method needs to be implemented in HexagonalGrid
    int totalnodes = grid.vertices.size()       ; // This method needs to be implemented in HexagonalGrid  
    int totalHexagons = grid.hexagonIdCounter   ; //nx*ny;

    int totalTriangles2 = grid2.hexagonIdCounter*6; // This method needs to be implemented in HexagonalGrid
    int totalnodes2 = grid2.vertices.size()       ; // This method needs to be implemented in HexagonalGrid  
    int totalHexagons2 = grid2.hexagonIdCounter   ; //nx*ny;


    bool debug_elem_flags          = false;
    bool debug_linear_constraints  = true;

    // Numerical parameters
    double tSimu=0.0;
    double deltat = 0.1; 
    double adaptive_time_step = 1.1;
    double totalTime = 1E6; // large number so the simulation reaches a steady state
    int timeStep = 0;
    int iter_print = 0 ;
    double tol = 1E-6;
    double deltat_max = 100; 

    // Material parameters
    double Ka = 1; 
    double Kp = 1.; 
    double gamma = 0.5;
    double force = 0.0;  //  0.1 (heal), 0.15(heal),0.175 ,0.2(heals), 0.206(heal), 0.2125(break), 0.225(break), 0.25 (break), 0.375 ,0.5 (break), 0.55 (break), 0.5875(break)     0.625(break)
    double fric  = 0.0001; 
    // Assuming grid.vertices is a container of std::pair<double, double>
    std::vector<std::pair<double, double>> vertexArray, vertexArray2;
    std::vector<std::tuple<int, int, int>> connectivityArray,connectivityArray2;
    std::vector<int> hexagonIdsForTriangles,hexagonIdsForTriangles2;
    std::vector<bool> flagsForTriangles,flagsForTriangles2;

    // Copy vertices from grid.vertices to vertexArray
    for (const auto& vertex : grid.vertices) 
    {
        vertexArray.push_back(vertex);
    }

    for (const auto& triangle : grid.triangles)
     {
        connectivityArray.emplace_back(triangle.v1, triangle.v2, triangle.v3);
    }



    // Populate the vector with hexagon IDs for each triangle
    for (int i = 0; i < totalTriangles; ++i) 
    {
        int hexagonId = grid.getHexagonIdForTriangle(i);
        int flagId = grid.getFlagForTriangle(i);
        auto hex_vertices  = grid.getVertexIndicesForHexagon(hexagonId) ;
        auto boundaryVertexIds   =  grid.getBoundaryVertexIds();

        hexagonIdsForTriangles.push_back(hexagonId);
        flagsForTriangles.push_back(flagId);
    }


    // Copy vertices from grid.vertices to vertexArray
    for (const auto& vertex : grid2.vertices) 
    {
        vertexArray2.push_back(vertex);
    }

    for (const auto& triangle : grid2.triangles)
    {
        
        connectivityArray2.emplace_back(triangle.v1, triangle.v2, triangle.v3);
    }

    // Populate the vector with hexagon IDs for each triangle
    for (int i = 0; i < totalTriangles2; ++i) 
    {
        int hexagonId = grid2.getHexagonIdForTriangle(i);
        int flagId = grid2.getFlagForTriangle(i);
        hexagonIdsForTriangles2.push_back(hexagonId);
        flagsForTriangles2.push_back(flagId);       
    }

    auto  boundaryVertexIds   =  grid.getBoundaryVertexIds();
    auto  boundaryVertexIds2  =  grid2.getBoundaryVertexIds();
   


    // Initialize and set various simulation parameters within the user structure
    //RCP<UserStructure> userStr = rcp(new UserStructure);
    SmartPtr<ParamStructure> userStr   = Create<ParamStructure>() ;// = CreateParamStructure<>();
    userStr->dparam.resize(100); // Ensure the parameter vector is of sufficient size
    userStr->iparam.resize(10); // Ensure the parameter vector is of sufficient size
    userStr->iparam[0] = 0;
    userStr->bparam.resize(20);   
    userStr->dparam[0] = Ka;
    userStr->dparam[1] = Kp;
    userStr->dparam[2] = gamma;
    userStr->dparam[3] = force;
    userStr->dparam[4] = ratio_;
    userStr->dparam[5] = radius;   
    userStr->dparam[6] = deltat;   
    userStr->bparam[0] = true;
    userStr->bparam[1] = restart;
    userStr->i_aux.resize(0);
    userStr->j_aux.resize(0);
    // // Initialize basicMesh corresponding to the new distributed mesh
    SmartPtr<BasicMeshGenerator> basicMesh =  Create<BasicMeshGenerator>() ;  //rcp(new BasicMeshGenerator());     
    basicMesh->setMesh(ElemType::Triang, BasisFuncType::SubdivSurfs, 2);
    basicMesh->setMeshType(MeshType::Sequential);
    basicMesh->_glob_x_nodes.resize(totalnodes * 3, 0);
    basicMesh->setNElem(totalTriangles);
    basicMesh->setNPts(totalnodes);
    basicMesh->_glob_connec.resize(totalTriangles * 3, 0);
    basicMesh->_glob_x_nodes.resize(totalnodes * 3, 0);
    basicMesh->_glob_creases.resize(totalnodes*2,0);
    basicMesh->_glob_eflags.resize(totalTriangles , 0);

    if (myRank==0)
    {
        for (int i = 0; i < totalnodes; i++)
        {
            double x[3] = {0.0, 0.0, 0.0}; // Initialize x array. Assumes Z = 0 for 2D vertices.

            // Assign x and y from vertexArray. Ensure that Z is set to 0 or some default value for 2D meshes.
            x[0] = vertexArray[i].first;  // X-coordinate
            x[1] = vertexArray[i].second; // Y-coordinate
            x[2] = 0 ;

            basicMesh->_glob_x_nodes[_nDim * i + 0] = x[0];
            basicMesh->_glob_x_nodes[_nDim * i + 1] = x[1];
            basicMesh->_glob_x_nodes[_nDim * i + 2] = x[2]; // For 2D vertices, this will be 0 or adjust if your data has Z-coordinates.
        }
        // // Set global connectivity

        for (int i = 0; i < totalTriangles; i++)
        {
            basicMesh->_glob_connec[3 * i + 0] = std::get<0>(connectivityArray[i]);
            basicMesh->_glob_connec[3 * i + 1] = std::get<1>(connectivityArray[i]);
            basicMesh->_glob_connec[3 * i + 2] = std::get<2>(connectivityArray[i]);
        }

        for (int i = 0; i < totalnodes; i++)
        {
            bool isBoundary = false; // Assume the vertex is not a boundary vertex initially
            // Check if i is in the boundaryVertexIds array
            for (int boundaryVertexId : boundaryVertexIds) {
                if (i == boundaryVertexId) {
                    isBoundary = true; // Found i in boundaryVertexIds, it is a boundary vertex
                    break; // No need to check further
                }
            }

            // Set _glob_creases based on whether i is a boundary vertex
            if (isBoundary)
                basicMesh->_glob_creases[2*i] = 1;
        }
        

        for (int i = 0; i < totalHexagons; i++)
        {
            // Example: Assuming a function to calculate or retrieve the barycenter node index for hexagon i
            int master = grid.getBarycenterNodeIndexForHexagon(i);
            // Example: Assuming a function to retrieve the indices of the six vertices for hexagon i
            std::vector<int> slaves = grid.getVertexIndicesForHexagon(i);

            double x_master = 0.0;
            double y_master = 0.0;
            double z_master = 0.0;

            for (int j = 0; j < 6; j++)
            {
                x_master += (1.0 / 6) * basicMesh->_glob_x_nodes[_nDim * slaves[j] + 0] ;
                y_master += (1.0 / 6) * basicMesh->_glob_x_nodes[_nDim * slaves[j] + 1] ;
                z_master += (1.0 / 6) * basicMesh->_glob_x_nodes[_nDim * slaves[j] + 2] ;
            }

            basicMesh->_glob_x_nodes[_nDim * master + 0]  =x_master ; 
            basicMesh->_glob_x_nodes[_nDim * master + 1]  =y_master; 
            basicMesh->_glob_x_nodes[_nDim * master + 2]  =z_master  ;         


            std::vector<double> wgt(6, 1.0 / 6); // Initialize weights
            double offset = 0.0;
            // Adjust the following line according to your actual structure for _linearConstraints
            basicMesh->_linearConstraints.push_back({master, slaves, wgt, offset});

        }

        for (int i = 0; i < totalTriangles; i++)
        {
            basicMesh->_glob_eflags[i]   = hexagonIdsForTriangles[i];

        }
    }


    SmartPtr<DistributedMesh> disMesh  =   Create<DistributedMesh>();   //rcp(new DistributedMesh);
    SmartPtr<DistributedMesh> disMesh3  =   Create<DistributedMesh>();   //rcp(new DistributedMesh);
    // Distribute the mesh for tension calculations and update it
    if (restart)
    {
        Teuchos::RCP<MeshLoader> loadedMesh = Teuchos::rcp(new MeshLoader);
        loadedMesh->setElemType(ElemType::Triang);
        loadedMesh->setBasisFuncType(BasisFuncType::SubdivSurfs);
        loadedMesh->setBasisFuncOrder(2);
        loadedMesh->loadMesh(file_name,MeshType::Parallel);
        disMesh->setMesh(loadedMesh);
    }
    else
    {
        disMesh->setMesh(basicMesh);
    }
    disMesh->setBalanceMesh(false); // Balance the mesh for better performance
    disMesh->Update();    
    disMesh->printFileLegacyVtk("disMesh");

    // Distribute the mesh for tension calculations and update it

    disMesh3->setMesh(basicMesh);
    disMesh3->setBalanceMesh(false); // Balance the mesh for better performance
    disMesh3->Update();    
    disMesh3->printFileLegacyVtk("disMesh3");



    cout<<"dismesh created"<<endl;
	SmartPtr<DistributedMesh> disMesh2;
    int map[totalnodes2];
    if (restart)
    { 
      
        SmartPtr<BasicMeshGenerator> basicMesh2 =  Create<BasicMeshGenerator>() ;  //rcp(new BasicMeshGenerator());    
        basicMesh2->setMesh(ElemType::Triang, BasisFuncType::SubdivSurfs, 2);
        basicMesh2->setMeshType(MeshType::Sequential);
        basicMesh2->_glob_x_nodes.resize(totalnodes2 * 3, 0);
        basicMesh2->setNElem(totalTriangles2);
        basicMesh2->setNPts(totalnodes2);
        basicMesh2->_glob_connec.resize(totalTriangles2 * 3, 0);
        basicMesh2->_glob_x_nodes.resize(totalnodes2 * 3 , 0);
        basicMesh2->_glob_creases.resize(totalnodes2 * 2,0);
        basicMesh2->_glob_eflags.resize(totalTriangles2 , 0);

        if (myRank==0)
        { 
            for (int i2 = 0; i2 < totalnodes2; i2++)
            {
                double x[3] = {0.0, 0.0, 0.0}; 
                double x2[3] = {0.0, 0.0, 0.0}; 
                // Assign x and y from vertexArray. Ensure that Z is set to 0 or some default value for 2D meshes.
                bool node_found = false;
                for (int i = 0; i < totalnodes; i++)
                {
                    if ( abs(vertexArray2[i2].first - vertexArray[i].first)  < tol  and  abs(vertexArray2[i2].second - vertexArray[i].second) < tol  )
                    {
                        x[0] = disMesh->_nodeData->getValue(0,i,IndexType::Global);
                        x[1] = disMesh->_nodeData->getValue(1,i,IndexType::Global);
                        x[2] = disMesh->_nodeData->getValue(2,i,IndexType::Global);
                        node_found = true;
                        map[i2]   = i;
                        break; 
                    }
                }
                if (!node_found)
                {
                    throw std::runtime_error("A runtime error has occurred!");
                }
                x2[0] = x[0];
                x2[1] = x[1];
                x2[2] = x[2];

                basicMesh2->_glob_x_nodes[_nDim * i2 + 0] = x2[0] ;
                basicMesh2->_glob_x_nodes[_nDim * i2 + 1] = x2[1] ;
                basicMesh2->_glob_x_nodes[_nDim * i2 + 2] = x2[2]  ;
            }

            // Set global connectivity
            for (int i = 0; i < totalTriangles2; i++)
            {
                basicMesh2->_glob_connec[3 * i + 0] = std::get<0>(connectivityArray2[i]);
                basicMesh2->_glob_connec[3 * i + 1] = std::get<1>(connectivityArray2[i]);
                basicMesh2->_glob_connec[3 * i + 2] = std::get<2>(connectivityArray2[i]);
            }
            for (int i = 0; i < totalnodes2; i++)
            {
                bool isBoundary = false; // Assume the vertex is not a boundary vertex initially
                // Check if i is in the boundaryVertexIds array
                for (int boundaryVertexId : boundaryVertexIds2) {
                    if (i == boundaryVertexId) {
                        isBoundary = true; // Found i in boundaryVertexIds, it is a boundary vertex
                        break; // No need to check further
                    }
                }

                // Set _glob_creases based on whether i is a boundary vertex
                if (isBoundary)
                    basicMesh2->_glob_creases[2*i] = 1;
    
            }
        }
        for (int i = 0; i < totalHexagons2; i++) 
        {
            int master = grid2.getBarycenterNodeIndexForHexagon(i);
            std::vector<int> slaves = grid2.getVertexIndicesForHexagon(i);
            std::vector<double> wgt(6, 1.0 / 6); // Initialize weights
            double offset = 0.0;
            basicMesh2->_linearConstraints.push_back({master, slaves, wgt, offset});
        }

        for (int i = 0; i < totalTriangles2; i++)
        {
            basicMesh2->_glob_eflags[i] = hexagonIdsForTriangles2[i];
        }
        disMesh2  =   Create<DistributedMesh>();   //rcp(new DistributedMesh);
        // Distribute the mesh for tension calculations and update it
        disMesh2->setMesh(basicMesh2);
        disMesh2->setBalanceMesh(false); // Balance the mesh for better performance
        disMesh2->Update();     

        totalHexagons  = totalHexagons2;
        totalnodes     = totalnodes2;
        totalTriangles = totalTriangles2;

        std::vector<std::pair<double, double>> circularGrid;
        for (int i = 0; i < disMesh3->loc_nPts(); i++) 
        {
            double x = disMesh3->nodeCoord(i, 0, IndexType::Local);
            double y = disMesh3->nodeCoord(i, 1, IndexType::Local);
            circularGrid.emplace_back(x, y);
        }

        std::vector<Vertex3D> sphericalGrid = transformToHemisphericalGrid(circularGrid, radius);

        for (int i = 0; i < disMesh3->loc_nPts(); ++i) 
        {
            disMesh3->_nodeData->setValue(0, i, IndexType::Local, sphericalGrid[i].x);
            disMesh3->_nodeData->setValue(1, i, IndexType::Local, sphericalGrid[i].y);
            disMesh3->_nodeData->setValue(2, i, IndexType::Local, sphericalGrid[i].z);
        }                                                                                                         
        disMesh3->printFileLegacyVtk("dofHand3");          
    }
    else
    {
        std::vector<std::pair<double, double>> circularGrid;
        for (int i = 0; i < disMesh->loc_nPts(); i++) 
        {
            double x = disMesh->nodeCoord(i, 0, IndexType::Local);
            double y = disMesh->nodeCoord(i, 1, IndexType::Local);
            circularGrid.emplace_back(x, y);
        }

        std::vector<Vertex3D> sphericalGrid = transformToHemisphericalGrid(circularGrid, radius);

        for (int i = 0; i < disMesh->loc_nPts(); ++i) 
        {
            disMesh->_nodeData->setValue(0, i, IndexType::Local, sphericalGrid[i].x);
            disMesh->_nodeData->setValue(1, i, IndexType::Local, sphericalGrid[i].y);
            disMesh->_nodeData->setValue(2, i, IndexType::Local, sphericalGrid[i].z);
        }   
                                                                                                        
    }
    cout<<"Basic mesh created"<<endl;
    for (int i = 0; i < disMesh->loc_nElem(); i++)
    {
        int hex_id    =  disMesh->elemTag(0, i, IndexType::Local);

        int cell_flag{};
        if (restart)
        {
            cell_flag  =  flagsForTriangles2[i]; 
        }
        else
        {
            cell_flag =  flagsForTriangles[i]; 
        }

        userStr->i_aux.push_back(hex_id);
        userStr->j_aux.push_back(cell_flag);
    }

    if (!restart)
    {
        for (int i = 0; i < totalHexagons; i++)
        {
            std::vector<int> slaves = grid.getVertexIndicesForHexagon(i);
            std::vector<std::pair<double, double>> hexagonVertices;
            // Loop over slave nodes to collect the hexagon vertices
            for (int j = 0; j < slaves.size(); ++j)
            {
                double x = disMesh->nodeCoord(slaves[j], 0, IndexType::Global);
                double y = disMesh->nodeCoord(slaves[j], 1, IndexType::Global);
                hexagonVertices.emplace_back(x, y);
            }
            // Use random device and Mersenne Twister engine for better randomness
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 0.3);
            double distortion_range  = dis(gen);
            // Transform the hexagon vertices to be closer to a square
            auto transformedVertices = transformHexagonToIrregularConvex(hexagonVertices, distortion_range);
            // Example usage of changeAreaBasedOnPerturbation with random area factor
            std::uniform_real_distribution<> areaFactorDist(0.85, 1.15); // Random area factor between 0.5 and 2.0
            double areaFactor = areaFactorDist(gen); // Randomly generate area factor
            auto scaledVertices = changeAreaBasedOnPerturbation(hexagonVertices, areaFactor);
            // Apply the transformed coordinates back to the mesh
            for (int j = 0; j < slaves.size(); ++j)
            {
                double x_prime = scaledVertices[j].first;
                double y_prime = scaledVertices[j].second;

                disMesh->_nodeData->setValue(0, slaves[j], IndexType::Global, x_prime);
                disMesh->_nodeData->setValue(1, slaves[j], IndexType::Global, y_prime);
            }
        }
    }


    // Initialize a DOFsHandler for position calculations
    SmartPtr<DOFsHandler> dofHand;
	if (restart)
		dofHand = Create<DOFsHandler>(disMesh2);
	else
	    dofHand = Create<DOFsHandler>(disMesh);
    try 
    {
        // Set the name tag and define degrees of freedom and auxiliary fields
        dofHand->setNameTag("dofHand");
        dofHand->setDOFs({"x", "y", "z"}); // Main degrees of freedom
        dofHand->setNodeAuxF({"x0", "y0", "z0","creaseId"}); // Auxiliary fields
        dofHand->setElemAuxF({"cell_id","cell_flag"}); // Auxiliary fields
        dofHand->Update(); // Finalize the DOFsHandler setup
    }
    catch (runtime_error &err) 
    {
        cout << myRank << ": DOFHandler could not be created " << err.what() << endl;
        MPI_Finalize();
        return 1; // Exit if there was an error creating the DOFsHandler
    }


    cout<<"dofHand is created"<<endl;

    for (int i = 0; i < dofHand->mesh->loc_nPts(); i++)
    {
        double x = dofHand->mesh->_nodeData->getValue(0,i,IndexType::Local );
        double y = dofHand->mesh->_nodeData->getValue(1,i,IndexType::Local );
        double z = dofHand->mesh->_nodeData->getValue(2,i,IndexType::Local );
        int  crease  =  dofHand->mesh->nodeCrease(i, IndexType::Local); 
        if (!restart)
		{
        	dofHand->nodeAuxF->setValue("x0", i, IndexType::Local, x);
        	dofHand->nodeAuxF->setValue("y0", i, IndexType::Local, y);
        	dofHand->nodeAuxF->setValue("z0", i, IndexType::Local, z);
		}
		else
		{
            
           double x0_old = disMesh3->_nodeData->getValue(0,map[i],IndexType::Local );
           double y0_old = disMesh3->_nodeData->getValue(1,map[i],IndexType::Local );
           double z0_old = disMesh3->_nodeData->getValue(2,map[i],IndexType::Local );
           cout<<i<<" "<<map[i]<<endl;
           dofHand->nodeAuxF->setValue("x0", i, IndexType::Local, x0_old);
           dofHand->nodeAuxF->setValue("y0", i, IndexType::Local, y0_old);
           dofHand->nodeAuxF->setValue("z0", i, IndexType::Local, z0_old);           
		}
        dofHand->nodeAuxF->setValue("creaseId", i, IndexType::Local, crease);
        dofHand->nodeDOFs->setValue("x", i, IndexType::Local, x);
        dofHand->nodeDOFs->setValue("y", i, IndexType::Local, y);
        dofHand->nodeDOFs->setValue("z", i, IndexType::Local, z);
    }
    if (restart)
    	disMesh = disMesh2; 

    SmartPtr<HiPerProblem> hiperProbl = Create<HiPerProblem>();
    // Associate the user-defined structure with the problem for simulation context
    hiperProbl->setParameterStructure(userStr);
    // Set parameters for consistency checks in the simulation
    hiperProbl->setConsistencyCheckDelta(1.E-6);
    // Associate the DOFsHandler with the HiPerProblem
    hiperProbl->setDOFsHandlers({dofHand});

    // Configure integration methods for the problem
    hiperProbl->setIntegration("Integ", {"dofHand"}); // Define integration strategy
    hiperProbl->setIntegration("BorderInteg",{"dofHand"});
    hiperProbl->setCubatureGauss("Integ", 1); // Use 1-point Gauss integration
    hiperProbl->setCubatureBorderGauss("BorderInteg",1);
    // Otherwise, use a default strategy
    hiperProbl->setElementFillings("Integ", LS);
    hiperProbl->setElementFillings("BorderInteg", LS_Border);
    hiperProbl->Update(); // Update to finalize the HiPerProblem configuration

    cout<<"hiperproblem is created"<<endl;


    for (int i = 0; i < dofHand->mesh->loc_nElem(); i++)
    {
        int cell_id  =  dofHand->mesh->_elemFlags->getValue(0, i, IndexType::Local);
        
        int cell_flag= 0; 
        if (restart) 
           cell_flag =  flagsForTriangles2[i]; //FIXME::disMesh->_elemFlags->getValue(1, i, IndexType::Local);
        else
           cell_flag =  flagsForTriangles[i];             
        dofHand->elemAuxF->setValue("cell_id"  , i, IndexType::Local,cell_id);
        dofHand->elemAuxF->setValue("cell_flag", i, IndexType::Local,cell_flag);
    }

    hiperProbl->FillLinearSystem();
    dofHand->UpdateGhosts();

    RCP<MUMPSDirectLinearSolver> linSolver =  rcp(new MUMPSDirectLinearSolver());
    linSolver->setHiPerProblem(hiperProbl);
    linSolver->setMatrixType(MUMPSDirectLinearSolver::MatrixType::General); //General, POD or Symmetric
    linSolver->setAnalysisType(MUMPSDirectLinearSolver::AnalysisType::Parallel); //Sequential or parallel
    linSolver->setOrderingLibrary(MUMPSDirectLinearSolver::OrderingLibrary::Auto); //Many options here
    linSolver->setVerbosity(MUMPSDirectLinearSolver::Verbosity::None);
    linSolver->setDefaultParameters();
    linSolver->setWorkSpaceMemoryIncrease(200);
    linSolver->Update();
    cout<<"linSolver is created"<<endl;
    RCP<NewtonRaphsonNonlinearSolver> nonlinSolver =  rcp ( new NewtonRaphsonNonlinearSolver());
    nonlinSolver->setLinearSolver(linSolver);
    nonlinSolver->setMaxNumIterations(6);
    nonlinSolver->setResTolerance(1E-4);
    nonlinSolver->setSolTolerance(1E-4);
    nonlinSolver->setLineSearch(false);
    nonlinSolver->setPrintIntermInfo(true);
    nonlinSolver->setConvRelTolerance(false);
    nonlinSolver->Update();          

    std::ofstream outFile2("wound.txt"); // Create or overwrite the file
    dofHand->nodeDOFs0->setValue(dofHand->nodeDOFs);
    dofHand->printFileLegacyVtk("dofHand",true);


    while (true)
    {
        cout<<"Time no:: "<<timeStep<<" and delta_t:: "<<deltat<<endl;
        //Initial guess
        dofHand->nodeDOFs->setValue(dofHand->nodeDOFs0);
        bool converged = nonlinSolver->solve();
        if (converged)
        {
            dofHand->printFileLegacyVtk("mesh_to_read_"+to_string(timeStep),true);
            timeStep++;
            dofHand->nodeDOFs0->setValue(dofHand->nodeDOFs); 


            for (int i = 0; i < dofHand->mesh->loc_nPts(); i++)
            {
                double x = dofHand->nodeDOFs->getValue(0, i, IndexType::Local);
                double y = dofHand->nodeDOFs->getValue(1, i, IndexType::Local);
                double z = dofHand->nodeDOFs->getValue(2, i, IndexType::Local);
                dofHand->mesh->_nodeData->setValue(0,i,IndexType::Local,x );
                dofHand->mesh->_nodeData->setValue(1,i,IndexType::Local,y );
                dofHand->mesh->_nodeData->setValue(2,i,IndexType::Local,z );
            }

            deltat *= adaptive_time_step;
            if (deltat>deltat_max)
                deltat =deltat_max; 
              
            if (restart)
            {   

                std::vector<std::vector<double>> nodes;
                for (int i = 0; i < dofHand->mesh->loc_nPts(); i++)
                {
                    double x = dofHand->mesh->_nodeData->getValue(0,i,IndexType::Local );
                    double y = dofHand->mesh->_nodeData->getValue(1,i,IndexType::Local );
                    double z = dofHand->mesh->_nodeData->getValue(2,i,IndexType::Local );

                    // double x0 = dofHand->nodeAuxF->getValue("x0", i, IndexType::Local);
                    // double y0 = dofHand->nodeAuxF->getValue("y0", i, IndexType::Local);
                    // double z0 = dofHand->nodeAuxF->getValue("z0", i, IndexType::Local);

                    int crease = dofHand->mesh->nodeCrease(i, IndexType::Local);

                    if (crease>0 and z>-3)
                    {
                        nodes.push_back({x, y, z});
                    }
                }

                outFile2 << calculateHexagonArea3D(nodes) << std::endl; // Write iter_print to the file































            }

        }
        else
        {
            deltat /= adaptive_time_step;
            dofHand->nodeDOFs->setValue(dofHand->nodeDOFs0);

        }
        userStr->dparam[6]  = deltat ; 



    std::vector<Hexagon> hexagons;

    for (int i = 0; i < totalHexagons; i++)
    {
        std::vector<int> boundaryNodeIndices = (restart) ? grid2.getVertexIndicesForHexagon(i)
                                                         : grid.getVertexIndicesForHexagon(i);

        std::vector<Vertex3D> boundaryNodes;

        // Loop over boundary nodes to collect the hexagon vertices
        for (int j = 0; j < boundaryNodeIndices.size(); ++j)
        {
            double x = dofHand->mesh->nodeCoord(boundaryNodeIndices[j], 0, IndexType::Global);
            double y = dofHand->mesh->nodeCoord(boundaryNodeIndices[j], 1, IndexType::Global);
            double z = dofHand->mesh->nodeCoord(boundaryNodeIndices[j], 2, IndexType::Global);
            boundaryNodes.emplace_back(x, y, z);
        }
        hexagons.push_back(Hexagon{boundaryNodes}); 
    }

    string name = "dofHand_" + to_string(timeStep) + ".vtk";
    writeHexagonalMeshToVTK(name, hexagons);                                                            









        if (nonlinSolver->numberOfIterations() < 3 and !restart) //nonlinSolver->numberOfIterations() < 3) 
        {
            std::ofstream outFile("iter_print.txt"); // Create or overwrite the file
            if (outFile.is_open()) 
            {
                outFile << timeStep-1 << std::endl; // Write iter_print to the file
                outFile.close(); // Close the file
            } 
            else 
            {
                std::cerr << "Error: Could not open iter_print.txt for writing." << std::endl;
                return 1; // Return an error code
            }
            return 0; // Return success
        }

    }


}






    // hiperProbl = Create<HiPerProblem>();
    // // Associate the user-defined structure with the problem for simulation context
    // hiperProbl->setParameterStructure(userStr);
    // // Set parameters for consistency checks in the simulation
    // hiperProbl->setConsistencyCheckDelta(1.E-6);
    // // Associate the DOFsHandler with the HiPerProblem
    // hiperProbl->setDOFsHandlers({dofHand});

    // // Configure integration methods for the problem
    // hiperProbl->setIntegration("Integ", {"dofHand"}); // Define integration strategy
    // hiperProbl->setIntegration("BorderInteg",{"dofHand"});
    // hiperProbl->setCubatureGauss("Integ", 1); // Use 1-point Gauss integration
    // hiperProbl->setCubatureBorderGauss("BorderInteg",1);
    // // Otherwise, use a default strategy
    // hiperProbl->setElementFillings("Integ", LS);
    // hiperProbl->setElementFillings("BorderInteg", LS_Border);


    // for (int i = 0; i < dofHand->mesh->loc_nPts(); i++)
    // {
    //     if(constraint_flag[i] == 0)
    //     {
    //         double x = dofHand->nodeDOFs->getValue(0, i, IndexType::Local);
    //         double y = dofHand->nodeDOFs->getValue(1, i, IndexType::Local);
    //         double z = dofHand->nodeDOFs->getValue(2, i, IndexType::Local);


    //         std::vector<int> nbors  =  disMesh->_graph->getNborList(i,  IndexType::Local);

    //         for (auto n : nbors)
    //         {
    //             if(n < i)
    //             {

    //                 double xn = dofHand->nodeDOFs->getValue(0, n, IndexType::Global);
    //                 double yn = dofHand->nodeDOFs->getValue(1, n, IndexType::Global);
    //                 double zn = dofHand->nodeDOFs->getValue(2, n, IndexType::Global);

    //                 double norm = sqrt((x-xn)*(x-xn) + (y-yn)*(y-yn) + (z-zn)*(z-zn));

    //                 if(norm < 1.E-2)
    //                 {
    //                     hiperProbl->setLinearConstraint({0,0,i,IndexType::Local},{0,0,n,IndexType::Global},1.0,0.0);
    //                     hiperProbl->setLinearConstraint({0,1,i,IndexType::Local},{0,1,n,IndexType::Global},1.0,0.0);
    //                     hiperProbl->setLinearConstraint({0,2,i,IndexType::Local},{0,2,n,IndexType::Global},1.0,0.0);

    //                     constraint_flag[i] = 1;
    //                 }
    //             }

                
    //         }
    //     }
        
    // }

    // hiperProbl->Update(); // Update to finalize the HiPerProblem configuration
    // linSolver =  rcp(new MUMPSDirectLinearSolver());
    // linSolver->setHiPerProblem(hiperProbl);
    // linSolver->setMatrixType(MUMPSDirectLinearSolver::MatrixType::General); //General, POD or Symmetric
    // linSolver->setAnalysisType(MUMPSDirectLinearSolver::AnalysisType::Parallel); //Sequential or parallel
    // linSolver->setOrderingLibrary(MUMPSDirectLinearSolver::OrderingLibrary::Auto); //Many options here
    // linSolver->setVerbosity(MUMPSDirectLinearSolver::Verbosity::None);
    // linSolver->setDefaultParameters();
    // linSolver->setWorkSpaceMemoryIncrease(200);
    // linSolver->Update();


    // nonlinSolver =  rcp ( new NewtonRaphsonNonlinearSolver());
    // nonlinSolver->setLinearSolver(linSolver);
    // nonlinSolver->setMaxNumIterations(6);
    // nonlinSolver->setResTolerance(1E-5);
    // nonlinSolver->setSolTolerance(1E-5);
    // nonlinSolver->setLineSearch(false);
    // nonlinSolver->setPrintIntermInfo(true);
    // nonlinSolver->setConvRelTolerance(false);
    // nonlinSolver->Update();  
