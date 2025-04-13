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
#include <cmath>
#include "miscellaneous2.h"


using namespace std;
using namespace hiperlife::Tensor;
using namespace  miscellaneous2;


void LS(hiperlife::FillStructure& fillStr)
{   
    using namespace hiperlife;
    using namespace ttl;
    using namespace autodiff;

    // Define Energy
    auto energy = [&fillStr](std::vector<dual2nd> nborDOFs_vec) -> dual2nd
    {   
        using namespace hiperlife;
        using namespace ttl;
        using index::i, index::j, index::l, index::m, index::n, index::o;

        SubFillStructure& subFill = fillStr["dofHand"];
        int numDOFs = subFill.numDOFs;
        int nDim = subFill.nDim;
        int pDim = subFill.pDim;
        int eNN    = subFill.eNN;
        int numAuxF= subFill.numAuxF;
        wrapper<dual2nd,2> nborDOFs(nborDOFs_vec.data(), eNN, numDOFs);
        wrapper<double,2>  nborDOFs_n(subFill.nborDOFs0.data(), eNN, numDOFs);         
        wrapper<double,2>  nborAuxF(subFill.nborAuxF.data(), eNN, numAuxF);                  
        wrapper<double,2>  nborCoords(subFill.nborCoords.data(), eNN, nDim);
        int    hex_id           = fillStr.paramStr->i_aux[subFill.loc_elemID];
        int    flag_hexagon     = fillStr.paramStr->j_aux[subFill.loc_elemID];
        double jac;
        double deltat   =   fillStr.paramStr->dparam[6];  
        double Ka        =   fillStr.paramStr->dparam[0]*deltat;
        double Kp        =   fillStr.paramStr->dparam[1]*deltat;
        double radius    =   fillStr.paramStr->dparam[5]; 
        double K         =   100*deltat;
        double eta       =   10;

        tensor<double,2> nborAuxF_coords  =  nborAuxF(all,range(0,2));
        tensor<double,1> nborAuxF_crease  =  nborAuxF(all,3);             

        tensor<dual2nd,2> nborDOFs_hexagon_tensor     =   nborDOFs(range(1,6),all) ; 
        tensor<double,2>  nborDOFs_hexagon_tensor_0   =   nborAuxF_coords(range(1,6),all) ; 
        tensor<double,2>  nborDOFs_hexagon_tensor_n   =   nborDOFs_n(range(1,6),all) ; 

        dual2nd A   = calculate3DHexagonArea(nborDOFs_hexagon_tensor) ;
        dual2nd P   = calculatePerimeter(nborDOFs_hexagon_tensor);
        double A_0  = calculate3DHexagonArea(nborDOFs_hexagon_tensor_0) ;
        double P_0  = calculatePerimeter(nborDOFs_hexagon_tensor_0);


        if (flag_hexagon==1)
        { 
            dual2nd penalty_term = 0.0 ;
            tensor<dual2nd,1> linear_constraint  = nborDOFs(0,all) ; ;           
            for (int o = 0; o < 6; ++o)
            {
                dual2nd x2         =  nborDOFs_hexagon_tensor(o,all)(j)* nborDOFs_hexagon_tensor(o,all)(j);

                penalty_term           +=   pow(x2-radius*radius,2); 
                linear_constraint      -=  (1.0/6.0)* nborDOFs_hexagon_tensor(o,all); 
                
            }
            dual2nd linear_constraint2  = linear_constraint(o)*linear_constraint(o);

            dual2nd dissipation_term = (nborDOFs_hexagon_tensor(i,j)- nborDOFs_hexagon_tensor_n(i,j))*(nborDOFs_hexagon_tensor(i,j)- nborDOFs_hexagon_tensor_n(i,j));

            return 0.5*(Ka*(A-A_0)*(A-A_0)+ Kp*(P-P_0)*(P-P_0)+ K*penalty_term + eta*dissipation_term );
            
        }
        else
        {
            dual2nd zero = 0;   
            return zero;
        }

        
    };

    SubFillStructure& subFill = fillStr["dofHand"];
    int eNN  = subFill.eNN;
    int numDOFs = subFill.numDOFs;
    int    flag_hexagon     = fillStr.paramStr->j_aux[subFill.loc_elemID];
    std::vector<dual2nd> dofs(eNN * numDOFs);
    for (int i = 0; i < eNN * numDOFs; i++)
        dofs[i] = subFill.nborDOFs[i];
    

    dual2nd en;
    hessian(energy, wrt(dofs), at(dofs), en, fillStr.Bk(0), fillStr.Ak(0,0));
    wrapper<double,2> Bk(fillStr.Bk(0).data(), eNN, numDOFs);
    wrapper<double,4> Ak(fillStr.Ak(0,0).data(), eNN, numDOFs, eNN, numDOFs);
    wrapper<double,2> nborDOFs(subFill.nborDOFs.data(), eNN, numDOFs);

}


void LS_Border(hiperlife::FillStructure& fillStr)
{   
    using namespace hiperlife;
    using namespace ttl;
    using namespace autodiff;
    using namespace hiperlife::Tensor;

    // Define Energy
    auto energy = [&fillStr](std::vector<dual2nd> nborDOFs_vec) -> dual2nd
    {   
        using namespace hiperlife;
        using namespace ttl;
        using namespace hiperlife::Tensor;
        using index::i, index::j, index::l, index::m, index::n,index::L;

        SubFillStructure& subFill = fillStr["dofHand"];
        int numDOFs = subFill.numDOFs;
        int nDim = subFill.nDim;
        int pDim = subFill.pDim;
        int eNN    = subFill.eNN;
        int numAuxF= subFill.numAuxF;
        wrapper<dual2nd,2> nborDOFs(nborDOFs_vec.data(), eNN, numDOFs);
        wrapper<double,2>  nborDOFs_n(subFill.nborDOFs0.data(), eNN, numDOFs);         
        wrapper<double,2>  nborAuxF(subFill.nborAuxF.data(), eNN, numAuxF);                  
        wrapper<double,2>  nborCoords(subFill.nborCoords.data(), eNN, nDim);
        wrapper<double,1> bf(subFill.getDer(0),eNN);

        tensor<double,2> nborAuxF_coords  =  nborAuxF(all,range(0,2));
        tensor<double,1> nborAuxF_crease  =  nborAuxF(all,3);       

        int    hex_id           = fillStr.paramStr->i_aux[subFill.loc_elemID];
        int    flag_hexagon     = fillStr.paramStr->j_aux[subFill.loc_elemID];
        double jac;
        double Ka        =   fillStr.paramStr->dparam[0];
        double Kp        =   fillStr.paramStr->dparam[1];
        double radius    =   fillStr.paramStr->dparam[5]; 
        double K         =   1E2;
        double eta       =   10;
        double deltat   =   fillStr.paramStr->dparam[6];  
        double gamma     =   2.0*deltat; 
        double ratio       =  fillStr.paramStr->dparam[4];   
        double ratio_2     =  0.01;           


        tensor<double,1> x0 = bf(L)*nborDOFs_n(L,l);
        if (x0(2)>-3) // Around the wound.
        {       
            gamma = ratio*gamma;
            ratio_2 = 0 ; 
        }

        tensor<dual2nd,2> nborDOFs_line(2,3);
        nborDOFs_line(0,all)   =   nborDOFs(1,all) ; 
        nborDOFs_line(1,all)   =   nborDOFs(6,all) ; 
        tensor<dual2nd,1>  ell =   nborDOFs(1,all) -  nborDOFs(6,all);
        tensor<double,1>   ell0 =   nborAuxF_coords(1,all) -  nborAuxF_coords(6,all);

        dual2nd ell_mag  = pow(ell(0)* ell(0) + ell(1)* ell(1)  + ell(2)* ell(2)  , 0.5);
        double ell_mag0  = pow(ell0(0)* ell0(0) + ell0(1)* ell0(1)  + ell0(2)* ell0(2)  , 0.5);


        return gamma*(ell_mag   + ratio_2*(ell_mag-ell_mag0)*(ell_mag-ell_mag0));


    };

    SubFillStructure& subFill = fillStr["dofHand"];
    int eNN  = subFill.eNN;
    int numDOFs = subFill.numDOFs;
    int    flag_hexagon     = fillStr.paramStr->j_aux[subFill.loc_elemID];
    std::vector<dual2nd> dofs(eNN * numDOFs);
    for (int i = 0; i < eNN * numDOFs; i++)
        dofs[i] = subFill.nborDOFs[i];
    

    dual2nd en;
    hessian(energy, wrt(dofs), at(dofs), en, fillStr.Bk(0), fillStr.Ak(0,0));
    wrapper<double,2> Bk(fillStr.Bk(0).data(), eNN, numDOFs);
    wrapper<double,4> Ak(fillStr.Ak(0,0).data(), eNN, numDOFs, eNN, numDOFs);
    wrapper<double,2> nborDOFs(subFill.nborDOFs.data(), eNN, numDOFs);

}




