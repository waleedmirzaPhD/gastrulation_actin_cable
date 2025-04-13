/// C++ headers
#include <iostream>
#include <mpi.h>
#include <fstream>

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

inline static double Δt = 1.E-6;
// Define global variables
double  alpha_1 = 0;
double  alpha_2 = 0;
double  alpha_3 = 0;
int flag_struct = 0;

std::pair<double, double> fitExponentialDecay(const std::vector<double>& x, const std::vector<double>& v) {
    if (x.size() != v.size() || x.size() < 2) {
        throw std::invalid_argument("Vectors x and v must have the same size and contain at least two elements.");
    }

    // Find the maximum velocity (v0) and its index
    auto max_iter = std::max_element(v.begin(), v.end());
    double v0 = *max_iter;
    if (v0 <= 0) {
        throw std::invalid_argument("Maximum velocity (v0) must be positive.");
    }
    size_t max_index = std::distance(v.begin(), max_iter);
    double x_max = x[max_index];

    // Use only points where x[i] >= x[max_index], normalize x to start at 0 for v0
    double weighted_sum = 0.0;
    double log_sum = 0.0;

    for (size_t i = max_index; i < x.size(); ++i) {
        double log_ratio = std::log(v[i] / v0);
        double normalized_x = x[i] - x_max; // Normalize x
        weighted_sum += normalized_x * log_ratio;
        log_sum += log_ratio;
    }

    // Compute lambda as -weighted_sum / log_sum
    double lambda = -weighted_sum / log_sum;
    return {v0, lambda};
}



// Function to initialize global variables
void initialize_globals(int argc, char *argv[]) 
{
    alpha_1 = std::stod(argv[1]);
    alpha_3 = std::stod(argv[2]);
    alpha_2 = std::stod(argv[3]);;
}

void ElemFilling(hiperlife::FillStructure& fillStr)
{
    using namespace hiperlife;
    using namespace hiperlife::Tensor;
    using namespace autodiff;
    using namespace hiperlife;
    using ttl::tensor, ttl::wrapper, ttl::unsorted_wrapper, ttl::vector, ttl::matrix;
    using ttl::all, ttl::range;
    using ttl::index::i, ttl::index::j, ttl::index::k, ttl::index::l;
    using ttl::index::I, ttl::index::J;



    // Define Rayleighian = dEnergy(Neohookean) + dEnergy(body force)
    auto residual = [&fillStr](std::vector<dual> nborDOFs_vec) -> std::vector<dual>
    {
        double τ_a = 1; 
        double λ = 1.0; // 
        double diff = 2E-4; // --> keep it to the smallest values

        // double alpha_1 = 0.8;    //2, 0.,2    (2,3,4,5 /0.25,0.5,0.75,0.9/)    (3,1,2)  (2,2,1)
        // double alpha_2 = 0.0001; // ---> this exist is the non-dimensional maxwell time
        // double alpha_3 = 0.25;   //--->propotional to activty

        // These non-dim parameters are postprocessing of the other non-dim parameters
        double visc = alpha_1*alpha_1/alpha_3;   
        double γ = alpha_3; 
        double ξ =   0.0;
        double flag = 0.;
        int flag_deactivate = 0 ; 
        
        SubFillStructure& subFill = fillStr[0];
        int eNN  = subFill.eNN;       
        int pDim = subFill.pDim;      
        int numDOFs = subFill.numDOFs;
        wrapper<dual,2> nborDOFs(nborDOFs_vec.data(),eNN,numDOFs);
        wrapper<double,2> nborDOFs0(subFill.nborDOFs0.data(),eNN,numDOFs);

        unsorted_wrapper<dual,1> nborh = nborDOFs(all,0);
        unsorted_wrapper<dual,2> nborv = nborDOFs(all,range(1,2));
        unsorted_wrapper<dual,2> nbors = nborDOFs(all,range(3,5));
        unsorted_wrapper<double,1> nborh0 = nborDOFs0(all,0);
        unsorted_wrapper<double,2> nbors0 = nborDOFs0(all,range(3,5));

        wrapper<double, 1> bf(subFill.nborBFs(), eNN);
        tensor<double, 2> Dbf(eNN, pDim);
        double jac;
        GlobalBasisFunctions::gradients(Dbf, jac, subFill);

        std::vector<dual> res(eNN*numDOFs);
        wrapper<dual,2> Bk(res.data(), eNN, numDOFs);

        unsorted_wrapper<dual,1> Bh = Bk(all,0);
        unsorted_wrapper<dual,2> Bv = Bk(all,range(1,2));
        unsorted_wrapper<dual,2> Bs = Bk(all,range(3,5));

        dual h = nborh(I) * bf(I);
        double h0 = nborh0(I) * bf(I);
        vector<dual> Dh = nborh(I) * Dbf(I,i);
        vector<dual> v = bf(I) * nborv(I,i);
        matrix<dual> Dv = nborv(I,i) * Dbf(I,j);
        matrix<dual> ω = 0.5*(Dv-Dv.T());
        matrix<dual> d = 0.5*(Dv+Dv.T());
        dual div = Dv(i,i);
        tensor<double,3> voigt = {{{1.0,0.0},{0.0,0.0}},{{0.0,0.0},{0.0,1.0}},{{0.0,1.0},{1.0,0.0}}};
        matrix<dual> s = (bf(I) * nbors(I,k)) * voigt(k,i,j);
        matrix<double> s0 = (bf(I) * nbors0(I,k)) * voigt(k,i,j);
        tensor<dual,3> Ds = ((Dbf(I,l) * nbors(I,k)) * voigt(k,i,j))(i,j,l);

        dual resh = h - h0 + Δt * ((h-1)/τ_a + Dh(i) * v(i)  +  h * div );
        Bh(I) = jac * (bf(I) * resh + Δt * diff * Dbf(I,i) * Dh(i));
        Bv(I,i) = jac * (γ * bf(I) * v(i) + Dbf(I,j) * (s(i,j) + λ * (h * Identity2(i,j)))); 
        Bs(I,l) = jac * bf(I) * voigt(l,i,j) * (s(i,j) + alpha_2 * ((s(i,j) - s0(i,j))/Δt + Ds(i,j,k) * v(k) - flag*s(i,k) * (ω(j,k) + ξ * d(j,k)) - flag*s(k,j) * (ω(i,k) + ξ * d(i,k)) ) - visc * h * Dv(i,j) - visc * h *Dv(j,i));

        return res;
    };

    // Compute nborDOFs as dual type
    SubFillStructure& subFill = fillStr["dofHand"];
    int eNN  = subFill.eNN;
    int numDOFs = subFill.numDOFs;
    std::vector<dual> dofs(eNN*numDOFs);
    for (int i = 0; i < eNN*numDOFs; i++)
        dofs[i] = subFill.nborDOFs[i];

    // Compute rayleighian, jacobian, and hessian
    std::vector<dual> v(eNN*numDOFs);
    jacobian(residual, wrt(dofs), at(dofs), v, fillStr.Bk(0), fillStr.Ak(0,0));

    wrapper<double,4> Ak(fillStr.Ak(0, 0).data(), eNN, numDOFs, eNN, numDOFs);
}

int main(int argc, char** argv)
{
    using namespace hiperlife;

    hiperlife::Init(argc, argv);
    initialize_globals(argc, argv);

    cout<<alpha_1<<" "<<alpha_2<<" "<<alpha_3<<endl;
    cout<<alpha_1*alpha_1/alpha_3<<endl;
    /// **************************************************************//
    /// *****                   MESH CREATION                    *****//
    /// **************************************************************//

    BasisFuncType bfType = BasisFuncType::Linear;
    ElemType eType = ElemType::Triang;
    int bfOrder = 1;
    int nGauss = 3;
    int nEl = 10000; 
    double L =12;
    std::string fname = "data/data_1";
    SmartPtr<StructMeshGenerator> structMesh = Create<StructMeshGenerator>();
    structMesh->setMesh(eType, bfType, bfOrder);
    structMesh->setPeriodicBoundaryCondition({Axis::Yaxis});
    structMesh->genRectangle(nEl, 1, L,0.001);                            
    structMesh->translateX(-0.5*L);
    structMesh->translateY(-0.5);

    Teuchos::RCP<MeshLoader> loadedMesh = Teuchos::rcp(new MeshLoader);
    loadedMesh->setElemType(ElemType::Triang);
    loadedMesh->setBasisFuncType(BasisFuncType::Lagrangian);
    loadedMesh->setBasisFuncOrder(bfOrder);
    loadedMesh->loadMesh("cell_ablation.vtk",MeshType::Parallel);


    ///===========            Distribute mesh            ===========///
    SmartPtr<DistributedMesh> disMesh = Create<DistributedMesh>();
    if (!flag_struct)
        disMesh->setMesh(loadedMesh);
    else
        disMesh->setMesh(structMesh);    
    disMesh->setBalanceMesh(true);
    disMesh->Update();

    if (disMesh->myRank() == 0)
        std::cout << "The parameters are " << alpha_1<<" "<<alpha_3 << std::endl;

    SmartPtr<DistributedMesh> disMesh2 = disMesh;
    for (int i = 0; i < disMesh->loc_nPts(); i++)
    {
        double x = disMesh->nodeCoord(i, 0, IndexType::Local);
        double y = disMesh->nodeCoord(i, 1, IndexType::Local);
        disMesh->_nodeData->setValue(0,i,IndexType::Local,x);
    }    

    /// **************************************************************//
    /// *****               DOFHANDLER CREATION                  *****//
    /// **************************************************************//

    ///===========           Create DOFHandler            ===========///
    SmartPtr<DOFsHandler> dofHand = Create<DOFsHandler>(disMesh);
    dofHand->setNameTag("dofHand");
    dofHand->setDOFs({"h","vx","vy","sxx","syy","sxy"});
    dofHand->setNodeAuxF({"time"});
    dofHand->Update();

    dofHand->nodeDOFs0->setValue("h",1.);
    std::vector<double> local_nodes; 
    std::vector<std::pair<double, double>> x_coord; 
    local_nodes.resize(0);
    if (!flag_struct)
    {      
        for(int i = 0; i < disMesh->loc_nPts(); i++)
        {
            auto x = disMesh->nodeCoords(i, IndexType::Local);
            double width_x = 0.1;
            double width_y = 1.;
            double slope   = 50;
            // Using tanh function for smooth transition
            double h0 = 1 - 0.2495 * (1 + tanh(slope * (width_x - abs(x[0])))) * (1 + tanh(slope * (width_y - abs(x[1]))));
            //double h0 =  0.5*(1 - tanh(slope * (width_x - abs(x[0]))));
            dofHand->nodeDOFs0->setValue("h", i, IndexType::Local, h0);
            dofHand->nodeDOFs->setValue("h", i, IndexType::Local, h0);
        }
        dofHand->nodeDOFs->setValue(dofHand->nodeDOFs0);
        dofHand->setBoundaryCondition("h",0.0);
        dofHand->setBoundaryCondition("vx",0.0);
        dofHand->setBoundaryCondition("vy",0.0);
        // dofHand->setConstraint("vy", 0.0);  
        // dofHand->setConstraint("sxx", 0.0);          
        // dofHand->setConstraint("sxy", 0.0);     
        // dofHand->setConstraint("syy", 0.0);     
    }
    else
    {
        for(int i = 0; i < disMesh->loc_nPts(); i++)
        {
            auto x = disMesh->nodeCoords(i, IndexType::Local);
            double width_x = 0.1;
            double width_y = 1;
            double slope   =50;
            // Using tanh function for smooth transition
            double h0 = 1 - 0.248 * (1 + tanh(slope * (width_x - abs(x[0])))) * (1 + tanh(slope * (width_y - abs(x[1]))));
            dofHand->nodeDOFs0->setValue("h", i, IndexType::Local, h0);
            if (x[0]<-0.5*L + 1E-6 or x[0]>0.5*L - 1E-6)
            {
                dofHand->nodeDOFs->setValue("vx", i, IndexType::Local, 0); 
                dofHand->setConstraint("vx", i, IndexType::Local, 0.0);  
            }
        }
        dofHand->nodeDOFs->setValue("vy", 0); 
        dofHand->setConstraint("vy",  0.0);  
    }
    dofHand->UpdateGhosts();

    SmartPtr<HiPerProblem> hiperProbl = Create<HiPerProblem>();
    hiperProbl->setDOFsHandlers({dofHand});
    hiperProbl->setIntegration("Integ", {"dofHand"});
    hiperProbl->setElementFillings("Integ", ElemFilling);
    hiperProbl->setCubatureGauss("Integ", nGauss);
    hiperProbl->Update();

    SmartPtr<MUMPSDirectLinearSolver> linSolver = Create<MUMPSDirectLinearSolver>();
    linSolver->setHiPerProblem(hiperProbl);
    linSolver->setDefaultParameters();
    linSolver->Update();



    SmartPtr<AztecOOIterativeLinearSolver> linsolver = Create<AztecOOIterativeLinearSolver>();  //rcp ( new AztecOOIterativeLinearSolver());
    linsolver->setHiPerProblem(hiperProbl);
    linsolver->setTolerance(1e-5);
    linsolver->setMaxNumIterations(10000);
    linsolver->setSolver(AztecOOIterativeLinearSolver::Solver::Gmres);
    linsolver->setPrecond(AztecOOIterativeLinearSolver::Precond::DomainDecomp);
    linsolver->setSubdomainSolve(AztecOOIterativeLinearSolver::SubdomainSolve::Ilut);
    linsolver->setVerbosity(AztecOOIterativeLinearSolver::Verbosity::None);
    linsolver->setDefaultParameters();
    linsolver->Update();



    SmartPtr<NewtonRaphsonNonlinearSolver> nonlinSolver = Create<NewtonRaphsonNonlinearSolver>();
    nonlinSolver->setLinearSolver(linSolver);
    nonlinSolver->setMaxNumIterations(10);
    nonlinSolver->setResTolerance(1.E-8);
    nonlinSolver->setSolTolerance(1.E-8);
    nonlinSolver->setLineSearch(false);
    nonlinSolver->setPrintIntermInfo(true);
    nonlinSolver->setConvRelTolerance(false);
    nonlinSolver->Update();

    double time = 0.0;
    int timeStep = 0;
    double maxTime = 10000;
    int maxNStep = 1000000;
    double stepFactor = 0.9;
    double maxDelt = 0.3;
    int nSave = 1;
    string oname = "out_";
    dofHand->printFileVtk(oname + to_string(0), true,time);
    bool flag = false;
    double width = 0;
    std::vector<double> x_wound; 
    x_wound.push_back(0.1); 
    x_wound.push_back(0);
    std::vector<double> rho_wound; 
    rho_wound.push_back(0.0); 
    rho_wound.push_back(0);
    std::ofstream outFile("output.txt");  // Open a file to write the time and width
    std::ofstream outFile_rho("output_rho.txt");  // Open a file to write the time and width
/*
    std::ofstream outFile_v0("output_v0.txt");
    std::ofstream outFile_v1("output_v1.txt");
    std::ofstream outFile_v2("output_v2.txt");
    std::ofstream outFile_v3("output_v3.txt");
    std::ofstream outFile_v4("output_v4.txt");
    std::ofstream outFile_v5("output_v5.txt");
    std::ofstream outFile_v6("output_v6.txt");
    std::ofstream outFile_v7("output_v7.txt");
    std::ofstream outFile_v8("output_v8.txt");
    std::ofstream outFile_v9("output_v9.txt");
    std::ofstream outFile_v10("output_v10.txt");


    std::vector<std::ofstream> outFiles;
    for (int i = 0; i <= 10; ++i) {
        outFiles.emplace_back("output_v" + std::to_string(i) + ".txt");
    }
*/
    std::ofstream outFile_hydro("output_hydro.txt");

    double rho_in_wound = 0;
    double v0{};
	double lambda{};
    int num = 100;
    std::vector<double> xCoords(num);
    for (int i = 0; i < num; ++i)
    {
       xCoords[i]  = 0.1 +  i*0.1 ;
	}


    while ((timeStep < maxNStep) and (time < maxTime))
    {
        // Write
        if (dofHand->myRank() == 0)
            cout << endl << " TS: " << to_string(timeStep) << " Time " << time <<  " Delta " << Δt << endl;

        // Initial guess        
        dofHand->nodeDOFs->setValue(dofHand->nodeDOFs0);
        hiperProbl->UpdateGhosts();

        // Newton-Raphson method
        int ierr = nonlinSolver->solve();

        // cout << dofHand->myRank() << " " << min << " " << max << endl;
        // Check convergence
        if (ierr != 0)
        {
            // Save solution
            dofHand->nodeDOFs0->setValue(dofHand->nodeDOFs);

            // Update variables
            timeStep ++;

            dofHand->nodeAuxF->setValue("time", time);
            double v_wound_x =  dofHand->interpolateDOFs("vx", x_wound);
            double v_wound_y =  dofHand->interpolateDOFs("vy", x_wound);
			/*
			double v_0       =  dofHand->interpolateDOFs("vx", 0);
            double v_1       =  dofHand->interpolateDOFs("vx", delta_x);
			double v_2       =  dofHand->interpolateDOFs("vx", 2*delta_x); 
            double v_3       =  dofHand->interpolateDOFs("vx", 3*delta_x);
            double v_4       =  dofHand->interpolateDOFs("vx", 4*delta_x);
            double v_5       =  dofHand->interpolateDOFs("vx", 5*delta_x);
            double v_6       =  dofHand->interpolateDOFs("vx", 6*delta_x);
            double v_7       =  dofHand->interpolateDOFs("vx", 7*delta_x);
            double v_8       =  dofHand->interpolateDOFs("vx", 8*delta_x);
            double v_9       =  dofHand->interpolateDOFs("vx", 9*delta_x);
            double v_10      =  dofHand->interpolateDOFs("vx", 10*delta_x);
            
            std::vector<double> v_values(num,0);
            for (int i = 0; i < num; ++i) 
			{
			    for (int j = -10; j <= 10; ++j)
				{
				    double x_pos = xCoords[i] ; 
                	v_values[i] += dofHand->interpolateDOFs("vx", x_pos, j*0.1)/21.0;
				}
            }
            */

			std::vector<double> v_values(num,0);
			for (int i = 0; i < num; ++i) 
			{   
    			//for (int j = -10; j <= 10; ++j)
				   
        			double x_pos = xCoords[i]; 
	                // Ensure the correct type for "vx" and arguments
        			v_values[i] += dofHand->interpolateDOFs(std::string("vx"),{ xCoords[i], 0});
    			
				    //if (disMesh->myRank()==0) cout<< xCoords[i]<<" "<< 0 <<" "<< v_values[i]<<endl;  

			}
            rho_in_wound =  dofHand->interpolateDOFs("h", rho_wound);

            x_wound[0] += v_wound_x*Δt;
            x_wound[1] += v_wound_y*Δt;
            double v_wound_mag = sqrt(v_wound_x*v_wound_x + v_wound_y*v_wound_y); 

            width = width + v_wound_mag*Δt;


            if (timeStep % nSave == 0)
            {
                string solName = oname + to_string(timeStep);
                dofHand->printFileVtk(solName, true,time);
            }

          	//auto [v0, lambda] = fitExponentialDecay(xCoords, v_values);


            // Write time and width to the text file
            if (disMesh->myRank()==0  and outFile.is_open())
            {
                outFile << time << "," << width << std::endl;
				outFile_rho << time << "," << rho_in_wound << std::endl;
				//for (int i = 0; i <= num; ++i)

                outFile_hydro << time; // Write the time first
                for (int i = 0; i < num; ++i) { // Loop through the indices of v_values
                        outFile_hydro << "," << v_values[i]; // Append each value to the line
             }
             outFile_hydro << std::endl; // End the line
            if (rho_in_wound>0.98)
                break;






                /*
                outFile_v0  << time << "," << v0 << std::endl
                outFile_v1  << time << "," << v1 << std::endl;
                outFile_v2  << time << "," << v2 << std::endl;
                outFile_v3  << time << "," << v3 << std::endl;
				outFile_v4  << time << "," << v4 << std::endl;
                outFile_v5  << time << "," << v5 << std::endl;
				outFile_v6  << time << "," << v6 << std::endl;
				outFile_v7  << time << "," << v7 << std::endl;
                outFile_v8  << time << "," << v8 << std::endl;
				outFile_v9  << time << "," << v9 << std::endl;
                outFile_v10 << time << "," << v10<< std::endl;
                for (int i = 0; i <= 10; ++i) 
				{
                   outFiles[i] << time << "," << v_values[i] << std::endl;
                }
                */

               


            }


            if(nonlinSolver->numberOfIterations()<5)
                Δt /= stepFactor;
            if (Δt >= maxDelt)
                Δt = maxDelt;
            time += Δt;
        }
        else
        {
            // Correct time-step size
            Δt *= stepFactor;
        }

        // Check steady-state
        if (Δt < 1.e-10 )
            break;
    }
    hiperlife::Finalize();
    return 0;
}    

