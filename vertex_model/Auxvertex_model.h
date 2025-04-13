/*
    *******************************************************************************
    Copyright (c) 2017-2021 Universitat Politècnica de Catalunya
    Authors: Daniel Santos-Olivan, Alejandro Torres-Sanchez and Guillermo Vilanova
    Contributors:
    *******************************************************************************
    This file is part of hiperlife - High Performance Library for Finite Elements
    Project homepage: https://git.lacan.upc.edu/HPLFEgroup/hiperlifelib.git
    Distributed under the MIT software license, see the accompanying
    file LICENSE or http://www.opensource.org/licenses/mit-license.php.
    *******************************************************************************
*/


#include <iostream>
// Include necessary headers for mesh handling, problem definition, mathematical operations, etc.
#include "hl_DistributedMesh.h"
#include "hl_FillStructure.h"
#include "hl_DOFsHandler.h"
#include "hl_HiPerProblem.h"
#include "hl_Tensor.h"
#include "hl_Math.h"
#include "hl_Array.h"

using namespace std;
using namespace hiperlife; // Assuming hiperlife is a namespace for the simulation framework

using Teuchos::rcp;
using Teuchos::RCP;

// Forward declarations for functions that presumably fill structures with specific simulation settings or equations
void LS(hiperlife::FillStructure& fillStr);
void LS_Border(hiperlife::FillStructure& fillStr);



