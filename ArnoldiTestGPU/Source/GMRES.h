//WARNING: declaration of GMRES is defined in file "Matrix_Vector_emulator.h" due to cross declaration!
//In this project this file is void

#ifndef __H_GMRES_H__
#define __H_GMRES_H__

#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include "Macros.h"
#include "cuda_supp.h"
#include "Products.h"
#include "memory_operations.h"
#include "Matrix_Vector_emulator.h"
#include "LAPACK_routines.h"


int GMRES(cublasHandle_t handle, int N, user_map_vector Axb, void *user_struct, real *x, real* RHS, real *tol, int *basis_size, int restarts, bool verbose, unsigned int skip=100, real machine_epsilon_provided=-1, real *residual_history=NULL);
    



#endif