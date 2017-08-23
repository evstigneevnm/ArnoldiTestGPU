#ifndef __ARNOLDI_Implicit_restart_Arnoldi_H__
#define __ARNOLDI_Implicit_restart_Arnoldi_H__

#include "Macros.h"
#include "timer.h"
#include "cuda_supp.h"
#include "memory_operations.h"
#include "Arnoldi_Driver.h"
#include "Select_Shifts.h"
#include "QR_Shifts.h"
#include "Matrix_Vector_emulator.h"
#include "file_operations.h"



real Implicit_restart_Arnoldi_GPU_data(bool verbose, int N, user_map_vector Axb, void *user_struct, real *vec_f_d, char which[2], int k, int m, real complex* eigenvaluesA, real tol, int max_iter, real *eigenvectors_real=NULL, real *eigenvectors_imag=NULL);






#endif