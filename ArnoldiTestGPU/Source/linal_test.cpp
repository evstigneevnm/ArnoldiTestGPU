#include "Macros.h"
#include "cuda_supp.h"
#include "memory_operations.h"
#include "file_operations.h"
#include "LAPACK_routines.h"

//#include "GMRES.h"
#include "Matrix_Vector_emulator.h"




real check_residual(cublasHandle_t handle, int N, user_map_vector Axb, void *user_struct, real *source, real *RHS)
{
    real* r_d;

    Arnoldi::device_allocate_all_real(N,1, 1, 1,&r_d);
    Axb(user_struct, source, r_d);
    Arnoldi::vectors_add_GPU(handle, N, -1.0, RHS, r_d);
    real RHSnorm = Arnoldi::vector_norm2_GPU(handle, N, RHS);
    real residual = Arnoldi::vector_norm2_GPU(handle, N, r_d)/RHSnorm; 
    Arnoldi::device_deallocate_all_real(1, r_d);
    return residual;

}



int main(int argc, char const *argv[])
{
    int KS=3;
    real *H=new real[(KS+1)*KS];
    for(int j=0;j<=KS;j++)
        for(int k=0;k<KS;k++)
            H[I2(j,k,KS)]=0.0;

    real *b=new real[KS+1];
    real *x=new real[KS+1];
    
    b[0]=1.0;
    b[1]=2.0;
    b[2]=4.0;
    
    H[I2(0,0,KS)]=1.0; H[I2(0,1,KS)]=1.0; H[I2(0,2,KS)]=2.0;
                       H[I2(1,1,KS)]=1.0; H[I2(1,2,KS)]=3.0;
                                          H[I2(2,2,KS)]=3.0;
                                          H[I2(3,2,KS)]=-1.0;

    real *subH=new real[(KS-1)*(KS-1)];
    
    for(int j=0;j<KS-1;j++)
        for(int k=0;k<KS-1;k++){
            subH[I2(j,k,KS-1)]=H[I2(j,k,KS)];
        }

    solve_upper_triangular_system(KS, KS, H, b);
    

    printf("done triang. solve\n");
    
    int N;
    real *A_h, *b_h, *x_h, *P_h;
    real *A_d, *b_d, *x_d, *P_d;

    char *matrix_file_name="A.dat";
    char *rhs_file_name="b.dat";
    char *x0_file_name="x0.dat";
    char *preconditioner_file_name="iP.dat";
    N = read_matrix_size(matrix_file_name);
    printf("matrix size = %i\n", N);
    
    Arnoldi::allocate_real(N,N,1,2,&A_h,&P_h);
    Arnoldi::allocate_real(N,1,1,1,&b_h);
    Arnoldi::allocate_real(N,1,1,1,&x_h);


//  init arays
    read_matrix(preconditioner_file_name, N, N, P_h);
    read_matrix(matrix_file_name, N, N, A_h);
    read_vector(rhs_file_name, N, b_h);
    read_vector(x0_file_name, N, x_h);

//  init cuda and copy arrays to device from host
    if(!Arnoldi::InitCUDA(3)) {
        return 0;
    }
    Arnoldi::device_allocate_all_real(N,N, 1, 2,&A_d, &P_d);
    Arnoldi::to_device_from_host_real_cpy(A_d, A_h, N, N, 1);
    Arnoldi::to_device_from_host_real_cpy(P_d, P_h, N, N, 1);
    Arnoldi::device_allocate_all_real(N,1,1, 1, &b_d);
    Arnoldi::to_device_from_host_real_cpy(b_d, b_h, N, 1,1);
    Arnoldi::device_allocate_all_real(N,1,1, 1, &x_d);
    Arnoldi::to_device_from_host_real_cpy(x_d, x_h, N, 1,1);

//  init cublas
    cublasHandle_t handle;      //init cublas
    cublasStatus_t ret;
    ret = cublasCreate(&handle);
    Arnoldi::checkError(ret, " cublasCreate(). ");

//  init structures
    Ax_struct_1 *SC_matmul=new Ax_struct_1[1];
    SC_matmul->N=N;
    SC_matmul->A_d=A_d;
    SC_matmul->handle=handle;

    Ax_struct *SC_precond_void=new Ax_struct[1];
    SC_precond_void->N=N;
    
    Ax_struct_1 *SC_precond=new Ax_struct_1[1];
    SC_precond->N=N;
    SC_precond->A_d=P_d;
    SC_precond->handle=handle;


    real tollerance=1.0e-8;
    int basis=20;
    int iterations=15;
    int flag =  GMRES(handle, N, (user_map_vector) user_Ax_function_1, (Ax_struct_1 *) SC_matmul, (user_map_vector) user_Preconditioner_function, (Ax_struct_1 *) SC_precond, x_d, b_d, &tollerance, &basis, iterations, true);

    printf("\nTermination flag=%i, achived tolerance=%le, total nuber of iterations=%i\n", flag, tollerance, basis);
    //copy back to host and delete device arrays
    
    //check for real residual
    real actual_residual = check_residual(handle, N, (user_map_vector) user_Ax_function_1, (Ax_struct_1 *) SC_matmul, x_d, b_d);
    printf("Actual residual is %le\n", (double) actual_residual);

    delete [] SC_matmul;   
    delete [] SC_precond_void;
    delete [] SC_precond;
    cublasDestroy(handle);
    Arnoldi::to_host_from_device_real_cpy(A_h, A_d, N, N,1);
    Arnoldi::to_host_from_device_real_cpy(x_h, x_d, N, 1, 1);
    Arnoldi::device_deallocate_all_real(4, A_d, b_d, x_d, P_d);

    //print resuts, delete host arrays and exit
    //print_matrix("A1.dat", N, N, A_h);
    //print_vector("b1.dat", N, b_h);
    print_vector("x.dat", N, x_h);
    Arnoldi::deallocate_real(4, A_h, b_h, x_h, P_h);
    return 0;
    
}