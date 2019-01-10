//#include "GMRES.h"
#include "Matrix_Vector_emulator.h"
#include "LAPACK_routines.h"

void get_residualG(cublasHandle_t handle, int N, user_map_vector Axb, void *user_struct, real *source, real *RHS, real* r)
{
    
    Axb(user_struct, source, r);
    Arnoldi::vectors_add_GPU(handle, N, -1.0, RHS, r);
    Arnoldi::set_vector_inverce_GPU(N, r);
    
}

real get_errorG(cublasHandle_t handle,int N, real *r, real RHSnorm)
{
    return Arnoldi::vector_norm2_GPU(handle, N, r)/RHSnorm;
}

    
void rotmat(const real a, const real b, real *c, real *s)
{
    //Givens rotations through hypotenuse
    if(b==0.0){
        c[0]=1.0;
        s[0]=0.0;
    }
    else if(fabsf(b)>fabsf(a)){
        real temp=a/b;
        s[0]=1.0/sqrt(1.0+temp*temp);
        c[0]=temp*s[0];
    }
    else{
        real temp=b/a;
        c[0]=1.0/sqrt(1.0+temp*temp);
        s[0]=temp*c[0];        
    }

}


int GMRES(cublasHandle_t handle, int N, user_map_vector Axb, void *user_struct, user_map_vector Precond, void *user_struct_precond, real *x, real* RHS, real *tol, int *basis_size, int restarts, bool verbose) //, unsigned int skip, real machine_epsilon_provided, real *residual_history
{

    int flag = 1; //termiantion flag
    real error = 1.0; //residual
    real tolerance = tol[0];
    int Krylov_size = basis_size[0];
    real machine_epsilon_provided=-1;
    real machine_epsilon=machine_epsilon_provided;
    if(machine_epsilon<=0.0){
        machine_epsilon=Arnoldi::get_machine_epsilon();
    }

    // host arrays:
    real *H_h, *cs_h, *sn_h, *e1_h, *y_h;
    //  GPU arrays:
    real *r_d, *r_tilde_d, *V_d, *H_d,  *e1_d, *vl_d, *w_d, *y_d;
    //set all initial values for arrays
    //    V(1:n,1:m+1) = zeros(n,m+1);
    //    H(1:m+1,1:m) = zeros(m+1,m);
    //    cs(1:m) = zeros(m,1);
    //    sn(1:m) = zeros(m,1);
    //    e1    = zeros(n,1);
    //    e1(1) = 1.0;

    //host allocation
    Arnoldi::allocate_real((Krylov_size+1)*Krylov_size,1,&H_h);
    Arnoldi::allocate_real((Krylov_size+1),4,&cs_h, &sn_h, &e1_h, &y_h);


    //device allocation
    Arnoldi::device_allocate_all_real(N, (Krylov_size+1),1, 1, &V_d);
    Arnoldi::device_allocate_all_real((Krylov_size+1), (Krylov_size), 1, 1, &H_d);
    Arnoldi::device_allocate_all_real((Krylov_size+1), 1, 1, 1, &y_d);
    //Arnoldi::device_allocate_all_real((Krylov_size+1), 1, 1, 2, &cs, &sn);
    Arnoldi::device_allocate_all_real(N,1,1, 5, &e1_d, &r_d, &r_tilde_d, &vl_d, &w_d);
    for(int j=0;j<=Krylov_size;j++){
        Arnoldi::set_vector_value_GPU(N, 0.0, &V_d[j*N]);
        Arnoldi::set_vector_value_GPU(Krylov_size, 0.0, &H_d[j*Krylov_size]);
        e1_h[j]=0.0;
        y_h[j]=0.0;
        for(int k=0;k<Krylov_size;k++)
            H_h[I2(j, k, Krylov_size)]=0;
    }
    Arnoldi::set_vector_value_GPU(N, 0.0, e1_d);
    Arnoldi::set_vector_value_GPU(1, (real)1.0, &e1_d[0]);
    e1_h[0]=(real)1.0;

    real bnrm = Arnoldi::vector_norm2_GPU(handle, N, RHS);  
    if(bnrm < machine_epsilon){
        if(verbose)
            printf( "||b||_2 <%le! assuming ||b||_2=1.0\n", (double) machine_epsilon);
        bnrm = 1.0;
    }
    real vn = Arnoldi::vector_norm2_GPU(handle, N, x);
    if(verbose)
        printf("\n||b||_2=%le, ||x0||_2=%le\n",(double)bnrm, (double)vn);   
    
    get_residualG(handle, N, Axb, user_struct, x, RHS, &r_tilde_d[0]);
    Precond(user_struct_precond, r_tilde_d, r_d);
    //put preconditioner here!
    error = get_errorG(handle, N, &r_d[0], bnrm);
    if(error < tolerance){
        flag = 0;       
    }
    else{
        printf("\ninitial residual =%le\n",(double)error); 
    }
    if(isnan(error)){
        fprintf(stderr,"\nGMRES: Nans in user defined function!\n");
        flag = -3;
    }
    if(flag==1)
    for(int iter=0;iter<restarts;iter++){
        get_residualG(handle, N, Axb, user_struct, x, RHS, &r_d[0]);
//       z = ( b-A*x );
//       r = iM*z;         //put preconditioner here!
        Precond(user_struct_precond, r_d, r_tilde_d);
        //Arnoldi::vector_copy_GPU(handle, N, &r_d[0], r_tilde_d);
        real r_norm = Arnoldi::vector_norm2_GPU(handle, N, r_tilde_d);
        Arnoldi::normalize_vector_GPU(handle, N, r_tilde_d); 

        Arnoldi::set_matrix_colomn_GPU(N, Krylov_size, V_d, r_tilde_d, 0);  //       V(:,1) = r / norm( r );
        Arnoldi::set_vector_value_GPU(1, r_norm, &e1_d[0]);               //       s = norm( r )*e1;
        e1_h[0]=r_norm;

//       for i = 1:m                                   % construct orthonormal basis using Gram-Schmidt
        for(int i=0;i<Krylov_size;i++){
//          global_iter=global_iter+1;
            Arnoldi::vector_copy_GPU(handle, N, &V_d[i*N], vl_d); //         vl=V(:,i);
            Axb(user_struct, vl_d, r_d);           //         z = A*vl;
//          w = iM * z;                      //put preconditioner here!             
            Precond(user_struct_precond, r_d, w_d);
            for(int k=0;k<=i;k++){            //         for k = 1:i,
                Arnoldi::vector_copy_GPU(handle, N, &V_d[k*N], vl_d);//             vl=V(:,k);
                real alpha = Arnoldi::vector_dot_product_GPU(handle, N, w_d, vl_d); //              alpha=w'*vl;
                if(isnan(alpha)){
                    fprintf(stderr,"\n   GMRES: Nans in basis construciton!\n");
                    flag = -3;
                    break;
                }
                Arnoldi::vectors_add_GPU(handle, N, -alpha, vl_d, w_d);//             w = w - alpha*vl;
                real c_norm = 1.0;//             c=1;
                int orth_it = 0;//             orth_it=0;
                while(c_norm>1000*machine_epsilon){     //             while (norm(c)>100*machine_epsilon)
                    real c = Arnoldi::vector_dot_product_GPU(handle, N, w_d, vl_d); //                  c = w'*vl;
                    c_norm = fabsf(c);
                    Arnoldi::vectors_add_GPU(handle, N, -c, vl_d, w_d);//                  w=w-c.*vl;
                    alpha = alpha + c;//                  alpha=alpha+c;
                    if(orth_it>10){//                  if orth_it>10
                        fprintf(stderr,"\n    Gram-Schmidt orthogonalization error at %i, %le\n",i, (double)fabsf(c));
                        //                     str=sprintf('Arnoldi orthogonalization error at %i, %e',i,norm(c));
                        //                     disp(str);
                        flag = -2;
                        break;          //                     break;
                    }
//                  end;
                    orth_it++;//                  orth_it=orth_it+1;
//              end
                }
            
                H_h[I2(k, i, Krylov_size)] = alpha; // H[k,i]=alpha
            }
//          end
            real w_norm = Arnoldi::vector_norm2_GPU(handle, N, w_d);
            H_h[I2(i+1, i, Krylov_size)] = w_norm;    //      H(i+1,i) = norm( w );
            Arnoldi::normalize_vector_GPU(handle, N, w_d);  
            Arnoldi::set_matrix_colomn_GPU(N, (Krylov_size+1), V_d, w_d, i+1);  //V(:,i+1) = w / norm( w );
            
            //stoped here! think on how to implement rotations on a GPU!
            for(int k=0;k<i;k++){ //      for k = 1:i-1,                              % apply Givens rotation
                real temp = cs_h[k]*H_h[I2(k, i, Krylov_size)] + sn_h[k]*H_h[I2(k+1, i, Krylov_size)];
                H_h[I2(k+1, i, Krylov_size)] = -sn_h[k]*H_h[I2(k, i, Krylov_size)] + cs_h[k]*H_h[I2(k+1, i, Krylov_size)];
                H_h[I2(k, i, Krylov_size)] = temp;
                //      end
            }

//PUT cublasDrotg here
//            constructGivensRotationMatrix_GPU(handle, &H_d[HIndex(i,i)], &H_d[HIndex(i+1,i)], &cs_d[i], &sn_d[i]);

        //[cs(i),sn(i)] = rotmat( H(i,i), H(i+1,i) ); % form i-th rotation matrix
            rotmat( H_h[I2(i, i, Krylov_size)], H_h[I2(i+1, i, Krylov_size)], &cs_h[i], &sn_h[i]); 
//          H(i,i) = cs(i)*H(i,i) + sn(i)*H(i+1,i);
            H_h[I2(i, i, Krylov_size)] = cs_h[i]*H_h[I2(i, i, Krylov_size)] + sn_h[i]*H_h[I2(i+1, i, Krylov_size)];  
//          H(i+1,i) = 0.0;
            H_h[I2(i+1, i, Krylov_size)] = 0.0;

//          temp   = cs(i)*s(i);                        % approximate residual norm
            real temp = cs_h[i]*e1_h[i];
//          s(i+1) = -sn(i)*s(i);
            e1_h[i+1] = -sn_h[i]*e1_h[i];
//          s(i)   = temp;
            e1_h[i] = temp;

//          error  = abs(s(i+1)) / bnrm2;
//          resid(global_iter,1)=error;
//          if ( error <= tol ),                        % update approximation
//              y = H(1:i,1:i) \ s(1:i);                 % and exit
//              x = x + V(:,1:i)*y;
//              break;
//          end
            error = fabsf(e1_h[i+1])/bnrm;
            if(error <= tolerance){
                real *subH=(real*)malloc(sizeof(real)*(i+1)*(i+1));
                if ( !subH ){
                    fprintf(stderr,"\n unable to allocate real memeory for subH!\n");
                    return -4;
                }
                for(int j=0;j<=i;j++){
                    y_h[j]=e1_h[j];
                    for(int k=0;k<=i;k++){
                        subH[I2(j,k,(i+1))]=H_h[I2(j,k,Krylov_size)];
                    }
                }
                solve_upper_triangular_system((i+1), (i+1), subH, y_h);
                free(subH);
                
                Arnoldi::to_device_from_host_real_cpy(y_d, y_h, Krylov_size, 1, 1); 
                Arnoldi::matrixMultVector_part_GPU(handle, N, V_d, (Krylov_size+1), 1.0, y_d, (i+1), 1.0, x);
                flag=0;
                basis_size[0]=(i)*(iter+1);
                break;
            }
        }

        if( error <= tolerance ){
            flag=0;
            break;
        }
        for(int j=0;j<Krylov_size;j++){
            y_h[j]=e1_h[j];
        }
        solve_upper_triangular_system(Krylov_size, Krylov_size, H_h, y_h);
//       y = H(1:m,1:m) \ s(1:m);
//       x = x + V(:,1:m)*y;                            % update approximation
        Arnoldi::to_device_from_host_real_cpy(y_d, y_h, (Krylov_size+1), 1, 1); 
        Arnoldi::matrixMultVector_part_GPU(handle, N, V_d, (Krylov_size+1), 1.0, y_d, Krylov_size, 1.0, x);        
//       z = ( b-A*x );                              % compute residual
//       r = iM*z;        
        get_residualG(handle, N, Axb, user_struct, x, RHS, &r_d[0]);
        //Put preconditioner here!
        Precond(user_struct_precond, r_d, r_tilde_d);
        //Arnoldi::vector_copy_GPU(handle, N, &r_d[0], r_tilde_d);

//       s(i+1) = norm(r);
        e1_h[Krylov_size] = Arnoldi::vector_norm2_GPU(handle, N, r_tilde_d);
        error=e1_h[Krylov_size]/bnrm;
//       error = s(i+1) / bnrm2;                        % check convergence
        if ( error <= tolerance ){
            flag=0;
            basis_size[0]=(Krylov_size+1)*(iter+1);            
            break;
        }
    }
//    iter=iter*i;
    // if ( error > tolerance )
    //     flag = 1; 
    if(flag == 1){
        basis_size[0]=(Krylov_size)*(restarts); 
    }
    // cudaError_t cuerr=cudaDeviceSynchronize();
    // if (cuerr != cudaSuccess)
    // {
    //     fprintf(stderr,"cudaDeviceSybc failed: %s\n",
    //     cudaGetErrorString(cuerr));
    // } 

    //free device data
    Arnoldi::device_deallocate_all_real(8, r_d, r_tilde_d, V_d, H_d, e1_d, vl_d, w_d, y_d);
 

    //free host data
    Arnoldi::deallocate_real(5, H_h, cs_h, sn_h, e1_h, y_h);
    
    tol[0]=error;

    return flag;
}