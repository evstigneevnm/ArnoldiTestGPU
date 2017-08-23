#include "Implicit_restart_Arnoldi.h"



__global__ void real_to_cublasComplex_kernel(int N, real *vec_source_re, real *vec_source_im, cublasComplex *vec_dest){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
		vec_dest[i].x=vec_source_re[i];
		vec_dest[i].y=vec_source_im[i];
	}
	
}

__global__ void real_to_cublasComplex_kernel(int N, real *vec_source_re, cublasComplex *vec_dest){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
		vec_dest[i].x=vec_source_re[i];
		vec_dest[i].y=0.0;
	}
	
}

__global__ void cublasComplex_to_real_kernel(int N, cublasComplex *vec_source,  real *vec_dest_re,  real *vec_dest_im){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
		vec_dest_re[i]=vec_source[i].x;
		vec_dest_im[i]=vec_source[i].y;
	}
	
}

void real_complex_to_cublas_complex(int N, complex real* cpu_complex,  cublasComplex *gpu_complex){


	real *cpu_real, *cpu_imag, *gpu_real, *gpu_imag;
	Arnoldi::device_allocate_all_real(N,1, 1, 2,&gpu_real, &gpu_imag);
	Arnoldi::allocate_real(N, 1, 1, 2,&cpu_real, &cpu_imag);
	for(int j=0;j<N;j++){
		cpu_real[j]=creal(cpu_complex[j]);
		cpu_imag[j]=cimag(cpu_complex[j]);
	}
	Arnoldi::to_device_from_host_real_cpy(gpu_real, cpu_real, N, 1,1);
	Arnoldi::to_device_from_host_real_cpy(gpu_imag, cpu_imag, N, 1,1);

	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);

	real_to_cublasComplex_kernel<<< blocks, threads>>>(N, gpu_real, gpu_imag, gpu_complex);


	Arnoldi::deallocate_real(2,cpu_real, cpu_imag);
	Arnoldi::device_deallocate_all_real(2, gpu_real, gpu_imag);
}


void real_device_to_cublas_complex(int N, real* gpu_real, cublasComplex *gpu_complex){


	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);

	real_to_cublasComplex_kernel<<< blocks, threads>>>(N, gpu_real, gpu_complex);


}



void cublas_complex_to_complex_real(int N, cublasComplex *gpu_complex, complex real* cpu_complex){

	real *cpu_real, *cpu_imag, *gpu_real, *gpu_imag;
	Arnoldi::device_allocate_all_real(N,1, 1, 2,&gpu_real, &gpu_imag);
	Arnoldi::allocate_real(N, 1, 1, 2,&cpu_real, &cpu_imag);


	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	cublasComplex_to_real_kernel<<< blocks, threads>>>(N, gpu_complex,  gpu_real,  gpu_imag);

	Arnoldi::to_host_from_device_real_cpy(cpu_real, gpu_real, N, 1,1);
	Arnoldi::to_host_from_device_real_cpy(cpu_imag, gpu_imag, N, 1,1);

	for(int j=0;j<N;j++){
		cpu_complex[j]=cpu_real[j]+I*cpu_imag[j];
	}



	Arnoldi::deallocate_real(2,cpu_real, cpu_imag);
	Arnoldi::device_deallocate_all_real(2, gpu_real, gpu_imag);

}




void cublas_complex_to_device_real(int N, cublasComplex *gpu_complex, real* gpu_real, real* gpu_imag){


	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);

	cublasComplex_to_real_kernel<<< blocks, threads>>>(N, gpu_complex, gpu_real, gpu_imag);

}



__global__ void permute_matrix_colums_kernel(int MatrixRaw, int coloms, int *sorted_list_d, cublasComplex *vec_source,  cublasComplex *vec_dest){


	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<MatrixRaw){
	
		for(int j=0;j<coloms;j++){
			int index=sorted_list_d[j];
			vec_dest[I2(i,j,MatrixRaw)]=vec_source[I2(i,index,MatrixRaw)];
		}

	}
	
}



void permute_matrix_colums(int MatrixRaw, int coloms, int *sorted_list_d, cublasComplex *vec_source,  cublasComplex *vec_dest){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(MatrixRaw+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	permute_matrix_colums_kernel<<< blocks, threads>>>(MatrixRaw, coloms, sorted_list_d, vec_source,  vec_dest);

}



__global__ void  RHS_of_eigenproblem_real_device_kernel(int N, real lambda_real, real* Vec_real, real lambda_imag, real* Vec_imag, real *Vec_res){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
	
		Vec_res[i]=lambda_real*Vec_real[i]-lambda_imag*Vec_imag[i];
		

	}



}


__global__ void  RHS_of_eigenproblem_imag_device_kernel(int N, real lambda_real, real* Vec_real, real lambda_imag, real* Vec_imag, real *Vec_res){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
	
		Vec_res[i]=lambda_imag*Vec_real[i]+lambda_real*Vec_imag[i];
		

	}



}


void RHS_of_eigenproblem_device_real(int N, real lambda_real, real* Vec_real, real lambda_imag, real* Vec_imag, real *Vec_res){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);

	RHS_of_eigenproblem_real_device_kernel<<< blocks, threads>>>(N, lambda_real, Vec_real, lambda_imag, Vec_imag, Vec_res);

}

void RHS_of_eigenproblem_device_imag(int N, real lambda_real, real* Vec_real, real lambda_imag, real* Vec_imag, real *Vec_res){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);

	RHS_of_eigenproblem_imag_device_kernel<<< blocks, threads>>>(N, lambda_real, Vec_real, lambda_imag, Vec_imag, Vec_res);

}


__global__ void  Residual_eigenproblem_device_kernel(int N, real* Vl_r_d, real* Vr_r_d, real* Vec_res){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
	
		Vec_res[i]=Vl_r_d[i]-Vr_r_d[i];
		

	}

}

void Residual_eigenproblem_device(int N, real* Vl_r_d, real* Vr_r_d, real* Vre_d){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	
	Residual_eigenproblem_device_kernel<<< blocks, threads>>>(N, Vl_r_d, Vr_r_d, Vre_d);


}



//which: 
//		"LR" - largest real, "LM" - largest magnitude
//


real Implicit_restart_Arnoldi_GPU_data(bool verbose, int N, user_map_vector Axb, void *user_struct, real *vec_f_d, char which[2], int k, int m, real complex* eigenvaluesA, real tol, int max_iter, real *eigenvectors_real_d, real *eigenvectors_imag_d){



	real *vec_c=new real[m];
	real *vec_h=new real[m];
	real *vec_q=new real[m];
	real *H=new real[m*m];
	real *R=new real[m*m];
	real *Q=new real[m*m];
	real *H1=new real[m*m];
	real *H2=new real[m*m];
	matrixZero(m, m, H);
	matrixZero(m, m, R);
	matrixZero(m, m, Q);
	matrixZero(m, m, H1);
	matrixZero(m, m, H2);
	real complex *eigenvectorsH=new real complex[m*m];
	real complex *eigenvaluesH=new real complex[m*m];
	real complex *eigenvectorsH_kk=new real complex[k*k];
	real complex *eigenvaluesH_kk=new real complex[k*k];
	real *ritz_vector=new real[m];

	real *V_d, *V1_d, *Q_d; //matrixes on GPU
	real *vec_f1_d, *vec_v_d, *vec_w_d, *vec_c_d, *vec_h_d, *vec_q_d; //vectors on GPU
	real *Vl_r_d, *Vl_i_d, *Vr_r_d, *Vr_i_d, *Vre_d, *Vim_d; //vectors on GPU for eigenvector residuals
	//real *eigenvectors_real_d, *eigenvectors_imag_d;	//Matrix Eigenvectors
	bool external_eigenvectors=true;
	if(eigenvectors_real_d==NULL){
		external_eigenvectors=false;
		Arnoldi::device_allocate_all_real(N,k, 1, 2, &eigenvectors_real_d, &eigenvectors_imag_d);
	}

	Arnoldi::device_allocate_all_real(N,m, 1, 2, &V_d, &V1_d);
	Arnoldi::device_allocate_all_real(N, 1,1, 3, &vec_f1_d, &vec_w_d, &vec_v_d);
	Arnoldi::device_allocate_all_real(m, 1,1, 3, &vec_c_d, &vec_h_d, &vec_q_d);
	Arnoldi::device_allocate_all_real(m,m, 1, 1, &Q_d);
	Arnoldi::device_allocate_all_real(N, 1,1, 6, &Vl_r_d, &Vl_i_d, &Vr_r_d, &Vr_i_d, &Vre_d, &Vim_d);


	// Allocate memory for eigenvectors!
	cublasComplex *eigenvectorsH_d, *eigenvectorsA_d, *eigenvectorsA_unsorted_d;


	eigenvectorsH_d=Arnoldi::device_allocate_complex(k, k, 1);
	eigenvectorsA_d=Arnoldi::device_allocate_complex(N, k, 1);
	eigenvectorsA_unsorted_d=Arnoldi::device_allocate_complex(N, k, 1);


	cublasHandle_t handle;		//init cublas
	cublasStatus_t ret;
	ret = cublasCreate(&handle);
	Arnoldi::checkError(ret, " cublasCreate(). ");


	int k0=1;
	int iterations=0;
	real ritz_norm=1.0;
	timer_start();
	while(((iterations++)<max_iter)&&(ritz_norm>tol)){
	
		Arnoldi_driver(handle, N, Axb, user_struct, V_d, H, vec_f_d, k0-1, m, vec_v_d, vec_w_d, vec_c_d, vec_h_d, vec_h);	//Build orthogonal Krylov subspace
		

		select_shifts(m, H, which, eigenvectorsH, eigenvaluesH, ritz_vector); //select basisi shift depending on 'which'

		QR_shifts(k, m, Q, H, eigenvaluesH, &k0); //Do QR shifts of basis. Returns active eigenvalue indexes and Q-matrix for basis shift
		
		real vec_f_norm=Arnoldi::vector_norm2_GPU(handle, N, vec_f_d); 					
		for(int i=0;i<k0;i++){
			ritz_vector[i]=ritz_vector[i]*vec_f_norm;
		}
		get_matrix_colomn(m, m, Q, vec_q, k0);	
		real hl=H[I2(k0,k0-1,m)];
		real ql=Q[I2(m-1,k0-1,m)];
       	//f = V*vec_q*hl + f*ql;
		Arnoldi::to_device_from_host_real_cpy(vec_q_d, vec_q, m, 1,1); //vec_q -> vec_q_d
		Arnoldi::matrixMultVector_GPU(handle, N, V_d, m, hl, vec_q_d, ql, vec_f_d);
		//matrixMultVector(N, V, m, hl, vec_q, ql, vec_f1, vec_f);	//GG
		
		//fix this shit!!! V
		//we must apply Q only as matrix mXk0 on a submatrix  V NXm!!!
		for(int i=0;i<m;i++){
			for(int j=k0;j<m;j++){
				Q[I2(i,j,m)]=1.0*delta(i,j);
			}
		}
		//Copy matrixQtoGPUmemory!
		//here!
		Arnoldi::to_device_from_host_real_cpy(Q_d, Q, m, m, 1); //Q -> Q_d
		Arnoldi::matrixMultMatrix_GPU(handle, N, m, m, V_d, 1.0, Q_d, 0.0, V1_d);	//OK
		
		//matrix_copy(N, m, V1, V);									//GG
		Arnoldi::vector_copy_GPU(handle, N*m, V1_d, V_d);

		ritz_norm=vector_normC(k0,ritz_vector);
		if(verbose)
			printf("it=%i, ritz norm_C=%.05e \n", iterations, ritz_norm);
		else{
		//	if(iterations%50==0)
		//		printf("it=%i, ritz norm_C=%.05e \n", iterations, ritz_norm);
		}
	
	}
	timer_stop();
	timer_print();

	if(verbose)
		printf("\ncomputing eigenvectors...\n");
	//compute eigenvectors

	real complex *HC=new real complex[k*k];
	for(int i=0;i<k;i++){
		for(int j=0;j<k;j++){
			HC[I2(i,j,k)]=H[I2(i,j,m)]+0.0*I;
		}
	}

	MatrixComplexEigensystem(eigenvectorsH_kk, eigenvaluesH_kk, HC, k);
	
	delete [] HC;
	// 160720
	// this works in matlab: eigvsA=V*eigvsH, sort as desired (LR or LM).
	
	// Now store EigenvectorsH to GPU as cublasComplex.
	real_complex_to_cublas_complex(k*k, eigenvectorsH_kk,  eigenvectorsH_d);
	
	// Convert V as cublasComblex.
	// Multiply to get eigvsA1.
	real_device_to_cublas_complex(N*k, V_d, eigenvectorsA_d);
	Arnoldi::matrixMultComplexMatrix_GPU(handle, N, k, k, eigenvectorsA_d, eigenvectorsH_d, eigenvectorsA_unsorted_d);
	// sort eigsH as desired in list 'll' and shuffle as ll:eigvsA1->eigvsA.
	int *sorted_list=new int[k];
	int *sorted_list_d=Arnoldi::device_allocate_int(k, 1, 1);

	get_sorted_index(k, which, eigenvaluesH_kk, sorted_list);
	Arnoldi::to_device_from_host_int_cpy(sorted_list_d, sorted_list, k, 1, 1);
	permute_matrix_colums(N, k, sorted_list_d, eigenvectorsA_unsorted_d,  eigenvectorsA_d);

	cudaFree(sorted_list_d);
	delete [] sorted_list;
	sorted_list=new int[m];
	get_sorted_index(m, which,  eigenvaluesH, sorted_list);
	delete [] sorted_list;
	// perform residual estimation as:
	// for(int k=0;k<k0;k++){
	//  	norm_j=norm2(A*eigvsA(k)-eigsH(k)*eigvsA(k));
	//		printf("Residual: %e", norm_j);
	// }
	// convert to real complex on CPU and print matrix of eigvsA.
	real *residualAV=new real[k];
	for(int i=0;i<k;i++){
		residualAV[i]=-1;
	}
	cublas_complex_to_device_real(N*k, eigenvectorsA_d, eigenvectors_real_d, eigenvectors_imag_d);

	for(int i=0;i<k;i++){

		Arnoldi::get_matrix_colomn_GPU(N, k, eigenvectors_real_d, Vre_d, i);
		Arnoldi::get_matrix_colomn_GPU(N, k, eigenvectors_imag_d, Vim_d, i); //select coloms ->Vre_d;Vim_d;
		Axb(user_struct, Vre_d, Vl_r_d);				//LHS of the real eigenproblem
		Axb(user_struct, Vim_d, Vl_i_d);				//LHS of the imag eigenproblem
		real lambda_real=creal(eigenvaluesH_kk[i]);
		real lambda_imag=cimag(eigenvaluesH_kk[i]);
		RHS_of_eigenproblem_device_real(N, lambda_real, Vre_d, lambda_imag, Vim_d, Vr_r_d);
		RHS_of_eigenproblem_device_imag(N, lambda_real, Vre_d, lambda_imag, Vim_d, Vr_i_d);
		Residual_eigenproblem_device(N, Vl_r_d, Vr_r_d, Vre_d);
		Residual_eigenproblem_device(N, Vl_i_d, Vr_i_d, Vim_d);
		real norm2_real=Arnoldi::vector_norm2_GPU(handle, N, Vre_d);
		real norm2_imag=Arnoldi::vector_norm2_GPU(handle, N, Vim_d);
		residualAV[i]=sqrt(norm2_real*norm2_real+norm2_imag*norm2_imag);
	}


	
	cudaFree(eigenvectorsH_d);
	cudaFree(eigenvectorsA_unsorted_d);

	if(verbose)
		printf("done\n");



	printf("\nNumber of correct eigenvalues=%i Eigenvalues: \n", k0);
  	for(int i=0;i<k;i++){ 
  		real ritz_val=ritz_vector[i];
  		printf("\n (%.08le, %.08le), ritz: %.04le, residual: %.04le",  (double) creal(eigenvaluesH_kk[i]), (double) cimag(eigenvaluesH_kk[i]), (double)ritz_val, (double)residualAV[i] );
  	}
	printf("\n");
	delete [] residualAV;

	bool do_plot=true;
	if((verbose)&&(do_plot)){
		printf("plotting output matrixes and vectors...\n");
		real *vec_f_local=new real[N];
		real *V_local=new real[N*m];
		real *V1_local=new real[N*m];
		Arnoldi::to_host_from_device_real_cpy(vec_f_local, vec_f_d, N, 1, 1); //vec_f_d -> vec_f
		Arnoldi::to_host_from_device_real_cpy(V_local, V_d, N, m, 1); //vec_V_d -> vec_V
		Arnoldi::to_host_from_device_real_cpy(V1_local, V1_d, N, m, 1);
		real complex *eigenvectorsA=new real complex[N*k];

		cublas_complex_to_complex_real(N*k, eigenvectorsA_d, eigenvectorsA);

		real *V_real_local=new real[N*k];
		real *V_imag_local=new real[N*k];
		Arnoldi::to_host_from_device_real_cpy(V_real_local, eigenvectors_real_d, N, k, 1);
		Arnoldi::to_host_from_device_real_cpy(V_imag_local, eigenvectors_imag_d, N, k, 1);

		print_matrix("EigVecA.dat", N, k, eigenvectorsA);
		print_matrix("V1.dat", N, m, V1_local);
		print_matrix("V_real.dat", N, k, V_real_local);//eigenvectors_real_d
		print_matrix("V_imag.dat", N, k, V_imag_local);//eigenvectors_imag_d
		print_matrix("V.dat", N, m, V_local);
		print_matrix("H.dat", m, m, H);
		print_matrix("H1.dat", m, m, H1);
		print_matrix("H2.dat", m, m, H2);
		print_matrix("R.dat", m, m, R);
		print_matrix("Q.dat", m, m, Q);	
		print_matrix("EigVecH.dat", m, m, eigenvectorsH);
		print_vector("EigH.dat", m, eigenvaluesH);
		print_vector("f.dat", N, vec_f_local);	

		delete [] eigenvectorsA, vec_f_local, V_local, V1_local;
		delete [] V_real_local,V_imag_local;
		printf("done\n");
	}
	cudaFree(eigenvectorsA_d);
	if(!external_eigenvectors){
		cudaFree(eigenvectors_real_d);
		cudaFree(eigenvectors_imag_d);
	}

	Arnoldi::device_deallocate_all_real(15, V_d, V1_d, Vl_r_d, Vl_i_d, Vr_r_d, Vr_i_d, Vre_d, Vim_d, vec_f1_d, vec_w_d, vec_v_d, vec_c_d, vec_h_d, vec_q_d, Q_d);

	delete [] vec_c, vec_h, vec_q;
	delete [] H, R, Q, H1, H2;
	delete [] eigenvectorsH, eigenvaluesH, eigenvectorsH_kk, eigenvaluesH_kk, ritz_vector;


	return ritz_norm;


}