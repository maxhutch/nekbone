#include <stdio.h>
#include "cuda.h"
#include "magma.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) exit(code);
   }
}

extern "C" void local_grad3_cuda_(double *ur,
                                 double *us, 
				 double *ut,
				 double *u,
				 int *n,
				 double *D,
				 double *Dt){

  int size1 = *n + 1;
  int size2 = size1 * size1;
  int size3 = size1 * size1 * size1;
  int i;

  double *ur_d, *us_d, *ut_d, *u_d, *D_d, *Dt_d;
  gpuErrchk(cudaMalloc(&ur_d, size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&us_d, size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&ut_d, size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&u_d , size3 * sizeof(double)))

  gpuErrchk(cudaMalloc(&D_d , size2 * sizeof(double)))
  gpuErrchk(cudaMalloc(&Dt_d, size2 * sizeof(double)))

  gpuErrchk(cudaMemcpy(u_d , u , size3*sizeof(double), cudaMemcpyHostToDevice))
  gpuErrchk(cudaMemcpy(D_d , D , size2*sizeof(double), cudaMemcpyHostToDevice))
  gpuErrchk(cudaMemcpy(Dt_d, Dt, size2*sizeof(double), cudaMemcpyHostToDevice))

  // dgemm('N','N',n1,n3,n2,1.0,a,n1,b,n2,0.0,c,n1)
  magma_dgemm('N', 'N', size1, size2, size1, 
              1.0,  D_d, size1, 
	            u_d, size1,
              0.0, ur_d, size1);
  for (i = 0; i < size1; i++){
    magma_dgemm('N', 'N', size1, size1, size1,
                1.0,  u_d + i*size2, size1,
		     Dt_d          , size1,
		0.0, us_d + i*size2, size1);
  }
  magma_dgemm('N', 'N', size2, size1, size1,
              1.0, u_d , size2,
	           Dt_d, size1,
	      0.0, ut_d, size2);

  gpuErrchk(cudaMemcpy(ur, ur_d, size3*sizeof(double), cudaMemcpyDeviceToHost))
  gpuErrchk(cudaMemcpy(us, us_d, size3*sizeof(double), cudaMemcpyDeviceToHost))
  gpuErrchk(cudaMemcpy(ut, ut_d, size3*sizeof(double), cudaMemcpyDeviceToHost))
  
  gpuErrchk(cudaFree(ur_d)) gpuErrchk(cudaFree(us_d)) gpuErrchk(cudaFree(ut_d)) gpuErrchk(cudaFree(u_d))
  gpuErrchk(cudaFree(D_d)) gpuErrchk(cudaFree(Dt_d))
}

extern "C" void local_grad3_t_cuda_(double *u,
                                    double *ur,
                                    double *us, 
				    double *ut,
				    int *n,
				    double *D,
				    double *Dt,
				    double *w){

  int size1 = *n + 1;
  int size2 = size1 * size1;
  int size3 = size1 * size1 * size1;
  int i;

  double *ur_d, *us_d, *ut_d, *u_d, *D_d, *Dt_d;
  gpuErrchk(cudaMalloc(&ur_d, size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&us_d, size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&ut_d, size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&u_d , size3 * sizeof(double)))

  gpuErrchk(cudaMalloc(&D_d , size2 * sizeof(double)))
  gpuErrchk(cudaMalloc(&Dt_d, size2 * sizeof(double)))

  gpuErrchk(cudaMemcpy(ur_d, ur, size3*sizeof(double), cudaMemcpyHostToDevice))
  gpuErrchk(cudaMemcpy(us_d, us, size3*sizeof(double), cudaMemcpyHostToDevice))
  gpuErrchk(cudaMemcpy(ut_d, ut, size3*sizeof(double), cudaMemcpyHostToDevice))
  gpuErrchk(cudaMemcpy(D_d , D , size2*sizeof(double), cudaMemcpyHostToDevice))
  gpuErrchk(cudaMemcpy(Dt_d, Dt, size2*sizeof(double), cudaMemcpyHostToDevice))

  // dgemm('N','N',n1,n3,n2,1.0,a,n1,b,n2,0.0,c,n1)
  magma_dgemm('N', 'N', size1, size2, size1, 
              1.0, Dt_d, size1, 
	           ur_d, size1,
              0.0,  u_d, size1);
  for (i = 0; i < size1; i++){
    magma_dgemm('N', 'N', size1, size1, size1,
                1.0, us_d + i*size2, size1,
		      D_d          , size1,
		1.0,  u_d + i*size2, size1);
  }
  magma_dgemm('N', 'N', size2, size1, size1,
              1.0, ut_d, size2,
	            D_d, size1,
	      1.0,  u_d, size2);

  gpuErrchk(cudaMemcpy(u, u_d, size3*sizeof(double), cudaMemcpyDeviceToHost))
  
  gpuErrchk(cudaFree(ur_d)) gpuErrchk(cudaFree(us_d)) gpuErrchk(cudaFree(ut_d)) gpuErrchk(cudaFree(u_d))
  gpuErrchk(cudaFree(D_d)) gpuErrchk(cudaFree(Dt_d))
}

static __global__ void transform_k(double* ur,
                                   double* us,
				   double* ut,
				   double* trans,
				   int n){

  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;
  double wr, ws, wt;
  int i;

  for (i = idx; i < n; i += nthreads){
    wr = trans[6*i+0]*ur[i] + trans[6*i+1] * us[i] + trans[6*i+2]*ut[i];
    ws = trans[6*i+1]*ur[i] + trans[6*i+3] * us[i] + trans[6*i+4]*ut[i];
    wt = trans[6*i+2]*ur[i] + trans[6*i+4] * us[i] + trans[6*i+5]*ut[i];
    ur[i] = wr;
    us[i] = ws;
    ut[i] = wt;
  }
}


extern "C" void local_grad3_comb_cuda_(double *w,
                                      double* u,
                                      double* D,
				      double* Dt,
				      double* g,
				      int* n){
  int size1 = *n + 1;
  int size2 = size1 * size1;
  int size3 = size1 * size1 * size1;
  int i;

  double *u_d, *D_d, *Dt_d, *g_d;
  double *ur_d, *us_d, *ut_d; 

  gpuErrchk(cudaMalloc(&ur_d, size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&us_d, size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&ut_d, size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&u_d , size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&g_d , 6 * size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&D_d , size2 * sizeof(double)))
  gpuErrchk(cudaMalloc(&Dt_d, size2 * sizeof(double)))

  gpuErrchk(cudaMemcpy(u_d , u , size3*sizeof(double), cudaMemcpyHostToDevice))
  gpuErrchk(cudaMemcpy(D_d , D , size2*sizeof(double), cudaMemcpyHostToDevice))
  gpuErrchk(cudaMemcpy(Dt_d, Dt, size2*sizeof(double), cudaMemcpyHostToDevice))
  gpuErrchk(cudaMemcpy(g_d , g , 6*size3*sizeof(double), cudaMemcpyHostToDevice))

  // dgemm('N','N',n1,n3,n2,1.0,a,n1,b,n2,0.0,c,n1)
  magma_dgemm('N', 'N', size1, size2, size1, 
              1.0,  D_d, size1, 
	            u_d, size1,
              0.0, ur_d, size1);
  for (i = 0; i < size1; i++){
    magma_dgemm('N', 'N', size1, size1, size1,
                1.0,  u_d + i*size2, size1,
		     Dt_d          , size1,
		0.0, us_d + i*size2, size1);
  }
  magma_dgemm('N', 'N', size2, size1, size1,
              1.0, u_d , size2,
	           Dt_d, size1,
	      0.0, ut_d, size2);

  transform_k<<<128, 256>>>(ur_d, us_d, ut_d, g_d, size3);

  magma_dgemm('N', 'N', size1, size2, size1, 
              1.0, Dt_d, size1, 
	           ur_d, size1,
              0.0,  u_d, size1);
  for (i = 0; i < size1; i++){
    magma_dgemm('N', 'N', size1, size1, size1,
                1.0, us_d + i*size2, size1,
		      D_d          , size1,
		1.0,  u_d + i*size2, size1);
  }
  magma_dgemm('N', 'N', size2, size1, size1,
              1.0, ut_d, size2,
	            D_d, size1,
	      1.0,  u_d, size2);

  gpuErrchk(cudaMemcpy(w, u_d, size3*sizeof(double), cudaMemcpyDeviceToHost))
  
  gpuErrchk(cudaFree(ur_d)) gpuErrchk(cudaFree(us_d)) gpuErrchk(cudaFree(ut_d)) gpuErrchk(cudaFree(u_d))
  gpuErrchk(cudaFree(D_d)) gpuErrchk(cudaFree(Dt_d)) gpuErrchk(cudaFree(g_d))
}


extern "C" void ax_e_cuda_(double *w,
                                      double* u,
                                      double* D,
				      double* Dt,
				      double* g,
				      int* n,
				      int* m){
  int size1 = *n + 1;
  int size2 = size1 * size1;
  int size3 = size1 * size1 * size1;
  int num = *m;
  int i, j;

  double *u_d, *D_d, *Dt_d, *g_d;
  double *ur_d, *us_d, *ut_d; 

  // First, copy over D, Dt
  gpuErrchk(cudaMalloc(&D_d , size2 * sizeof(double)))
  gpuErrchk(cudaMalloc(&Dt_d, size2 * sizeof(double)))
  gpuErrchk(cudaMemcpy(D_d , D , size2*sizeof(double), cudaMemcpyHostToDevice))
  gpuErrchk(cudaMemcpy(Dt_d, Dt, size2*sizeof(double), cudaMemcpyHostToDevice))

  // Allocate space for other stuff
  gpuErrchk(cudaMalloc(&ur_d, size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&us_d, size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&ut_d, size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&u_d , size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&g_d , 6 * size3 * sizeof(double)))

  for (j = 0; j < num; j++){ 
    gpuErrchk(cudaMemcpy(u_d , u+  j*size3,   size3*sizeof(double), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(g_d , g+6*j*size3, 6*size3*sizeof(double), cudaMemcpyHostToDevice))

    // dgemm('N','N',n1,n3,n2,1.0,a,n1,b,n2,0.0,c,n1)
    magma_dgemm('N', 'N', size1, size2, size1, 
                1.0,  D_d, size1, 
                      u_d, size1,
                0.0, ur_d, size1);
    for (i = 0; i < size1; i++){
      magma_dgemm('N', 'N', size1, size1, size1,
                  1.0,  u_d + i*size2, size1,
                       Dt_d          , size1,
                  0.0, us_d + i*size2, size1);
    }
    magma_dgemm('N', 'N', size2, size1, size1,
                1.0, u_d , size2,
                     Dt_d, size1,
	        0.0, ut_d, size2);

    transform_k<<<128, 256>>>(ur_d, us_d, ut_d, g_d, size3);

    magma_dgemm('N', 'N', size1, size2, size1, 
                1.0, Dt_d, size1, 
                     ur_d, size1,
                0.0,  u_d, size1);
    for (i = 0; i < size1; i++){
      magma_dgemm('N', 'N', size1, size1, size1,
                  1.0, us_d + i*size2, size1,
                        D_d          , size1,
	          1.0,  u_d + i*size2, size1);
    }
    magma_dgemm('N', 'N', size2, size1, size1,
                1.0, ut_d, size2,
                      D_d, size1,
	        1.0,  u_d, size2);
    gpuErrchk(cudaMemcpy(w+j*size3, u_d, size3*sizeof(double), cudaMemcpyDeviceToHost))
  }
  
  gpuErrchk(cudaFree(ur_d)) gpuErrchk(cudaFree(us_d)) gpuErrchk(cudaFree(ut_d)) gpuErrchk(cudaFree(u_d))
  gpuErrchk(cudaFree(D_d)) gpuErrchk(cudaFree(Dt_d)) gpuErrchk(cudaFree(g_d))
}





