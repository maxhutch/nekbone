#include <stdio.h>
#include "cuda.h"
#include "cublas_v2.h"
#include "magma.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define MIN(a,b) (((a)<(b))?(a):(b))
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

static __global__ void set_addr(double** batch_u,
                                double** batch_us,
				double** batch_Dt,
				double** batch_D,
				double* u,
				double* us,
				double* Dt,
				double* D,
				int size,
				int num){
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;
  int i;
  for (i = idx; i < num; i+=nthreads){
    batch_u[i] = u + i*size;
    batch_us[i] = us + i*size;
    batch_Dt[i] = Dt;
    batch_D[i] = D;
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

#define NUM_BLOCK_MAX 1
#define NUM_STREAM_MAX 4
#define USE_BATCH
extern cudaStream_t* streams;
extern cublasHandle_t cublas_ctx;
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
  int i, j, jp;

  int num_block_l;
 
  const double zero = 0.0, one = 1.0;

  double *u_d, *D_d, *Dt_d, *g_d;
  double *ur_d, *us_d, *ut_d; 
  double *u_l, *ur_l, *us_l, *ut_l, *g_l;
  double *u_h, *w_h, *g_h;

  double **batch_Dt_d; double **batch_D_d;
  double **batch_u_d; double **batch_us_d; 
  double **batch_u_l; double **batch_us_l; 

  // First, copy over D, Dt
  gpuErrchk(cudaMalloc(&D_d , size2 * sizeof(double)))
  gpuErrchk(cudaMalloc(&Dt_d, size2 * sizeof(double)))
  gpuErrchk(cudaMemcpy(D_d , D , size2*sizeof(double), cudaMemcpyHostToDevice))
#if 1
  gpuErrchk(cudaMemcpy(Dt_d, Dt, size2*sizeof(double), cudaMemcpyHostToDevice))
#else
  gpuErrchk(cudaMemcpy(Dt_d, D_d, size2*sizeof(double), cudaMemcpyDeviceToDevice)) 
  magmablas_dtranspose_inplace(size1, Dt_d, size1);
#endif
  // Allocate space for other stuff
  gpuErrchk(cudaMalloc(&ur_d, NUM_BLOCK_MAX * NUM_STREAM_MAX * size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&us_d, NUM_BLOCK_MAX * NUM_STREAM_MAX * size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&ut_d, NUM_BLOCK_MAX * NUM_STREAM_MAX * size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&u_d , NUM_BLOCK_MAX * NUM_STREAM_MAX * size3 * sizeof(double)))
  gpuErrchk(cudaMalloc(&g_d , NUM_BLOCK_MAX * NUM_STREAM_MAX * 6 * size3 * sizeof(double)))

  gpuErrchk(cudaMalloc(&batch_u_d , NUM_BLOCK_MAX * NUM_STREAM_MAX * size1 * sizeof(double*)))
  gpuErrchk(cudaMalloc(&batch_us_d , NUM_BLOCK_MAX * NUM_STREAM_MAX * size1 * sizeof(double*)))
  gpuErrchk(cudaMalloc(&batch_Dt_d , NUM_BLOCK_MAX * size1 * sizeof(double*)))
  gpuErrchk(cudaMalloc(&batch_D_d , NUM_BLOCK_MAX * size1 * sizeof(double*)))

  // Pin the input/output buffers in-place
  u_h = u; w_h = w; g_h = g;
  gpuErrchk(cudaHostRegister(u_h, num*size3*sizeof(double), 0));
  gpuErrchk(cudaHostRegister(w_h, num*size3*sizeof(double), 0));
  gpuErrchk(cudaHostRegister(g_h, num*6*size3*sizeof(double), 0));

  for (j = 0; j < num; j+=NUM_BLOCK_MAX){ 
    jp = (j/NUM_BLOCK_MAX) % NUM_STREAM_MAX;
    num_block_l = MIN(NUM_BLOCK_MAX, num-j*NUM_BLOCK_MAX);
    u_l = u_d + jp*size3; ur_l = ur_d + jp*size3; us_l = us_d + jp*size3; ut_l = ut_d + jp*size3;
    g_l = g_d + jp*6*size3;
    batch_u_l = batch_u_d + jp*size1; batch_us_l = batch_us_d + jp*size1; 
    gpuErrchk(cudaMemcpyAsync(u_l, u_h+  j*size3,   size3*sizeof(double)*num_block_l, cudaMemcpyHostToDevice, streams[jp]))
    gpuErrchk(cudaMemcpyAsync(g_l, g_h+6*j*size3, 6*size3*sizeof(double)*num_block_l, cudaMemcpyHostToDevice, streams[jp]))

#ifdef USE_BATCH
    set_addr<<<1,32, 0, streams[jp]>>>(batch_u_l, batch_us_l, batch_Dt_d, batch_D_d,
                                       u_l, us_l, Dt_d, D_d, 
				       size2, size1*num_block_l);
#endif
    // dgemm('N','N',n1,n3,n2,1.0,a,n1,b,n2,0.0,c,n1)
    magmablasSetKernelStream(streams[jp]);
    cublasSetStream(cublas_ctx, streams[jp]);
    magma_dgemm('T', 'N', size1, size2*num_block_l, size1, 
                1.0,  Dt_d, size1, 
                      u_l, size1,
                0.0, ur_l, size1);
#ifdef USE_BATCH
    cublasDgemmBatched(cublas_ctx,
                       CUBLAS_OP_N, CUBLAS_OP_N, size1, size1, size1,
		       &one, (const double **) batch_u_l, size1,
		             (const double **) batch_Dt_d, size1,
		       &zero, batch_us_l, size1,
		           size1*num_block_l);
#else
    for (i = 0; i < size1*num_block_l; i++){
      magma_dgemm('N', 'N', size1, size1, size1,
                  1.0,  u_l + i*size2, size1,
                       Dt_d          , size1,
                  0.0, us_l + i*size2, size1);
    }
#endif
    for (i = 0;i < num_block_l; i++){
    magma_dgemm('N', 'N', size2, size1, size1,
                1.0, u_l +i*size3, size2,
                     Dt_d, size1,
	        0.0, ut_l +i*size3, size2);
    }

    transform_k<<<128, 256, 0, streams[jp]>>>(ur_l, us_l, ut_l, g_l, size3*num_block_l);

    magma_dgemm('T', 'N', size1, size2*num_block_l, size1, 
                1.0, D_d, size1, 
                     ur_l, size1,
                0.0,  u_l, size1);
#ifdef USE_BATCH
    cublasDgemmBatched(cublas_ctx,
                       CUBLAS_OP_N, CUBLAS_OP_N, size1, size1, size1,
		       &one, (const double **) batch_us_l, size1,
		             (const double **) batch_D_d, size1,
		       &one, batch_u_l, size1,
		             size1*num_block_l);
#else
    for (i = 0; i < size1*num_block_l; i++){
      magma_dgemm('N', 'N', size1, size1, size1,
                  1.0, us_l + i*size2, size1,
                        D_d          , size1,
	          1.0,  u_l + i*size2, size1);
    }
#endif
    for (i = 0; i < num_block_l; i++){
    magma_dgemm('N', 'N', size2, size1, size1,
                1.0, ut_l + i*size3, size2,
                      D_d, size1,
	        1.0,  u_l + i*size3, size2);
    }
    gpuErrchk(cudaMemcpyAsync(w_h+j*size3, u_l, size3*sizeof(double)*num_block_l, cudaMemcpyDeviceToHost, streams[jp]))
  }
  cudaDeviceSynchronize(); 

  // Unpin the memcpy buffers
  gpuErrchk(cudaHostUnregister(u_h));
  gpuErrchk(cudaHostUnregister(w_h));
  gpuErrchk(cudaHostUnregister(g_h));

  gpuErrchk(cudaFree(ur_d)) gpuErrchk(cudaFree(us_d)) gpuErrchk(cudaFree(ut_d)) gpuErrchk(cudaFree(u_d))
  gpuErrchk(cudaFree(D_d)) gpuErrchk(cudaFree(Dt_d)) gpuErrchk(cudaFree(g_d))
  gpuErrchk(cudaFree(batch_u_d)) gpuErrchk(cudaFree(batch_us_d)) gpuErrchk(cudaFree(batch_Dt_d)) gpuErrchk(cudaFree(batch_D_d))
}

