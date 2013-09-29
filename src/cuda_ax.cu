#include <stdio.h>
#include "cuda.h"
#include "cublas_v2.h"
#include "magma.h"
#include "ld_functions.h"

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
				double** batch_u2,
				double** batch_ut,
				double** batch_Dt,
				double** batch_D,
				double* u,
				double* us,
				double* ut,
				double* Dt,
				double* D,
				int size,
				int num){
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;
  int i;
  for (i = idx; i < num; i+=nthreads){
    batch_u[i]  = u  + i*size*size;
    batch_us[i] = us + i*size*size;
    batch_Dt[i] = Dt;
    batch_D[i]  = D;
    if (i*size < num) {
    batch_u2[i] = u  + i*size*size*size;
    batch_ut[i] = ut + i*size*size*size;
    }
  }
}


template<int p, int p_sq, int p_cube, int p_cube_padded, int pts_per_thread, int slab_size, int cta_size>
__global__
__launch_bounds__(432,2)
void ax_cuda_maxhutch(double* __restrict__ u, const double* __restrict__ g, const double* __restrict__ dxm1, const double* __restrict__ dxtm1, int n_cells)
{
  int tid = threadIdx.x;
  __shared__ double s_u[p_cube];
  __shared__ double temp[cta_size];

  for (int cell_id=blockIdx.x; cell_id < n_cells; cell_id += gridDim.x)
  {
    // Load u in shared for the entire cell
    int offset = cell_id*p_cube;
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;
      s_u[pt_id] = ld_functions::ld_cg(&u[offset + pt_id]);
      //s_u[pt_id] = u[offset + pt_id];
    }

    __syncthreads();

    // Initialize wa to 0.
    double wa[pts_per_thread];
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      wa[k] = 0.;
    }

    // Now compute w for one slab at a time
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;
      int pt_id_div_p = pt_id/p;
      int pt_id_mod_p = pt_id%p;
      int pt_id_div_p_sq = pt_id/p_sq;
      int pt_id_mod_p_sq = pt_id%p_sq;

      double ur, us, ut;

      //  Now that data is loaded in shared, compute ur
      {
        int s_offset = pt_id_div_p*p;
        int d_offset  = pt_id_mod_p;

        ur = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          ur += __ldg(&dxm1[d_offset + p*i])*s_u[s_offset + i];
      }

      // Compute us
      {
        int plane = pt_id_div_p_sq;
        int s_offset = plane*p_sq + pt_id_mod_p;
        int d_offset = p*( (pt_id-plane*p_sq)/p);

        us = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          us += s_u[s_offset + p*i]*__ldg(&dxtm1[d_offset + i]);
      }

      // Compute ut
      {
        int s_offset = pt_id_mod_p_sq;
        int d_offset = pt_id_div_p_sq*p;

        ut = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          ut += s_u[s_offset + p_sq*i]*__ldg(&dxtm1[d_offset + i]);
      }

      // Transform
      {

        int offset = (cell_id*p_cube + pt_id)*6;

        //TODO: Switch to SOA
        double metric[6];
        #pragma unroll
        for (int i=0;i<6;i++)
          metric[i] = g[offset+i];


        // SOA HACK
        /*
        int offset = cell_id*p_cube + pt_id;

        //TODO: Switch to SOA
        double metric[6];
        #pragma unroll
        for (int i=0;i<6;i++)
        {
          //metric[i] = g[offset];
          metric[i] = ld_functions::ld_cg(&g[offset]);
          offset += n_cells*p_cube;
        }
        */
          //metric[i] = ld_functions::ld_cg(&(g[offset+i]));

        double wr = metric[0]*ur + metric[1]*us + metric[2]*ut;
        double ws = metric[1]*ur + metric[3]*us + metric[4]*ut;
        double wt = metric[2]*ur + metric[4]*us + metric[5]*ut;

        ur = wr;
        us = ws;
        ut = wt;
      }

      // Store ur in shared memory
      temp[tid] = ur;
      __syncthreads();

      // Now that data is loaded in shared, compute wa
      int tid_mod_p = tid%p;
      int tid_div_p = tid/p;
      int tid_mod_p_sq = tid%p_sq;
      int tid_div_p_sq = tid/p_sq;

      {
        int d_offset  = tid_mod_p;
        int s_offset = tid_div_p*p;

        #pragma unroll
        for (int i=0;i<p;i++)
          wa[k] += __ldg(&dxtm1[d_offset+p*i])*temp[s_offset + i];
      }

      __syncthreads();
      temp[tid] = us;
      __syncthreads();

      // Compute us
      {

        int plane = tid_div_p_sq;
        int s_offset = plane*p_sq + tid_mod_p;
        int d_offset = p*( (tid-plane*p_sq)/p);
        int stride = p;

        #pragma unroll
        for (int i=0;i<p;i++)
          wa[k] += temp[s_offset + stride*i]*__ldg(&dxm1[d_offset + i]);
      }

      __syncthreads();
      // Store ut in shared memory
      temp[tid] = ut;
      __syncthreads();

      #pragma unroll
      for (int k2=0;k2<pts_per_thread;k2++)
      {
        int i_start = k*slab_size;
        int pt_id_2 = k2*cta_size + tid;
        int plane = pt_id_2/p_sq;

        int s_offset = tid_mod_p_sq;
        int d_offset = plane*p;

        #pragma unroll
        for (int i_count=0; i_count < slab_size; i_count++)
        {
          wa[k2] += temp[s_offset + p_sq*i_count]*__ldg(&dxm1[d_offset + i_start]);
          i_start++;
        }
      }
      __syncthreads();

    } // Loop over k

    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;
      u[offset + pt_id] = wa[k];
    }
  } // Loop over blocks

}


#define NUM_BLOCK_MAX 32
#define NUM_STREAM_MAX 8
#define USE_BATCH
extern cudaStream_t* streams;
extern cublasHandle_t cublas_ctx;
static double* g_d;
static double *u_d, *D_d, *Dt_d;
static double *ur_d, *us_d, *ut_d; 
 
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

  double *u_l, *ur_l, *us_l, *ut_l, *g_l;
  double *u_h, *w_h;

  double **batch_Dt_d; double **batch_D_d;
  double **batch_u_d;  double **batch_u_l; 
  double **batch_u2_d; double **batch_u2_l; 
  double **batch_us_d; double **batch_us_l; 
  double **batch_ut_d; double **batch_ut_l; 



#ifdef USE_BATCH
  if (size1 != 12){
    gpuErrchk(cudaMalloc(&batch_u_d , NUM_BLOCK_MAX * NUM_STREAM_MAX * size1 * sizeof(double*)))
    gpuErrchk(cudaMalloc(&batch_us_d , NUM_BLOCK_MAX * NUM_STREAM_MAX * size1 * sizeof(double*)))
    gpuErrchk(cudaMalloc(&batch_u2_d , NUM_BLOCK_MAX * NUM_STREAM_MAX * sizeof(double*)))
    gpuErrchk(cudaMalloc(&batch_ut_d , NUM_BLOCK_MAX * NUM_STREAM_MAX * sizeof(double*)))
    gpuErrchk(cudaMalloc(&batch_Dt_d , NUM_BLOCK_MAX * size1 * sizeof(double*)))
    gpuErrchk(cudaMalloc(&batch_D_d , NUM_BLOCK_MAX * size1 * sizeof(double*)))
  }
#endif

  u_h = u; w_h = w;
  // Loop over blocks
  for (j = 0; j < num; j+=NUM_BLOCK_MAX){ 
    jp = (j/NUM_BLOCK_MAX) % NUM_STREAM_MAX; // stream ID
    num_block_l = MIN(NUM_BLOCK_MAX, num-j); // number of elements in block
    u_l  =  u_d + jp*NUM_BLOCK_MAX*size3; ur_l = ur_d + jp*NUM_BLOCK_MAX*size3; 
    us_l = us_d + jp*NUM_BLOCK_MAX*size3; ut_l = ut_d + jp*NUM_BLOCK_MAX*size3;
    g_l  =  g_d + j*6*size3;

    // copy over u
    gpuErrchk(cudaMemcpyAsync(u_l, u_h+  j*size3,   size3*sizeof(double)*num_block_l, cudaMemcpyHostToDevice, streams[jp]))
    if (size1 == 12){
      // 12x12x12 case
      const int cta_size = 432;
      const int p = 12;
      const int p_sq = 12*12;
      const int p_cube = 12*12*12;
      const int p_cube_padded = p_cube;

      // We could play with this
      const int pts_per_thread = 4;  // 6*288 = 12*12*12
      const int slab_size = 3;

      const int grid_size = num_block_l;

      ax_cuda_maxhutch<p,p_sq,p_cube,p_cube_padded,pts_per_thread,slab_size,cta_size>
      <<<grid_size,cta_size, 0, streams[jp]>>>(u_l, g_l, D_d, Dt_d, num_block_l);
    } else {
#ifdef USE_BATCH
    batch_u_l  = batch_u_d  + jp*NUM_BLOCK_MAX*size1; batch_us_l = batch_us_d + jp*NUM_BLOCK_MAX*size1; 
    batch_u2_l = batch_u2_d + jp*NUM_BLOCK_MAX; batch_ut_l = batch_ut_d + jp*NUM_BLOCK_MAX; 
    set_addr<<<1,32, 0, streams[jp]>>>(batch_u_l, batch_us_l, batch_u2_l, batch_ut_l,
                                       batch_Dt_d, batch_D_d,
                                       u_l, us_l, ut_l, Dt_d, D_d, 
				       size1, size1*num_block_l);
#endif
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
    cublasDgemmBatched(cublas_ctx,
                       CUBLAS_OP_N, CUBLAS_OP_N, size2, size1, size1,
		       &one, (const double **) batch_u2_l, size2,
		             (const double **) batch_Dt_d, size1,
		       &zero, batch_ut_l, size2,
		             num_block_l);
#else
    for (i = 0; i < size1*num_block_l; i++){
      magma_dgemm('N', 'N', size1, size1, size1,
                  1.0,  u_l + i*size2, size1,
                       Dt_d          , size1,
                  0.0, us_l + i*size2, size1);
    }
    for (i = 0;i < num_block_l; i++){
    magma_dgemm('N', 'N', size2, size1, size1,
                1.0, u_l +i*size3, size2,
                     Dt_d, size1,
	        0.0, ut_l +i*size3, size2);
    }
#endif
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
    cublasDgemmBatched(cublas_ctx,
                       CUBLAS_OP_N, CUBLAS_OP_N, size2, size1, size1,
		       &one, (const double **) batch_ut_l, size2,
		             (const double **) batch_D_d, size1,
		       &one, batch_u2_l, size2,
		             num_block_l);
#else
    for (i = 0; i < size1*num_block_l; i++){
      magma_dgemm('N', 'N', size1, size1, size1,
                  1.0, us_l + i*size2, size1,
                        D_d          , size1,
	          1.0,  u_l + i*size2, size1);
    }
    for (i = 0; i < num_block_l; i++){
    magma_dgemm('N', 'N', size2, size1, size1,
                1.0, ut_l + i*size3, size2,
                      D_d, size1,
	        1.0,  u_l + i*size3, size2);
    }
#endif
    }
    // Copy back w
    gpuErrchk(cudaMemcpyAsync(w_h+j*size3, u_l, size3*sizeof(double)*num_block_l, cudaMemcpyDeviceToHost, streams[jp]))
  }
  cudaDeviceSynchronize(); 

#ifdef USE_BATCH
  if (size1 != 12){
    gpuErrchk(cudaFree(batch_u_d));
    gpuErrchk(cudaFree(batch_u2_d));
    gpuErrchk(cudaFree(batch_us_d));
    gpuErrchk(cudaFree(batch_ut_d));
    gpuErrchk(cudaFree(batch_Dt_d));
    gpuErrchk(cudaFree(batch_D_d))
  }
#endif
}



extern "C" void setup_cg_cuda_(double* w,
                          double* u,
			  double* g,
			  double* D,
			  double* Dt,
			  int* n,
			  int* m){

  int size1 = *n + 1;
  int size2 = size1 * size1;
  int size3 = size1 * size1 * size1;
  int num = *m; 
  
  // Pin memcpy buffers
  gpuErrchk(cudaHostRegister(u, num*size3*sizeof(double), 0));
  gpuErrchk(cudaHostRegister(w, num*size3*sizeof(double), 0));
  gpuErrchk(cudaHostRegister(g, num*6*size3*sizeof(double), 0));
 
  // Copy over g
  gpuErrchk(cudaMalloc(&g_d,    num * 6 * size3 * sizeof(double)))
  gpuErrchk(cudaMemcpy(g_d , g, num *6*size3*sizeof(double), cudaMemcpyHostToDevice))

  // Copy over D, Dt
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

}

extern "C" void teardown_cg_cuda_(double* w,
                             double* u,
			     double* g,
			     int* n,
			     int* m){

  // Un-pin the buffers (so it can be paged again)
  gpuErrchk(cudaHostUnregister(u));
  gpuErrchk(cudaHostUnregister(w));
  gpuErrchk(cudaHostUnregister(g));

  // Free the device copy of g, u's, D's
  gpuErrchk(cudaFree(g_d));
  gpuErrchk(cudaFree(ur_d));
  gpuErrchk(cudaFree(us_d));
  gpuErrchk(cudaFree(ut_d));
  gpuErrchk(cudaFree( u_d));
  gpuErrchk(cudaFree( D_d));
  gpuErrchk(cudaFree(Dt_d));

}

