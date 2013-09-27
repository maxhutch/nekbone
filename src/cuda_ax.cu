
#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda_util.h"
#include "ld_functions.h"
#include "sm_utils.inl"

  /* GPU data pointers */
  static double *d_u, *d_w, *d_g, *d_dxm1, *d_dxtm1, *d_ur, *d_us, *d_ut;


  /*
   * Copy data to the GPU
  */
  void copyTo(int nelt, int nxyz, double *u) {

    cudaError_t err = cudaMemcpy(d_u, u, nelt*nxyz*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      printf("cudaMemcpy error\n");
      exit(1);
    }

    return; 
  }


  /*
   * Copy data from the GPU
  */
  void copyFrom(int nelt, int nxyz, double *w) {

    cudaError_t err = cudaMemcpy(w, d_w, nelt*nxyz*sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      printf("cudaMemcpy error\n");
      exit(1);
    }

    return;
  }


  /*
   * Stacked Matrix multiply
  */
  __global__ void gsmxm(double *a, int n1, double *b, int n2, double *c, int n3, int nlvl) {
    int id= blockDim.x*blockIdx.x+ threadIdx.x;
   
    int aSize= n1*n2; 
    int cSize= n1*n3; 
    int lvl = id/cSize;
    int rank = id % cSize;
    int row = rank % n1;
    int col = rank / n1;

    if (id < cSize*nlvl) {
      c[id] = 0.0;

      int k;
      for (k = 0; k < n2; k++) {
        c[id] += a[lvl*aSize+k*n1+row]*b[col*n2+k];
      }
    }

    return;
  }


  /*
   * Add two vectors
  */
  __global__ void gadd2(double *a, double *b, int n) {
    int id= blockDim.x*blockIdx.x+ threadIdx.x;

    if (id < n) {
      a[id] += b[id];
    }

    return;
  }


  /*
   * Perform geometry scaling
  */
  __global__ void geom(int n, double *ur, double *us, double *ut, double *g) {
    int id= blockDim.x*blockIdx.x+ threadIdx.x;

    if (id < n) {
      double wr = g[id*6+0]*ur[id] + g[id*6+1]*us[id] + g[id*6+2]*ut[id];
      double ws = g[id*6+1]*ur[id] + g[id*6+3]*us[id] + g[id*6+4]*ut[id];
      double wt = g[id*6+2]*ur[id] + g[id*6+4]*us[id] + g[id*6+5]*ut[id];
      ur[id] = wr;
      us[id] = ws;
      ut[id] = wt;
    }

    return;
  }

#if 1
template<int p, int p_sq, int p_cube, int p_cube_padded, int pts_per_thread, int slab_size, int cta_size>
__global__
__launch_bounds__(288,1)
void ax_cuda_kernel_v5(const double* __restrict__ u_glob, double* __restrict__ w, const double* __restrict__ g, const double* __restrict__ dxm1, const double* __restrict__ dxtm1, int n_cells)
{
  int tid = threadIdx.x;
  __shared__ double s_temp[cta_size];

  for (int cell_id=blockIdx.x; cell_id < n_cells; cell_id += gridDim.x)
  {
    double u[pts_per_thread];

    // Load u in shared for the entire cell
    int offset = cell_id*p_cube;
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;
      __syncthreads();
      s_temp[tid] = ld_functions::ld_cg(&u_glob[offset + pt_id]);
      __syncthreads();

      // Transpose to store in registers and allow shfl
      int pos = (tid%slab_size)*p_sq + tid/slab_size;
      u[k] = s_temp[pos];
      if (cell_id ==0 && (tid > 165 && tid < 168))
        printf("tid = %d, k = %d, pos = %d, u = %f\n",tid,k,pos,u[k]);
    }

    double wa[pts_per_thread];
    // Initialize wa to 0.
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      wa[k] = 0.;
    }

    // Now compute w for one slab at a time
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;
      int pt_id_div_p = pt_id/p;
      int pt_id_mod_p = pt_id%p;
      int pt_id_div_p_sq = pt_id/p_sq;
      int pt_id_mod_p_sq = pt_id%p_sq;

      int tid_mod_p = tid%p;
      int tid_div_p = tid/p;
      int tid_mod_p_sq = tid%p_sq;
      int tid_div_p_sq = tid/p_sq;

      double ur, us, ut;

      // Load slab in shared
      int pos = (tid%slab_size)*p_sq + tid/slab_size;
      __syncthreads();
      s_temp[pos] = u[k];
      __syncthreads();

      //  Now that data is loaded in shared, compute ur
      {
        //int s_offset = pt_id_div_p*p;
        //int d_offset  = pt_id_mod_p;

        int d_offset  = tid_mod_p;
        int s_offset = tid_div_p*p;

        ur = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          ur += __ldg(&dxm1[d_offset + p*i])*s_temp[s_offset + i];
      }

      if (cell_id == 0 && tid == 166)
        printf("k=%d, ur = %f\n",k,ur);

      // Compute us
      {
        //int plane = pt_id_div_p_sq;
        //int s_offset = plane*p_sq + pt_id_mod_p;
        //int d_offset = p*( (pt_id-plane*p_sq)/p);


        int plane = tid_div_p_sq;
        int s_offset = plane*p_sq + tid_mod_p;
        int d_offset = p*( (tid-plane*p_sq)/p);
        int stride = p;

        us = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          us += s_temp[s_offset + p*i]*__ldg(&dxtm1[d_offset + i]);
      }
      if (cell_id == 0 && tid == 166)
        printf("k=%d, us = %f\n",k,us);

      // Use shuffle to compute ut
#if 1
      // Compute ut
      {
        //int d_offset = pt_id_div_p_sq*p;

        int d_offset = (k*cta_size + pos)/p_sq*p;
        ut = 0.;
        #pragma unroll
        for (int j=0;j<pts_per_thread;j++)
        {
          #pragma unroll
          for (int m=0;m<slab_size;m++)
          {
            int i = j*slab_size+m;
            double val = u[j];
            double sh_val = utils::shfl(val,m,slab_size);
            if (cell_id == 0 && tid == 166)
              printf("k=%d, u = %f, dxtm1 = %f\n",k,sh_val,__ldg(&dxtm1[d_offset + i]));
            ut += sh_val*__ldg(&dxtm1[d_offset + i]);
          }
        }

        __syncthreads();
        s_temp[tid] = ut;
        __syncthreads();
        ut = s_temp[pos];
        __syncthreads();

        if (cell_id == 0 && tid == 166)
          printf("k=%d, ut = %f\n",k,ut);
          //ut += s_u[s_offset + p_sq*i]*__ldg(&dxtm1[d_offset + i]);

      }
#endif

      // Transform
      {
        int offset = (cell_id*p_cube + pt_id)*6;

        //TODO: Switch to SOA
        double metric[6];
        #pragma unroll
        for (int i=0;i<6;i++)
          metric[i] = g[offset+i];
          //metric[i] = ld_functions::ld_cg(&g[offset+i]);
          //metric[i] = g[offset+i];


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

      __syncthreads();
      // Store ur in shared memory
      s_temp[tid] = ur;
      __syncthreads();

      // Now that data is loaded in shared, compute wa
      {
        int d_offset  = tid_mod_p;
        int s_offset = tid_div_p*p;

        #pragma unroll
        for (int i=0;i<p;i++)
          wa[k] += __ldg(&dxtm1[d_offset+p*i])*s_temp[s_offset + i];
      }

      __syncthreads();
      s_temp[tid] = us;
      __syncthreads();

      // Compute us
      {

        int plane = tid_div_p_sq;
        int s_offset = plane*p_sq + tid_mod_p;
        int d_offset = p*( (tid-plane*p_sq)/p);

        #pragma unroll
        for (int i=0;i<p;i++)
          wa[k] += s_temp[s_offset + p*i]*__ldg(&dxm1[d_offset + i]);
      }


      // Use shuffle to compute contribution from ut
      __syncthreads();
      // Store ut in shared memory
      s_temp[tid] = ut;
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
          wa[k2] += s_temp[s_offset + p_sq*i_count]*__ldg(&dxm1[d_offset + i_start]);
          i_start++;
        }
      }
      __syncthreads();

    } // Loop over k

    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;
      w[offset + pt_id] = wa[k];
    }
  } // Loop over blocks

}
#endif



template<int p, int p_sq, int p_cube, int p_cube_padded, int pts_per_thread, int slab_size, int cta_size>
__global__
//__launch_bounds__(432,2)
__launch_bounds__(288,3)
void ax_cuda_kernel_v4(const double* __restrict__ u, double* __restrict__ w, const double* __restrict__ g, const double* __restrict__ dxm1, const double* __restrict__ dxtm1, int n_cells)
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
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;
      int pt_id_div_p = pt_id/p;
      int pt_id_mod_p = pt_id%p;
      int pt_id_div_p_sq = pt_id/p_sq;
      int pt_id_mod_p_sq = pt_id%p_sq;

      int tid_mod_p = tid%p;
      int tid_div_p = tid/p;
      int tid_mod_p_sq = tid%p_sq;
      int tid_div_p_sq = tid/p_sq;

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

      if (cell_id == 0 && tid == 166)
        printf("k=%d, ur = %f\n",k,ur);
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

      if (cell_id == 0 && tid == 166)
        printf("k=%d, us = %f\n",k,us);
      // Compute ut
      {
        int s_offset = pt_id_mod_p_sq;
        int d_offset = pt_id_div_p_sq*p;

        ut = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
        {
          if (cell_id == 0 && tid == 166)
            printf("k=%d, u = %f, dxtm1 = %f\n",k,s_u[s_offset + p_sq*i],__ldg(&dxtm1[d_offset + i]));
          ut += s_u[s_offset + p_sq*i]*__ldg(&dxtm1[d_offset + i]);
        }
      }

      if (cell_id == 0 && tid == 166)
        printf("k=%d, ut = %f\n",k,ut);
      // Transform
      {

        int offset = (cell_id*p_cube + pt_id)*6;

        //TODO: Switch to SOA
        double metric[6];
        #pragma unroll
        for (int i=0;i<6;i++)
          metric[i] = g[offset+i];
          //metric[i] = ld_functions::ld_cg(&g[offset+i]);
          //metric[i] = g[offset+i];


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

        #pragma unroll
        for (int i=0;i<p;i++)
          wa[k] += temp[s_offset + p*i]*__ldg(&dxm1[d_offset + i]);
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
      w[offset + pt_id] = wa[k];
    }
  } // Loop over blocks

}



template<int p, int p_sq, int p_cube, int p_cube_padded, int pts_per_thread, int slab_size, int cta_size>
__global__
__launch_bounds__(432,2)
void ax_cuda_kernel_v3(const double* __restrict__ u, double* __restrict__ w, const double* __restrict__ g, const double* __restrict__ dxm1, const double* __restrict__ dxtm1, int n_cells)
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
      w[offset + pt_id] = wa[k];
    }
  } // Loop over blocks

}



template<int p, int p_sq, int p_cube, int p_cube_padded, int pts_per_thread, int cta_size>
//__launch_bounds__(288,2)
__global__
void ax_cuda_kernel_v2(double* __restrict__ u, double* __restrict__ w, double* __restrict__ g, double* __restrict__ dxm1, double* __restrict__ dxtm1, int n_cells)
{
  int tid = threadIdx.x;
  __shared__ double s_u[p_cube];
  __shared__ double temp[cta_size];

  int slab_size = p/pts_per_thread;

  for (int cell_id=blockIdx.x; cell_id < n_cells; cell_id += gridDim.x)
  {
    //int cell_id = blockIdx.x;

    // Load u in shared for the entire cell
    int offset = cell_id*p_cube;
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;
      s_u[pt_id] = u[offset + pt_id];
    }

    __syncthreads();

    // Initialize wa to 0.
    double wa[pts_per_thread];
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
      wa[k] = 0.;

    // Now compute w for one slab at a time
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;

      double ur, us, ut;

      //  Now that data is loaded in shared, compute ur
      {
        int line = pt_id/p;
        int s_offset = line*p;

        int d_offset  = pt_id%p;
        int stride = p;

        ur = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          ur += __ldg(&dxm1[d_offset + stride*i])*s_u[s_offset + i];
      }

      // Compute us
      {
        int pt_id_mod_p = pt_id%p;
        int pt_id_div_p = pt_id/p;

        int plane = pt_id/p_sq;
        int s_offset = plane*p_sq + pt_id_mod_p;
        int d_offset = p*( (pt_id-plane*p_sq)/p);
        int stride = p;

        us = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          us += s_u[s_offset + stride*i]*__ldg(&dxtm1[d_offset + i]);
      }

      // Compute ut
      {
        int s_offset = pt_id%(p_sq);
        int plane = pt_id/p_sq;
        int d_offset = plane*p;
        int stride = p_sq;

        ut = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          ut += s_u[s_offset + stride*i]*__ldg(&dxtm1[d_offset + i]);
      }

      // Transform
      {
        int offset = (cell_id*p_cube + pt_id)*6;

        //TODO: Switch to SOA
        double metric[6];
        #pragma unroll
        for (int i=0;i<6;i++)
          metric[i] = g[offset+i];

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
      {
        int d_offset  = tid%p;
        int line = tid/p;
        int s_offset = line*p;
        int stride = p;

        #pragma unroll
        for (int i=0;i<p;i++)
          wa[k] += __ldg(&dxtm1[d_offset+stride*i])*temp[s_offset + i];
      }

      __syncthreads();
      temp[tid] = us;
      __syncthreads();

      // Compute us
      {
        int tid_mod_p = tid%p;
        int tid_div_p = tid/p;

        int plane = tid/p_sq;
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
        int pt_id = k2*cta_size + tid;
        int s_offset = tid%(p_sq);
        int plane = pt_id/p_sq;
        int d_offset = plane*p;
        int stride = p_sq;

        int i_start = k*slab_size;
        int i_end = (k+1)*slab_size;
        int i_count = 0;
        #pragma unroll
        for (int i=i_start;i<i_end;i++)
        {
          wa[k2] += temp[s_offset + stride*i_count]*__ldg(&dxm1[d_offset + i]);
          i_count++;
        }
      }

    } // Loop over k

    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;
      w[offset + pt_id] = wa[k];
    }
  } // Loop over blocks

}




template<int p, int p_cube, int p_cube_padded, int pts_per_thread, int cta_size>
__global__
__launch_bounds__(864,1)
void ax_cuda_kernel(double* __restrict__ u, double* __restrict__ w, double* __restrict__ g, double* __restrict__ dxm1, double* __restrict__ dxtm1, int n_cells)
{
  int tid = threadIdx.x;

  __shared__ double s_u[p_cube];

  for (int cell_id=blockIdx.x; cell_id < n_cells; cell_id += gridDim.x)
  {
    int cell_id = blockIdx.x;

    double ur[pts_per_thread];
    double us[pts_per_thread];
    double ut[pts_per_thread];


    int offset = cell_id*p_cube;
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;
      s_u[pt_id] = u[offset + pt_id];
    }

    __syncthreads();

    //  Now that data is loaded in shared, compute ur
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;

      int line = pt_id/p;
      int s_offset = line*p;

      int d_offset  = pt_id%p;
      int stride = p;

      ur[k] = 0.;
      #pragma unroll
      for (int i=0;i<p;i++)
        ur[k] += dxm1[d_offset + stride*i]*s_u[s_offset + i];
    }

    // Compute us
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;

      int pt_id_mod_p = pt_id%p;
      int pt_id_div_p = pt_id/p;

      int plane = pt_id/(p*p);
      int s_offset = plane*p*p + pt_id_mod_p;
      int d_offset = p*( (pt_id-plane*p*p)/p);
      int stride = p;

      us[k] = 0.;
      #pragma unroll
      for (int i=0;i<p;i++)
        us[k] += s_u[s_offset + stride*i]*__ldg(&dxtm1[d_offset + i]);

    }

    // Compute ut
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;
      int p_sq = p*p;

      int s_offset = pt_id%(p_sq);
      int plane = pt_id/p_sq;
      int d_offset = plane*p;
      int stride = p_sq;

      ut[k] = 0.;
      #pragma unroll
      for (int i=0;i<p;i++)
        ut[k] += s_u[s_offset + stride*i]*__ldg(&dxtm1[d_offset + i]);
    }

    // Transform
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;
      int offset = (cell_id*p_cube + pt_id)*6;

      //TODO: Switch to SOA
      double metric[6];
      #pragma unroll
      for (int i=0;i<6;i++)
        metric[i] = g[offset+i];

      double wr = metric[0]*ur[k] + metric[1]*us[k] + metric[2]*ut[k];
      double ws = metric[1]*ur[k] + metric[3]*us[k] + metric[4]*ut[k];
      double wt = metric[2]*ur[k] + metric[4]*us[k] + metric[5]*ut[k];

      ur[k] = wr;
      us[k] = ws;
      ut[k] = wt;
    }

    // Store ur in shared memory

    // Wait for all threads to be done with data in s_u
    __syncthreads();

    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
      s_u[cta_size*k + tid] = ur[k];

    // Wait for all threads to have loaded data in s_u
    __syncthreads();

    double wa[pts_per_thread];

    // Now that data is loaded in shared, compute wa
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;

      int d_offset  = pt_id%p;
      int line = pt_id/p;
      int s_offset = line*p;
      int stride = p;

      wa[k] = 0.;
      #pragma unroll
      for (int i=0;i<p;i++)
        wa[k] += __ldg(&dxtm1[d_offset+stride*i])*s_u[s_offset + i];


    }

    __syncthreads();

    // Store us in shared memory
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
      s_u[cta_size*k + tid] = us[k];

    __syncthreads();

    // Compute us
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;

      int pt_id_mod_p = pt_id%p;
      int pt_id_div_p = pt_id/p;

      int plane = pt_id/(p*p);
      int s_offset = plane*p*p + pt_id_mod_p;
      int d_offset = p*( (pt_id-plane*p*p)/p);
      int stride = p;

      #pragma unroll
      for (int i=0;i<p;i++)
        wa[k] += s_u[s_offset + stride*i]*__ldg(&dxm1[d_offset + i]);

    }

    __syncthreads();

    // Store ut in shared memory
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
      s_u[cta_size*k + tid] = ut[k];

    __syncthreads();

    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;
      int p_sq = p*p;

      int s_offset = pt_id%(p_sq);
      int plane = pt_id/p_sq;
      int d_offset = plane*p;
      int stride = p_sq;

      #pragma unroll
      for (int i=0;i<p;i++)
        wa[k] += s_u[s_offset + stride*i]*__ldg(&dxm1[d_offset + i]);
    }

    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;
      w[offset + pt_id] = wa[k];
    }


  } // Loop over blocks

}




  /*
   * Matrix-vector kernel
  */
  extern "C" 
  {
    void ax_e_cuda_(int *nx1,int *ny1,int *nz1,int *nelt,int *ldim, double *w, double *u,
                 double *g, double *dxm1, double *dxtm1) {

      int nxyz = (*nx1)*(*ny1)*(*nz1);
      int m1 = (*nx1);
      int m2 = m1*m1;

      int thrdsPBlck = 128;
      int numBlocks = (nxyz*(*nelt)+thrdsPBlck-1)/thrdsPBlck;

      //printf("nxyz=%d, numBlocks=%d, nelt=%d \n", nxyz, numBlocks, *nelt);

      if ( (*nx1) != (*ny1) || (*nx1) != (*nz1))
      {
        printf("non-cubic elements not supported in Cuda version\n");
        exit(1);
      }

      /* copy u to the gpu */

      copyTo(*nelt, nxyz, u);

      float time;
      cudaEvent_t start,stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);

      //printf("
      if ((*nx1)!=12)
      {
#if 0
        cudaMemcpy(d_w, d_u, (*nelt) * nxyz * sizeof(double), cudaMemcpyDeviceToDevice); 
#else
        /* Calculate ur, us, and ut */
        gsmxm<<<numBlocks, thrdsPBlck>>>(d_dxm1, m1, d_u, m1, d_ur, m2*(*nelt), 1);
        gsmxm<<<numBlocks, thrdsPBlck>>>(d_u, m1, d_dxtm1, m1, d_us, m1, (*nelt)*(*nz1));
        gsmxm<<<numBlocks, thrdsPBlck>>>(d_u, m2, d_dxtm1, m1, d_ut, m1, *nelt);

        /* calculate geom effects */

        geom<<<numBlocks, thrdsPBlck>>>(nxyz*(*nelt), d_ur, d_us, d_ut, d_g);

        /* calculate u from ur, us, ut */

        gsmxm<<<numBlocks,thrdsPBlck>>>(d_dxtm1, m1, d_ur, m1, d_w, m2*(*nelt), 1);
        gsmxm<<<numBlocks, thrdsPBlck>>>(d_us, m1, d_dxm1, m1, d_u, m1, (*nelt)*(*nz1));
        gadd2<<<numBlocks,thrdsPBlck>>>(d_w, d_u, nxyz*(*nelt));
        gsmxm<<<numBlocks, thrdsPBlck>>>(d_ut, m2, d_dxm1, m1, d_u, m1, *nelt);
        gadd2<<<numBlocks,thrdsPBlck>>>(d_w, d_u, nxyz*(*nelt));
#endif
      }
      else
      {
        // 12x12x12 case
        const int cta_size = 432;
        const int p = 12;
        const int p_sq = 12*12;
        const int p_cube = 12*12*12;
        const int p_cube_padded = p_cube;

        // We could play with this
        const int pts_per_thread = 4;  // 6*288 = 12*12*12
        const int slab_size = 3;

        const int grid_size = *nelt;

        //ax_cuda_kernel<p,p_cube,p_cube_padded,pts_per_thread,cta_size><<<grid_size,cta_size>>>(d_u, d_w, d_g, d_dxm1, d_dxtm1, *nelt);
        ax_cuda_kernel_v3<p,p_sq,p_cube,p_cube_padded,pts_per_thread,slab_size,cta_size><<<grid_size,cta_size>>>(d_u, d_w, d_g, d_dxm1, d_dxtm1, *nelt);
        //ax_cuda_kernel_v4<p,p_sq,p_cube,p_cube_padded,pts_per_thread,slab_size,cta_size><<<grid_size,cta_size>>>(d_u, d_w, d_g, d_dxm1, d_dxtm1, *nelt);
      }

      cudaCheckError();
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);

      cudaEventElapsedTime(&time, start, stop);
//      printf ("Time for the kernel: %f ms\n", time);
      /* copy w back to host */

      copyFrom(*nelt, nxyz, w);
//      cudaMemcpy(w, u, (*nelt) * nxyz * sizeof(double), cudaMemcpyHostToHost);
      return;
    }
  }


  /*
   * Initialize GPU operations
  */
  extern "C" 
  {
    void setup_cg_cuda_(int *nx1, int *ny1, int *nz1, int *nelt, int *ldim, double *g, double *dxm1, double *dxtm1) {

      int nxyz = (*nx1)*(*ny1)*(*nz1);

      cudaCheckError();
      /* malloc GPU memory for u, w, ur, us, ut, g, dxm1, dxtm1  */

      cudaError_t err = cudaMalloc((void **)&d_u, (*nelt)*nxyz*sizeof(double));
      if (err != cudaSuccess) {
        printf("cudamalloc error\n");
        exit(1);
      }
      err = cudaMalloc((void **)&d_w, (*nelt)*nxyz*sizeof(double));
      if (err != cudaSuccess) {
        printf("cudamalloc error\n");
        exit(1);
      }

      err = cudaMalloc((void **)&d_ur, (*nelt)*nxyz*sizeof(double));
      if (err != cudaSuccess) {
        printf("cudamalloc error\n");
        exit(1);
      }
      err = cudaMalloc((void **)&d_us, (*nelt)*nxyz*sizeof(double));
      if (err != cudaSuccess) {
        printf("cudamalloc error\n");
        exit(1);
      }
      err = cudaMalloc((void **)&d_ut, (*nelt)*nxyz*sizeof(double));
      if (err != cudaSuccess) {
        printf("cudamalloc error\n");
        exit(1);
      }

      err = cudaMalloc((void **)&d_g, (*nelt)*nxyz*2*(*ldim)*sizeof(double));
      if (err != cudaSuccess) {
        printf("cudamalloc error\n");
        exit(1);
      }
      err = cudaMalloc((void **)&d_dxm1, (*nx1)*(*nx1)*sizeof(double));
      if (err != cudaSuccess) {
        printf("cudamalloc error\n");
        exit(1);
      }
      err = cudaMalloc((void **)&d_dxtm1, (*nx1)*(*nx1)*sizeof(double));
      if (err != cudaSuccess) {
        printf("cudamalloc error\n");
        exit(1);
      }

      /* copy g, dxm1, dxtm1 to the gpu */

      err = cudaMemcpy(d_g, g, (*nelt)*nxyz*2*(*ldim)*sizeof(double), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        printf("cudaMemcpy error\n");
        exit(1);
      }
      err = cudaMemcpy(d_dxm1, dxm1, (*nx1)*(*nx1)*sizeof(double), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        printf("cudaMemcpy error\n");
        exit(1);
      }
      err = cudaMemcpy(d_dxtm1, dxtm1, (*nx1)*(*nx1)*sizeof(double), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        printf("cudaMemcpy error\n");
        exit(1);
      }

      return;
    }
  }


  /*
   * Clean up after GPU operations
  */
  extern "C" 
  {
    void teardown_cg_cuda_() {

      /* free GPU memory for u, w, ur, us, ut, g, dxm1, dxtm1  */

      cudaError_t err = cudaFree(d_u);
      if (err != cudaSuccess) {
        printf("cudaFree error\n");
        exit(1);
      }
      err = cudaFree(d_w);
      if (err != cudaSuccess) {
        printf("cudaFree error\n");
        exit(1);
      }

      err = cudaFree(d_ur);
      if (err != cudaSuccess) {
        printf("cudaFree error\n");
        exit(1);
      }
      err = cudaFree(d_us);
      if (err != cudaSuccess) {
        printf("cudaFree error\n");
        exit(1);
      }
      err = cudaFree(d_ut);
      if (err != cudaSuccess) {
        printf("cudaFree error\n");
        exit(1);
      }

      err = cudaFree(d_g);
      if (err != cudaSuccess) {
        printf("cudaFree error\n");
        exit(1);
      }
      err = cudaFree(d_dxm1);
      if (err != cudaSuccess) {
        printf("cudaFree error\n");
        exit(1);
      }
      err = cudaFree(d_dxtm1);
      if (err != cudaSuccess) {
        printf("cudaFree error\n");
        exit(1);
      }


      return;
    }
  }
