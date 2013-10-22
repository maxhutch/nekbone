
#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda_util.h"
#include "cublas_v2.h"
#include "ld_functions.h"
#include "sm_utils.inl"
#include "thrust/device_ptr.h"
#include "thrust/reduce.h"

#include "cgcuda.h"

cublasHandle_t cublas_handle;

//#define AOS

struct gpu_domain
{
  double *d_x; 
  double *d_f; 
  double *d_g; // metric terms
  double *d_c;
  double *d_r;
  double *d_w;
  double *d_p; 
  double *d_z;
  double *d_dxm1; // differentiation matrix D
  double *d_dxtm1; // D^T
  double *d_temp; // temporary array

  double *reduced_value;
  cudaEvent_t reduced_value_event;

  int nx1; 
  int ny1; 
  int nz1; 
  int nelt; 
  int ldim; 
  int nxyz;
  int nid;
  int niter;

  uint size_from[2];
  uint size_to[2];
  uint* d_map_offsets[2];
  uint* d_map_indices_from[2];
  uint* d_map_indices_from_COO[2];
  uint* d_map_indices_to[2];
  uint* d_flagged_primaries;

};
 /* GPU data pointers */

static gpu_domain gpu_dom;

void fill_gpu_maps(uint* size_from, uint* size_to, uint** d_map_offsets, uint** d_map_indices_from, uint** d_map_indices_from_COO, uint** d_map_indices_to, const uint* map)
{

  cudaCheckError();
  const uint* orig = map;

  // First compute the size of the map_indices_to and map_indices_from arrays
  // TODO: Can probably get that from "nz" array
  uint size_from_tmp = 0;
  uint size_to_tmp = 0;
  while( *map++ != -(uint)1 )                                              
  {
    *map++;  
    size_from_tmp++;
    do { size_to_tmp++; } while( (*map++) != -(uint)1 ); 
  } 
#ifdef DEBUG
  printf("size_from = %d, size_to=%d\n",size_from_tmp,size_to_tmp);
#endif

  // Fill host arrays first
  uint* h_map_offsets = (uint*) malloc((size_from_tmp+1)*sizeof(uint));
  uint* h_map_indices_from = (uint*) malloc(size_from_tmp*sizeof(uint));
  uint* h_map_indices_to = (uint*) malloc(size_to_tmp*sizeof(uint));
  uint* h_map_indices_from_COO = (uint*) malloc(size_to_tmp*sizeof(uint));

  uint i,j;
  uint count_from=0;
  uint count_to=0;
  h_map_offsets[0] = 0;

  cudaCheckError();
  map = orig;
  while((i=*map++)!=-(uint)1) 
  { 
    uint row_length = 0;
    h_map_indices_from[count_from++] = i;
    j=*map++; 
    do 
    { 
      h_map_indices_to[count_to] = j;
      h_map_indices_from_COO[count_to++] = i;
      row_length++;
    }
    while((j=*map++)!=-(uint)1);
  
    h_map_offsets[count_from] = count_to; 
  } 

  cudaCheckError();
#ifdef DEBUG
  printf("size_from=%d\n",size_from_tmp);
  printf("size_to=%d\n",size_to_tmp);
#endif
  cudaMalloc((void **) d_map_offsets, (size_from_tmp+1)*sizeof(uint));
  cudaMalloc((void **) d_map_indices_from, size_from_tmp*sizeof(uint));
  cudaMalloc((void **) d_map_indices_to, size_to_tmp*sizeof(uint));
  cudaMalloc((void **) d_map_indices_from_COO, size_to_tmp*sizeof(uint));

  cudaMemcpy(*d_map_offsets, h_map_offsets, (size_from_tmp+1)*sizeof(uint), cudaMemcpyHostToDevice);
  cudaMemcpy(*d_map_indices_from, h_map_indices_from, size_from_tmp*sizeof(uint), cudaMemcpyHostToDevice);
  cudaMemcpy(*d_map_indices_to, h_map_indices_to, size_to_tmp*sizeof(uint), cudaMemcpyHostToDevice);
  cudaMemcpy(*d_map_indices_from_COO, h_map_indices_from_COO, size_to_tmp*sizeof(uint), cudaMemcpyHostToDevice);


  *size_from = size_from_tmp;
  *size_to = size_to_tmp;
  cudaCheckError();

  //printf("map_offsets\n");
  //for (int i=0;i<(size_from+1);i++)
  //  printf("%d\n",h_map_offsets[i]);
  //printf("map_indices_from\n");
  //for (int i=0;i<(size_from);i++)
  //  printf("%d\n",h_map_indices_from[i]);

  //printf("map_indices_to\n");
  //for (int i=0;i<(size_to);i++)
  //  printf("%d\n",h_map_indices_to[i]);

}

void fill_flagged_primaries_map(uint* d_flagged_primaries, const uint* flagged_primaries)
{

}

template <typename T>
__global__
void local_gather_kernel(T* __restrict__ out, const T* __restrict__ in, const uint* __restrict__ offsets, const uint* __restrict__ map_indices_from, const uint* __restrict__ map_indices_to, int size  )
{
  for (int tid = blockDim.x*blockIdx.x + threadIdx.x; tid < size; tid+= gridDim.x * blockDim.x) 
  {
    uint store_loc = map_indices_from[tid];
    T t = out[store_loc]; 
    for (int i=offsets[tid];i<offsets[tid+1];i++)
    {
      t += in[map_indices_to[i]];
    }
    out[store_loc] = t;
  }
}

template <typename T>
__global__
void local_scatter_kernel(T* __restrict__ out, const T* __restrict__ in, const uint* __restrict__ offsets, const uint* __restrict__ map_indices_from, const uint* __restrict__ map_indices_to, int size  )
{
  for (int tid = blockDim.x*blockIdx.x + threadIdx.x; tid < size; tid+= gridDim.x * blockDim.x) 
  {
    T t = in[map_indices_from[tid]]; 

    for (int i=offsets[tid];i<offsets[tid+1];i++)
    {
      out[map_indices_to[i]] = t;  
    }
  }
}


template <typename T>
__global__
void local_scatter_kernel_COO(T* __restrict__ out, const T* __restrict__ in,  const uint* __restrict__ map_indices_from_COO, const uint* __restrict__ map_indices_to, int size  )
{
  for (int tid = blockDim.x*blockIdx.x + threadIdx.x; tid < size; tid+= gridDim.x * blockDim.x) 
  {
    T t = in[map_indices_from_COO[tid]]; 
    out[map_indices_to[tid]] = t;  
  }
}

void local_gather_cuda(double* out, const double* in, 
                       const uint *map_offsets, const uint* map_indices_from, const uint* map_indices_from_COO, const uint* map_indices_to, int size_from, int size_to)
{

  const int cta_size= 128;
  const int grid_size = min(4096,(size_from+cta_size-1)/cta_size);
  cudaCheckError();
  local_gather_kernel<<<grid_size,cta_size>>>(out,in,map_offsets,map_indices_from,map_indices_to,size_from);
  cudaCheckError();
}

void local_scatter_cuda(double* out, const double* in,
                       const uint *map_offsets, const uint* map_indices_from, const uint* map_indices_from_COO, const uint* map_indices_to, int size_from, int size_to)
{

  //const int cta_size= 128;
  //const int grid_size = min(4096,(size_from+cta_size-1)/cta_size);
  //local_scatter_kernel<<<grid_size,cta_size>>>(out,in,map_offsets,map_indices_from,map_indices_to,size_from );
  //cudaCheckError();

  const int cta_size= 128;
  const int grid_size = min(4096,(size_to+cta_size-1)/cta_size);
  local_scatter_kernel_COO<<<grid_size,cta_size>>>(out,in,map_indices_from_COO,map_indices_to,size_to);
  cudaCheckError();

}

__global__
void mask_kernel(double* w, int size)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if (tid < size)
  {
    w[tid] = 0.;
  }
}

#if 0
  /*
   * Copy data to the GPU
  */
  void copyTo(int nelt, int nxyz, double *u) {

    cudaCheckError();
    cudaMemcpy(d_u, u, nelt*nxyz*sizeof(double), cudaMemcpyHostToDevice);
    cudaCheckError();

    return; 
  }


  /*
   * Copy data from the GPU
  */
template<typename T>
  void copyFrom(int n, T *vec, T *d_vec) {

    cudaCheckError();
    cudaMemcpy(vec, d_vec, n*sizeof(T), cudaMemcpyDeviceToHost);
    cudaCheckError();

    return;
  }
#endif


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


template<int p, int p_sq, int p_cube, int p_cube_padded, int pts_per_thread, int slab_size, int cta_size, int num_ctas>
__global__
__launch_bounds__(cta_size,num_ctas)
void ax_cuda_kernel_v8_shared_D(const double* __restrict__ u_global, double* __restrict__ w, const double* __restrict__ g, const double* __restrict__ dxm1, const double* __restrict__ dxtm1, int n_cells)
{
  int tid = threadIdx.x;
  __shared__ double temp[cta_size];
  __shared__ double s_dxm1[p_sq];
  __shared__ double s_dxtm1[p_sq];

  for (int cell_id=blockIdx.x; cell_id < n_cells; cell_id += gridDim.x)
  {
    // Load u in shared for the entire cell
    int offset = cell_id*p_cube;

    int tid_mod_p = tid%p;
    int tid_div_p = tid/p;
    int tid_mod_p_sq = tid%p_sq;
    int tid_div_p_sq = tid/p_sq;

    double u[pts_per_thread];
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;

      u[k] = ld_functions::ld_cg(&u_global[offset + pt_id]);
    }

    // Store dxm1 and dxtm1 in shared memory
    if (tid < p_sq)
    {
      s_dxm1[tid] = ld_functions::ld_cg(&dxm1[tid]);
      s_dxtm1[tid] = ld_functions::ld_cg(&dxtm1[tid]);
    }


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
      //int pt_id_div_p = pt_id/p;
      //int pt_id_mod_p = pt_id%p;
      int pt_id_div_p_sq = pt_id/p_sq;
      //int pt_id_mod_p_sq = pt_id%p_sq;

      double ur, us, ut;

      // Load first slab in shared memory
      __syncthreads();
      temp[tid] = u[k];
      __syncthreads();
      

      //  Now that data is loaded in shared, compute ur
      {
        int s_offset = tid_div_p*p;
        int d_offset  = tid_mod_p;

        ur = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          ur += s_dxm1[d_offset + p*i]*temp[s_offset + i];
          //ur += __ldg(&dxm1[d_offset + p*i])*temp[s_offset + i];
      }

      // Compute us
      {
        int plane = tid_div_p_sq;
        int s_offset = plane*p_sq + tid_mod_p;
        int d_offset = p*( (tid-plane*p_sq)/p);

        us = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          us += temp[s_offset + p*i]*s_dxtm1[d_offset + i];
         // us += temp[s_offset + p*i]*__ldg(&dxtm1[d_offset + i]);
      }


      // Load all slabs in shared, one by one to compute ut
      ut = 0.;
      #pragma unroll
      for (int k2=0;k2<pts_per_thread;k2++)
      {
        int i_start = k2*slab_size;

        // Load in shared
        __syncthreads();
        temp[tid] = u[k2];
        __syncthreads();

        // Compute ut
        int s_offset = tid_mod_p_sq;
        int d_offset = pt_id_div_p_sq*p;

        #pragma unroll
        for (int icount=0;icount<slab_size;icount++)
        {
          //ut += temp[s_offset + p_sq*icount]*__ldg(&dxtm1[d_offset + i_start]);
          ut += temp[s_offset + p_sq*icount]*s_dxtm1[d_offset + i_start];
          i_start++;
        }
      }

      // Transform
      {


        /*
        int offset = (cell_id*p_cube + pt_id)*6;
        //TODO: Switch to SOA
        double metric[6];
        #pragma unroll
        for (int i=0;i<6;i++)
          metric[i] = g[offset+i];

        */

        // AoS version
#ifdef AOS
        int offset = cell_id*p_cube + pt_id;
        //TODO: Switch to SOA
        double metric[6];
        #pragma unroll
        for (int i=0;i<6;i++)
          metric[i] = g[offset+i*n_cells*p_cube];
#else
        int offset = (cell_id*p_cube + pt_id)*6;
        double metric[6];
        #pragma unroll
        for (int i=0;i<6;i++)
          metric[i] = g[offset+i];
#endif
          //metric[i] = ld_functions::ld_cg(&g[offset+i*n_cells*p_cube]);
          //metric[i] = g[offset+i*n_cells*p_cube];

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
      __syncthreads();
      temp[tid] = ur;
      __syncthreads();

      // Now that data is loaded in shared, compute wa

      {
        int d_offset  = tid_mod_p;
        int s_offset = tid_div_p*p;

        #pragma unroll
        for (int i=0;i<p;i++)
          wa[k] += s_dxtm1[d_offset+p*i]*temp[s_offset + i];
          //wa[k] += __ldg(&dxtm1[d_offset+p*i])*temp[s_offset + i];
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
          wa[k] += temp[s_offset + p*i]*s_dxm1[d_offset + i];
          //wa[k] += temp[s_offset + p*i]*__ldg(&dxm1[d_offset + i]);
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
          wa[k2] += temp[s_offset + p_sq*i_count]*s_dxm1[d_offset + i_start];
          //wa[k2] += temp[s_offset + p_sq*i_count]*__ldg(&dxm1[d_offset + i_start]);
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




template<int p, int p_sq, int p_cube, int p_cube_padded, int pts_per_thread, int slab_size, int cta_size, int num_ctas>
__global__
__launch_bounds__(cta_size,num_ctas)
void ax_cuda_kernel_v8(const double* __restrict__ u_global, double* __restrict__ w, const double* __restrict__ g, const double* __restrict__ dxm1, const double* __restrict__ dxtm1, int n_cells)
{
  int tid = threadIdx.x;
  __shared__ double temp[cta_size];

  for (int cell_id=blockIdx.x; cell_id < n_cells; cell_id += gridDim.x)
  {
    // Load u in shared for the entire cell
    int offset = cell_id*p_cube;

    int tid_mod_p = tid%p;
    int tid_div_p = tid/p;
    int tid_mod_p_sq = tid%p_sq;
    int tid_div_p_sq = tid/p_sq;

    double u[pts_per_thread];
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;

      u[k] = ld_functions::ld_cg(&u_global[offset + pt_id]);
    }

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
      int pt_id_div_p = pt_id/p;
      int pt_id_mod_p = pt_id%p;
      int pt_id_div_p_sq = pt_id/p_sq;
      int pt_id_mod_p_sq = pt_id%p_sq;

      double ur, us, ut;

      // Load first slab in shared memory
      __syncthreads();
      temp[tid] = u[k];
      __syncthreads();
      

      //  Now that data is loaded in shared, compute ur
      {
        int s_offset = tid_div_p*p;
        int d_offset  = tid_mod_p;

        ur = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          ur += __ldg(&dxm1[d_offset + p*i])*temp[s_offset + i];
      }

      // Compute us
      {
        int plane = tid_div_p_sq;
        int s_offset = plane*p_sq + tid_mod_p;
        int d_offset = p*( (tid-plane*p_sq)/p);

        us = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          us += temp[s_offset + p*i]*__ldg(&dxtm1[d_offset + i]);
      }


      // Load all slabs in shared, one by one to compute ut
      ut = 0.;
      #pragma unroll
      for (int k2=0;k2<pts_per_thread;k2++)
      {
        int i_start = k2*slab_size;

        // Load in shared
        __syncthreads();
        temp[tid] = u[k2];
        __syncthreads();

        // Compute ut
        int s_offset = tid_mod_p_sq;
        int d_offset = pt_id_div_p_sq*p;

        #pragma unroll
        for (int icount=0;icount<slab_size;icount++)
        {
          ut += temp[s_offset + p_sq*icount]*__ldg(&dxtm1[d_offset + i_start]);
          i_start++;
        }
      }

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
      __syncthreads();
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
//__launch_bounds__(288,3)
void ax_cuda_kernel_v7(const double* __restrict__ u_global, double* __restrict__ w, const double* __restrict__ g, const double* __restrict__ dxm1, const double* __restrict__ dxtm1, int n_cells)
{
  int tid = threadIdx.x;
  __shared__ double temp[cta_size];

  for (int cell_id=blockIdx.x; cell_id < n_cells; cell_id += gridDim.x)
  {
    // Load u in shared for the entire cell
    int offset = cell_id*p_cube;

    int tid_mod_p = tid%p;
    int tid_div_p = tid/p;
    int tid_mod_p_sq = tid%p_sq;
    int tid_div_p_sq = tid/p_sq;

    double u[p];
    #pragma unroll
    for (int k=0;k<pts_per_thread;k++)
    {
      int pt_id = k*cta_size + tid;

      __syncthreads();
      temp[tid] = ld_functions::ld_cg(&u_global[offset + pt_id]);
      __syncthreads();

      // Each thread stores the values of u along z
      #pragma unroll
      for (int j=0;j<slab_size;j++)
      {
        u[k*slab_size+j] = temp[j*p_sq + tid_mod_p_sq];
      }
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

      // Load first slab in shared memory
      __syncthreads();
      temp[tid] = u[pt_id_div_p_sq];
      __syncthreads();
      

      //  Now that data is loaded in shared, compute ur
      {
        int s_offset = tid_div_p*p;
        int d_offset  = tid_mod_p;

        ur = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          ur += __ldg(&dxm1[d_offset + p*i])*temp[s_offset + i];
      }

      // Compute us
      {
        int plane = tid_div_p_sq;
        int s_offset = plane*p_sq + tid_mod_p;
        int d_offset = p*( (tid-plane*p_sq)/p);

        us = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          us += temp[s_offset + p*i]*__ldg(&dxtm1[d_offset + i]);
      }

      // Compute ut
      {
        int d_offset = pt_id_div_p_sq*p;

        ut = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
        {
          ut += u[i]*__ldg(&dxtm1[d_offset + i]);
        }
      }

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
//__launch_bounds__(288,3)
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
        {
          ut += s_u[s_offset + p_sq*i]*__ldg(&dxtm1[d_offset + i]);
        }
      }

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

template<int p, int p_sq, int p_cube, int p_cube_padded, int cta_size>
__global__
void ax_cuda_kernel_v6(const double* __restrict__ u_global, double* __restrict__ w, const double* __restrict__ g, const double* __restrict__ dxm1, const double* __restrict__ dxtm1, int n_cells)
{
  int tid = threadIdx.x;
  __shared__ double temp[cta_size];

  double u[p], wa[p];

  for (int cell_id=blockIdx.x; cell_id < n_cells; cell_id += gridDim.x)
  {
    // Load u in shared for the entire cell
    int offset = cell_id*p_cube;
    #pragma unroll
    for (int k=0;k<p;k++)
    {
      int pt_id = k*cta_size + tid;
      u[k] = ld_functions::ld_cg(&u_global[offset + pt_id]);
    }

    __syncthreads();

    // Initialize wa to 0.
    #pragma unroll
    for (int k=0;k<p;k++)
      wa[k] = 0.;

    // Now compute w for one slab at a time
    #pragma unroll
    for (int k=0;k<p;k++)
    {
      int pt_id = k*cta_size + tid;
      int pt_id_div_p = pt_id/p;
      int pt_id_mod_p = pt_id%p;
      int pt_id_div_p_sq = pt_id/p_sq;
      int pt_id_mod_p_sq = pt_id%p_sq;

      double ur, us, ut;

      // Load in shared
      __syncthreads();
      temp[tid] = u[k];
      __syncthreads();

      //  Now that data is loaded in shared, compute ur
      {
        int s_offset = pt_id_div_p*p;
        int d_offset  = pt_id_mod_p;

        ur = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          ur += __ldg(&dxm1[d_offset + p*i])*temp[s_offset + i];
      }

      // Compute us
      {
        int plane = pt_id_div_p_sq;
        int s_offset = plane*p_sq + pt_id_mod_p;
        int d_offset = p*( (pt_id-plane*p_sq)/p);

        us = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          us += temp[s_offset + p*i]*__ldg(&dxtm1[d_offset + i]);
      }

      // Compute ut
      {
        int s_offset = pt_id_mod_p_sq;
        int d_offset = pt_id_div_p_sq*p;

        ut = 0.;
        #pragma unroll
        for (int i=0;i<p;i++)
          ut += u[i]*__ldg(&dxtm1[d_offset + i]);
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
      __syncthreads();
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
      for (int k2=0;k2<p;k2++)
      {
        int pt_id_2 = k2*cta_size + tid;
        int plane = pt_id_2/p_sq;

        int s_offset = tid_mod_p_sq;
        int d_offset = plane*p;

        wa[k2] += temp[s_offset]*__ldg(&dxm1[d_offset + k]);
      }
      __syncthreads();

    } // Loop over k

    #pragma unroll
    for (int k=0;k<p;k++)
    {
      int pt_id = k*cta_size + tid;
      w[offset + pt_id] = wa[k];
    }
  } // Loop over blocks

}


template<int p, int p_sq, int p_cube, int p_cube_padded, int pts_per_thread, int slab_size, int cta_size>
__global__
__launch_bounds__(432,2)
//__launch_bounds__(144,6)
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
template<int p, int p_cube, int p_cube_padded, int cta_size>
__global__
//__launch_bounds__(729,1)
void ax_cuda_kernel_v9(double* __restrict__ u, double* __restrict__ w, double* __restrict__ g, double* __restrict__ dxm1, double* __restrict__ dxtm1, int n_cells)
{
  int tid = threadIdx.x;

  __shared__ double s_u[p_cube];

  for (int cell_id=blockIdx.x; cell_id < n_cells; cell_id += gridDim.x)
  {
    int cell_id = blockIdx.x;

    double ur,us,ut,wa;

    int offset = cell_id*p_cube;
    s_u[tid] = u[offset + tid];

    __syncthreads();

    int tid_mod_p = tid%p;
    int tid_div_p = tid/p;
    int p_sq = p*p;
    int tid_mod_psq = tid%p_sq;
    int tid_div_psq = tid/p_sq;

    //  Now that data is loaded in shared, compute ur
    {
      int line = tid_div_p;
      int s_offset = line*p;

      int d_offset  = tid_mod_p;
      int stride = p;

      ur = 0.;
      #pragma unroll
      for (int i=0;i<p;i++)
        ur += dxm1[d_offset + stride*i]*s_u[s_offset + i];
    }

    // Compute us
    {
      int plane = tid_div_psq;
      int s_offset = plane*p_sq + tid_mod_p;
      int d_offset = p*( (tid-plane*p_sq)/p);
      int stride = p;

      us = 0.;
      #pragma unroll
      for (int i=0;i<p;i++)
        us += s_u[s_offset + stride*i]*__ldg(&dxtm1[d_offset + i]);

    }

    // Compute ut
    {

      int s_offset = tid_mod_psq;
      int plane = tid_div_psq;
      int d_offset = plane*p;
      int stride = p_sq;

      ut = 0.;
      #pragma unroll
      for (int i=0;i<p;i++)
        ut += s_u[s_offset + stride*i]*__ldg(&dxtm1[d_offset + i]);
    }

    // Transform
    {
      int offset = (cell_id*p_cube + tid)*6;

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

    // Wait for all threads to be done with data in s_u
    __syncthreads();

    s_u[tid] = ur;
    __syncthreads();

    // Wait for all threads to have loaded data in s_u

    // Now that data is loaded in shared, compute wa
    {
      int d_offset  = tid_mod_p;
      int line = tid_div_p;
      int s_offset = line*p;
      int stride = p;

      wa = 0.;
      #pragma unroll
      for (int i=0;i<p;i++)
        wa += __ldg(&dxtm1[d_offset+stride*i])*s_u[s_offset + i];
    }

    __syncthreads();

    // Store us in shared memory
    s_u[tid] = us;

    __syncthreads();

    // Compute us
    {
      int plane = tid_div_psq;
      int s_offset = plane*p_sq + tid_mod_p;
      int d_offset = p*( (tid-plane*p_sq)/p);
      int stride = p;

      #pragma unroll
      for (int i=0;i<p;i++)
        wa += s_u[s_offset + stride*i]*__ldg(&dxm1[d_offset + i]);
    }

    __syncthreads();

    // Store ut in shared memory
    s_u[tid] = ut;

    __syncthreads();

    {
      int s_offset = tid_mod_psq;
      int plane = tid_div_psq;
      int d_offset = plane*p;
      int stride = p_sq;

      #pragma unroll
      for (int i=0;i<p;i++)
        wa += s_u[s_offset + stride*i]*__ldg(&dxm1[d_offset + i]);
    }

    w[offset + tid] = wa;


  } // Loop over blocks

}



template<int p, int p_cube, int p_cube_padded, int pts_per_thread, int cta_size>
__global__
//__launch_bounds__(864,1)
__launch_bounds__(729,1)
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


template<int p, int p_cube, int p_cube_padded, int pts_per_thread, int cta_size>
__global__
__launch_bounds__(729,2)
void ax_cuda_kernel_v12(double* __restrict__ u, double* __restrict__ w, double* __restrict__ g, double* __restrict__ dxm1, double* __restrict__ dxtm1, int n_cells)
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


template<int p, int p_cube, int p_cube_padded, int cta_size>
__global__
//__launch_bounds__(864,1)
//__launch_bounds__(729,1)
void ax_cuda_kernel_v11(double* __restrict__ u, double* __restrict__ w, double* __restrict__ g, double* __restrict__ dxm1, double* __restrict__ dxtm1, int n_cells)
{
  int tid = threadIdx.x;

  __shared__ double s_u[p_cube];

  for (int cell_id=blockIdx.x; cell_id < n_cells; cell_id += gridDim.x)
  {
    int cell_id = blockIdx.x;

    double ur[1];
    double us[1];
    double ut[1];

    int offset = cell_id*p_cube;
    {
      int pt_id =  tid;
      s_u[pt_id] = u[offset + pt_id];
    }

    __syncthreads();

    //  Now that data is loaded in shared, compute ur
    {
      int pt_id = tid;

      int line = pt_id/p;
      int s_offset = line*p;

      int d_offset  = pt_id%p;
      int stride = p;

      ur[0] = 0.;
      #pragma unroll
      for (int i=0;i<p;i++)
        ur[0] += dxm1[d_offset + stride*i]*s_u[s_offset + i];
    }

    // Compute us
    {
      int pt_id = tid;

      int pt_id_mod_p = pt_id%p;
      int pt_id_div_p = pt_id/p;

      int plane = pt_id/(p*p);
      int s_offset = plane*p*p + pt_id_mod_p;
      int d_offset = p*( (pt_id-plane*p*p)/p);
      int stride = p;

      us[0] = 0.;
      #pragma unroll
      for (int i=0;i<p;i++)
        us[0] += s_u[s_offset + stride*i]*__ldg(&dxtm1[d_offset + i]);

    }

    // Compute ut
    {
      int pt_id = tid;
      int p_sq = p*p;

      int s_offset = pt_id%(p_sq);
      int plane = pt_id/p_sq;
      int d_offset = plane*p;
      int stride = p_sq;

      ut[0] = 0.;
      #pragma unroll
      for (int i=0;i<p;i++)
        ut[0] += s_u[s_offset + stride*i]*__ldg(&dxtm1[d_offset + i]);
    }

    // Transform
    {
      int pt_id = tid;
      int offset = (cell_id*p_cube + pt_id)*6;

      //TODO: Switch to SOA
      double metric[6];
      #pragma unroll
      for (int i=0;i<6;i++)
        metric[i] = g[offset+i];

      double wr = metric[0]*ur[0] + metric[1]*us[0] + metric[2]*ut[0];
      double ws = metric[1]*ur[0] + metric[3]*us[0] + metric[4]*ut[0];
      double wt = metric[2]*ur[0] + metric[4]*us[0] + metric[5]*ut[0];

      ur[0] = wr;
      us[0] = ws;
      ut[0] = wt;
    }

    // Store ur in shared memory

    // Wait for all threads to be done with data in s_u
    __syncthreads();

      s_u[tid] = ur[0];

    // Wait for all threads to have loaded data in s_u
    __syncthreads();

    double wa[1];

    // Now that data is loaded in shared, compute wa
    {
      int pt_id = tid;

      int d_offset  = pt_id%p;
      int line = pt_id/p;
      int s_offset = line*p;
      int stride = p;

      wa[0] = 0.;
      #pragma unroll
      for (int i=0;i<p;i++)
        wa[0] += __ldg(&dxtm1[d_offset+stride*i])*s_u[s_offset + i];


    }

    __syncthreads();

    // Store us in shared memory
      s_u[tid] = us[0];

    __syncthreads();

    // Compute us
    {
      int pt_id = tid;

      int pt_id_mod_p = pt_id%p;
      int pt_id_div_p = pt_id/p;

      int plane = pt_id/(p*p);
      int s_offset = plane*p*p + pt_id_mod_p;
      int d_offset = p*( (pt_id-plane*p*p)/p);
      int stride = p;

      #pragma unroll
      for (int i=0;i<p;i++)
        wa[0] += s_u[s_offset + stride*i]*__ldg(&dxm1[d_offset + i]);

    }

    __syncthreads();

    // Store ut in shared memory
      s_u[tid] = ut[0];

    __syncthreads();

    {
      int pt_id = tid;
      int p_sq = p*p;

      int s_offset = pt_id%(p_sq);
      int plane = pt_id/p_sq;
      int d_offset = plane*p;
      int stride = p_sq;

      #pragma unroll
      for (int i=0;i<p;i++)
        wa[0] += s_u[s_offset + stride*i]*__ldg(&dxm1[d_offset + i]);
    }

    {
      int pt_id = tid;
      w[offset + pt_id] = wa[0];
    }


  } // Loop over blocks

}



template<int p, int p_cube, int p_cube_padded,  int cta_size>
__global__
//__launch_bounds__(864,1)
__launch_bounds__(729,1)
void ax_cuda_kernel_v10(double* __restrict__ u, double* __restrict__ w, double* __restrict__ g, double* __restrict__ dxm1, double* __restrict__ dxtm1, int n_cells)
{
  int tid = threadIdx.x;

  __shared__ double s_u[p_cube];

  for (int cell_id=blockIdx.x; cell_id < n_cells; cell_id += gridDim.x)
  {
    int cell_id = blockIdx.x;

    double ur;
    double us;
    double ut;

    int offset = cell_id*p_cube;
    s_u[tid] = u[offset + tid];

    __syncthreads();

    //  Now that data is loaded in shared, compute ur
    {
      int pt_id = tid;

      int line = pt_id/p;
      int s_offset = line*p;

      int d_offset  = pt_id%p;
      int stride = p;

      ur = 0.;
      #pragma unroll
      for (int i=0;i<p;i++)
        ur += dxm1[d_offset + stride*i]*s_u[s_offset + i];
    }

    // Compute us
    {
      int pt_id = tid;

      int pt_id_mod_p = pt_id%p;
      int pt_id_div_p = pt_id/p;

      int plane = pt_id/(p*p);
      int s_offset = plane*p*p + pt_id_mod_p;
      int d_offset = p*( (pt_id-plane*p*p)/p);
      int stride = p;

      us = 0.;
      #pragma unroll
      for (int i=0;i<p;i++)
        us += s_u[s_offset + stride*i]*__ldg(&dxtm1[d_offset + i]);

    }

    // Compute ut
    {
      int pt_id = tid;
      int p_sq = p*p;

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
      int pt_id =  tid;
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

    // Wait for all threads to be done with data in s_u
    __syncthreads();

    s_u[tid] = ur;

    // Wait for all threads to have loaded data in s_u
    __syncthreads();

    double wa;

    // Now that data is loaded in shared, compute wa
    {
      int pt_id = tid;

      int d_offset  = pt_id%p;
      int line = pt_id/p;
      int s_offset = line*p;
      int stride = p;

      wa = 0.;
      #pragma unroll
      for (int i=0;i<p;i++)
        wa += __ldg(&dxtm1[d_offset+stride*i])*s_u[s_offset + i];


    }

    __syncthreads();

    // Store us in shared memory
      s_u[tid] = us;

    __syncthreads();

    // Compute us
    {
      int pt_id = tid;

      int pt_id_mod_p = pt_id%p;
      int pt_id_div_p = pt_id/p;

      int plane = pt_id/(p*p);
      int s_offset = plane*p*p + pt_id_mod_p;
      int d_offset = p*( (pt_id-plane*p*p)/p);
      int stride = p;

      #pragma unroll
      for (int i=0;i<p;i++)
        wa += s_u[s_offset + stride*i]*__ldg(&dxm1[d_offset + i]);

    }

    __syncthreads();

    // Store ut in shared memory
      s_u[tid] = ut;

    __syncthreads();

    {
      int pt_id = tid;
      int p_sq = p*p;

      int s_offset = pt_id%(p_sq);
      int plane = pt_id/p_sq;
      int d_offset = plane*p;
      int stride = p_sq;

      #pragma unroll
      for (int i=0;i<p;i++)
        wa += s_u[s_offset + stride*i]*__ldg(&dxm1[d_offset + i]);
    }

    w[offset + tid] = wa;


  } // Loop over blocks

}



  /*
   * Matrix-vector kernel
  */
void axcuda_e(double *w, double *u, double *g, double *dxm1, double *dxtm1, 
              int nx1, int ny1, int nz1, int nelt, int ldim) 
{
      //int thrdsPBlck = 128;
      //int numBlocks = (nxyz*nelt+thrdsPBlck-1)/thrdsPBlck;

      if ( nx1 != ny1 || nx1 != nz1)
      {
        printf("non-cubic elements not supported in Cuda version\n");
        exit(1);
      }

      cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
      cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

      float time;
      cudaEvent_t start,stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);

      //printf("
      if ( nx1 != 12 && nx1 != 9 )
      {
        printf("Current implementation only tested for polynomial orders 9 and 12, exiting");
        exit(1);
#if 0
        /* Calculate ur, us, and ut */

        gsmxm<<<numBlocks, thrdsPBlck>>>(dxm1, m1, u, m1, ur, m2*(*nelt), 1);
        gsmxm<<<numBlocks, thrdsPBlck>>>(u, m1, dxtm1, m1, us, m1, (*nelt)*(*nz1));
        gsmxm<<<numBlocks, thrdsPBlck>>>(u, m2, dxtm1, m1, ut, m1, *nelt);

        /* calculate geom effects */

        geom<<<numBlocks, thrdsPBlck>>>(nxyz*(*nelt), ur, us, ut, g);

        /* calculate u from ur, us, ut */

        gsmxm<<<numBlocks,thrdsPBlck>>>(dxtm1, m1, ur, m1, w, m2*(*nelt), 1);
        gsmxm<<<numBlocks, thrdsPBlck>>>(us, m1, dxm1, m1, u, m1, (*nelt)*(*nz1));
        gadd2<<<numBlocks,thrdsPBlck>>>(w, u, nxyz*(*nelt));
        gsmxm<<<numBlocks, thrdsPBlck>>>(ut, m2, dxm1, m1, u, m1, *nelt);
        gadd2<<<numBlocks,thrdsPBlck>>>(w, u, nxyz*(*nelt));
#endif

      }
      else if ( nx1 == 9 )
      {

#if 0 
        const int grid_size = *nelt;
        const int p = 9;
        const int p_sq = 9*9;
        const int p_cube = 9*9*9;
        const int p_cube_padded = p_cube;

        const int cta_size = 729;
               
        ax_cuda_kernel_v12<p,p_cube,p_cube_padded,1,cta_size><<<grid_size,cta_size>>>(d_u, d_w, d_g, d_dxm1, d_dxtm1, *nelt);
#else

        const int grid_size = nelt;
        const int p = 9;
        const int p_sq = 9*9;
        const int p_cube = 9*9*9;
        const int p_cube_padded = p_cube;

        const int cta_size = 243;
        const int pts_per_thread = 3;  // 6*288 = 12*12*12
        const int slab_size = 3;

        const int num_ctas = 4;

        ax_cuda_kernel_v8_shared_D<p,p_sq,p_cube,p_cube_padded,pts_per_thread,slab_size,cta_size,num_ctas><<<grid_size,cta_size>>>(u, w, g, dxm1, dxtm1, nelt);
#endif
      }
      else if ( nx1 == 12 )
      {
#if 0
        // 12x12x12 case
        const int cta_size = 432;
        //const int cta_size = 144;
        const int p = 12;
        const int p_sq = 12*12;
        const int p_cube = 12*12*12;
        const int p_cube_padded = p_cube;

        // We could play with this
        const int pts_per_thread = 4;  // 6*288 = 12*12*12
        const int slab_size = 3;


        //const int pts_per_thread = 12;  // 6*288 = 12*12*12
        //const int slab_size = 1;

        const int grid_size = *nelt;

        ax_cuda_kernel_v3<p,p_sq,p_cube,p_cube_padded,pts_per_thread,slab_size,cta_size><<<grid_size,cta_size>>>(d_u, d_w, d_g, d_dxm1, d_dxtm1, *nelt);

#else

        // 12x12x12 case
        const int p = 12;
        const int p_sq = 12*12;
        const int p_cube = 12*12*12;
        const int p_cube_padded = p_cube;

        // We could play with this
        const int grid_size = nelt;

        /*
        const int cta_size = 1728;
        const int pts_per_thread = 1;  // 6*288 = 12*12*12
        const int slab_size = 12;
        */

        /*
        const int cta_size = 864;
        const int pts_per_thread = 2;  // 6*288 = 12*12*12
        const int slab_size = 6;
        */

        // BEST CONFIG
        const int cta_size = 576;
        const int pts_per_thread = 3;  // 6*288 = 12*12*12
        const int slab_size = 4;
        const int num_ctas = 2;
        /*
        const int cta_size = 432;
        const int pts_per_thread = 4;  // 6*288 = 12*12*12
        const int slab_size = 3;
        */


        /*
        const int cta_size = 288;
        const int pts_per_thread = 6;  // 6*288 = 12*12*12
        const int slab_size = 2;
        */

        //ax_cuda_kernel_v6<p,p_sq,p_cube,p_cube_padded,cta_size><<<grid_size,cta_size>>>(d_u, d_w, d_g, d_dxm1, d_dxtm1, *nelt);
        //ax_cuda_kernel_v8<p,p_sq,p_cube,p_cube_padded,pts_per_thread,slab_size,cta_size,num_ctas><<<grid_size,cta_size>>>(d_u, d_w, d_g, d_dxm1, d_dxtm1, *nelt);
        ax_cuda_kernel_v8_shared_D<p,p_sq,p_cube,p_cube_padded,pts_per_thread,slab_size,cta_size,num_ctas><<<grid_size,cta_size>>>(u, w, g, dxm1, dxtm1, nelt);


        /*
        const int cta_size = 1728;
        ax_cuda_kernel_v9<p,p_cube,p_cube_padded,cta_size><<<grid_size,cta_size>>>(d_u, d_w, d_g, d_dxm1, d_dxtm1, *nelt);
        */

#endif
      }

      cudaCheckError();
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);

      cudaEventElapsedTime(&time, start, stop);
      //printf ("Time for the kernel: %f ms\n", time);

      /* copy w back to host */
      //copyFrom(*nelt, nxyz, w);

      return;
}


// 

void gs_op_cuda(double* w, int dom, int op, int in_transpose)
{
  bool transpose = (in_transpose!=0);

  // Reserve MPI buffer

  local_gather_cuda(w,w,gpu_dom.d_map_offsets[0^transpose],gpu_dom.d_map_indices_from[0^transpose],gpu_dom.d_map_indices_from_COO[0^transpose],gpu_dom.d_map_indices_to[0^transpose],gpu_dom.size_from[0], gpu_dom.size_to[0]);

  //TODO: Do MPI communication here


  // Init

  local_scatter_cuda(w,w,gpu_dom.d_map_offsets[1^transpose],gpu_dom.d_map_indices_from[1^transpose],gpu_dom.d_map_indices_from_COO[1^transpose], gpu_dom.d_map_indices_to[1^transpose],gpu_dom.size_from[1], gpu_dom.size_to[0]);
}

void add2s2_cuda(double *a, double *b, double c1, int n)
{
  cublasDaxpy(cublas_handle, n, &c1, b, 1, a, 1);
  cudaCheckError();
}


void add2s1_cuda(double *a, double *b, double c1, int n)
{

 // TODO: should probably merge into single kernel

  double one = 1.;
  cublasDscal(cublas_handle, n, &c1, a, 1); 
  cudaCheckError();
  cublasDaxpy(cublas_handle, n, &one, b, 1, a, 1);
  cudaCheckError();
}


void mask_cuda(double *w, int nid)
{
  if (nid == 0)
  {
    mask_kernel<<<1,1>>>(w,1);
  }
}

void rzero_cuda(double *a, int n)
{
  double zero = 0.;
  cudaCheckError();
  cublasDscal(cublas_handle, n, &zero, a, 1);
  cudaCheckError();
}

void copy_cuda(double *a, double* b, int n)
{
  cudaCheckError();
  cublasDcopy(cublas_handle, n, b, 1, a, 1); 
  cudaCheckError();
}


__global__ 
void glsc3_cuda_kernel(double* a, double* b, double* mult, double* result, int n)
{
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  while (tid < n)
  {
    result[tid] = a[tid]*b[tid]*mult[tid];
    tid += gridDim.x*blockDim.x;
  }
}


template <int block_size>
__global__ 
void CalcSumOneBlock(double* temp, double* reduced_value, int shared_array_size)
{
  volatile __shared__ double s_data[block_size];
  int tid = threadIdx.x;

  if (blockIdx.x==0)
  {
    if (tid < shared_array_size)
      s_data[tid] = temp[tid];
    else
      s_data[tid] = 0.;

    __syncthreads();

    if (block_size >= 1024) { if (tid < 512) { s_data[tid] = s_data[tid]+s_data[tid + 512]; } __syncthreads(); }
    if (block_size >=  512) { if (tid < 256) { s_data[tid] = s_data[tid]+s_data[tid + 256]; } __syncthreads(); }
    if (block_size >=  256) { if (tid < 128) { s_data[tid] = s_data[tid]+s_data[tid + 128]; } __syncthreads(); }
    if (block_size >=  128) { if (tid <  64) { s_data[tid] = s_data[tid]+s_data[tid +  64]; } __syncthreads(); }
    if (tid <  32) { s_data[tid] = s_data[tid]+s_data[tid +  32]; } 
    if (tid <  16) { s_data[tid] = s_data[tid]+s_data[tid +  16]; } 
    if (tid <   8) { s_data[tid] = s_data[tid]+s_data[tid +   8]; } 
    if (tid <   4) { s_data[tid] = s_data[tid]+s_data[tid +   4]; } 
    if (tid <   2) { s_data[tid] = s_data[tid]+s_data[tid +   2]; } 
    if (tid <   1) { s_data[tid] = s_data[tid]+s_data[tid +   1]; } 

    if (tid<1)
    {
      // Zero copy to host
      *(reduced_value)= s_data[0];
    }
  }
}


template < int cta_size >
__global__ 
void glsc3_cuda_kernel_v2(double* a, double* b, double* mult, double *result, int n)
{
  __shared__ volatile double s_data[cta_size];
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  double temp = 0.;
  while (tid < n)
  {
    temp += a[tid]*b[tid]*mult[tid];
    tid += gridDim.x*blockDim.x;
  }

  s_data[tid] = temp;

  __syncthreads();

  // Do shared memory reduction, TODO: use CUB
  if (cta_size >= 1024) { 
    if (tid < 512) s_data[tid]   = s_data[tid] + s_data[tid + 512] ; 
    __syncthreads();  
    }

  if (cta_size >=  512) { 
    if (tid < 256) s_data[tid] = s_data[tid] + s_data[tid + 256] ;
    __syncthreads(); 
  }

  if (cta_size >=  256) { 
    if (tid < 128) s_data[tid] = s_data[tid] + s_data[tid + 128] ; 
    __syncthreads(); 
  }

  if (cta_size >=  128) { 
    if (tid <  64) s_data[tid] = s_data[tid] + s_data[tid +  64] ; 
    __syncthreads(); 
  }

  if (tid <  32) s_data[tid] = s_data[tid] + s_data[tid +  32] ; 
  if (tid <  16) s_data[tid] = s_data[tid] + s_data[tid +  16] ;  
  if (tid <   8) s_data[tid] = s_data[tid] + s_data[tid +   8] ;  
  if (tid <   4) s_data[tid] = s_data[tid] + s_data[tid +   4] ;  
  if (tid <   2) s_data[tid] = s_data[tid] + s_data[tid +   2] ;  
  if (tid <   1) s_data[tid] = s_data[tid] + s_data[tid +   1] ;  

  if (tid==0) 
    result[blockIdx.x] = s_data[tid];

}

double glsc3_cuda(double *a, double* b, double* mult,  int n)
{
  cudaCheckError();

  double glsc3;
  // TODO: This could be made faster by having a single kernel followed by reduction across block, instead of writing to global memory
  int cta_size = 128;
  int grid_size=min(4096,( (n+cta_size-1)/cta_size));

  glsc3_cuda_kernel<<<grid_size,cta_size>>>(a,b,mult,gpu_dom.d_temp,n);

  thrust::device_ptr<double> beg(gpu_dom.d_temp);
  glsc3 = thrust::reduce(beg,beg+n);
  return glsc3;

  cudaCheckError();

}

void solveM_cuda(double *z, double* r, int n)
{
  copy_cuda(z,r,n);
}

void axcuda(double* w, double* u, double *g, double *dxm1, double* dxtm1, int nx1, int ny1, int nz1, int nelt, int ldim, int nid)
{

  axcuda_e(w,u,g,dxm1,dxtm1,nx1,ny1,nz1,nelt,ldim);

  // TODO: Currently, parameters dom and op are ignored
  gs_op_cuda(w,1,1,0); 

  int n = nx1*ny1*nz1*nelt;
  add2s2_cuda(w,u,.1,n);
  mask_cuda(w,nid);
}


extern "C"
{
  void gs_setup_cuda(const uint* map_local_0, const uint* map_local_1, const uint* flagged_primaries)
  {

    // Initialize data required for gather-scatter operation
    fill_gpu_maps(&(gpu_dom.size_from[0]), &(gpu_dom.size_to[0]), &(gpu_dom.d_map_offsets[0]), &(gpu_dom.d_map_indices_from[0]), &(gpu_dom.d_map_indices_from_COO[0]), &(gpu_dom.d_map_indices_to[0]), map_local_0);
    fill_gpu_maps(&(gpu_dom.size_from[1]), &(gpu_dom.size_to[1]), &(gpu_dom.d_map_offsets[1]), &(gpu_dom.d_map_indices_from[1]), &(gpu_dom.d_map_indices_from_COO[1]), &(gpu_dom.d_map_indices_to[1]), map_local_1);

    fill_flagged_primaries_map(gpu_dom.d_flagged_primaries,flagged_primaries);

  }


  void cg_cuda_init_(double* x, double* f, double* g, double* c, double* r, double* w, double* p, double* z, int* nx1, int* ny1, int* nz1, int* nelt, int* ldim, double* dxm1, double* dxtm1, int* niter, double* flop_cg, const int *gsh_handle, int* nid)
  {
    // Initialize gpu_dom structure
    // Note: the gsh structure on the GPU is already initialized, since gs_cuda_setup is called in the proxy_setupds function in driver.f

    gpu_dom.nid = (*nid);
    gpu_dom.nx1 = (*nx1);
    gpu_dom.ny1 = (*ny1);
    gpu_dom.nz1 = (*nz1);
    gpu_dom.ldim= (*ldim);
    gpu_dom.nelt = (*nelt);
    gpu_dom.niter = (*niter);

    int nxyz = (*nx1)*(*ny1)*(*nz1);
    
    // Initializing the Cublas library 
    if (cublas_handle==NULL)
      cublasCreate(&cublas_handle);

    cudaEventCreateWithFlags(&(gpu_dom.reduced_value_event),cudaEventDisableTiming);
    cudaCheckError();

    // malloc GPU memory for u, w, ur, us, ut, g, dxm1, dxtm1 
    //cudaMalloc((void **)&gpu_dom.d_u, gpu_dom.nelt*nxyz*sizeof(double));
    cudaMalloc((void **)&gpu_dom.d_w, gpu_dom.nelt*nxyz*sizeof(double));
    cudaMalloc((void **)&gpu_dom.d_f, gpu_dom.nelt*nxyz*sizeof(double));
    cudaMalloc((void **)&gpu_dom.d_c, gpu_dom.nelt*nxyz*sizeof(double));
    cudaMalloc((void **)&gpu_dom.d_r, gpu_dom.nelt*nxyz*sizeof(double));
    cudaMalloc((void **)&gpu_dom.d_p, gpu_dom.nelt*nxyz*sizeof(double));
    cudaMalloc((void **)&gpu_dom.d_z, gpu_dom.nelt*nxyz*sizeof(double));
    cudaMalloc((void **)&gpu_dom.d_x, gpu_dom.nelt*nxyz*sizeof(double));
    cudaMalloc((void **)&gpu_dom.d_temp, gpu_dom.nelt*nxyz*sizeof(double));
    cudaMalloc((void **)&gpu_dom.d_g, gpu_dom.nelt*nxyz*2*gpu_dom.ldim*sizeof(double));
    cudaMalloc((void **)&gpu_dom.d_dxm1, gpu_dom.nx1*gpu_dom.nx1*sizeof(double));
    cudaMalloc((void **)&gpu_dom.d_dxtm1, gpu_dom.nx1*gpu_dom.nx1*sizeof(double));

    cudaMallocHost(&gpu_dom.reduced_value,sizeof(double));

    cudaCheckError();

    // copy data to the GPU

    cudaMemcpy(gpu_dom.d_w, w, gpu_dom.nelt*nxyz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dom.d_f, f, gpu_dom.nelt*nxyz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dom.d_c, c, gpu_dom.nelt*nxyz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dom.d_r, r, gpu_dom.nelt*nxyz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dom.d_p, p, gpu_dom.nelt*nxyz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dom.d_z, z, gpu_dom.nelt*nxyz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dom.d_x, x, gpu_dom.nelt*nxyz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dom.d_p, p, gpu_dom.nelt*nxyz*sizeof(double), cudaMemcpyHostToDevice);


    // Switch to AoS for d_g
#ifdef AOS
    double* g_tmp = (double*) malloc(gpu_dom.nelt*nxyz*2*gpu_dom.ldim*sizeof(double));
    int lt = nxyz*gpu_dom.nelt;
    for (int i=0;i<lt;i++)
      for (int j=0;j<6;j++)
        g_tmp[j*lt+i] = g[i*6+j];
    cudaMemcpy(gpu_dom.d_g, g_tmp, gpu_dom.nelt*nxyz*2*gpu_dom.ldim*sizeof(double), cudaMemcpyHostToDevice);
#else
    cudaMemcpy(gpu_dom.d_g, g, gpu_dom.nelt*nxyz*2*gpu_dom.ldim*sizeof(double), cudaMemcpyHostToDevice);
#endif

    cudaMemcpy(gpu_dom.d_dxm1, dxm1, gpu_dom.nx1*gpu_dom.nx1*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dom.d_dxtm1, dxtm1, gpu_dom.nx1*gpu_dom.nx1*sizeof(double), cudaMemcpyHostToDevice);

    cudaCheckError();

    return;
  }


  void cg_cuda_(double* r, int* copy_to_cpu)
  {
    cudaEvent_t timer_start, timer_stop;
    cudaEventCreate(&timer_start);
    cudaEventCreate(&timer_stop);
    cudaEventRecord( timer_start );

    int n = gpu_dom.nx1 * gpu_dom.ny1 * gpu_dom.nz1 * gpu_dom.nelt;
    double pap = 0.0;

    // set machine tolerances
    double one = 1.;
    double eps = 1.e-20;

    if (one+eps == one) 
      eps = 1.e-14;
    if (one+eps == one) 
      eps = 1.e-7;

    double rtz1 = 1.0;

    cudaCheckError();
    rzero_cuda(gpu_dom.d_x, n);
    copy_cuda(gpu_dom.d_r, gpu_dom.d_f, n);

    // Zero out Dirichlet conditions
    mask_cuda(gpu_dom.d_r, gpu_dom.nid);

    double rnorm = sqrt(glsc3_cuda(gpu_dom.d_r, gpu_dom.d_c, gpu_dom.d_r, n));
      
    int iter = 0;
    printf("cg: %d %g\n",iter,rnorm);

    int miter = gpu_dom.niter;
    double alpha, beta;

    for (iter=1; iter <= miter; iter++)
    {
       solveM_cuda(gpu_dom.d_z, gpu_dom.d_r, n);

       double rtz2=rtz1;
       rtz1 = glsc3_cuda(gpu_dom.d_r, gpu_dom.d_c, gpu_dom.d_z, n);

       beta = rtz1/rtz2;
       if (iter==1) 
         beta=0.0;

#ifdef DEBUG
       printf("rtz1 = %12.8g\n",rtz1);
#endif
       add2s1_cuda(gpu_dom.d_p, gpu_dom.d_z, beta, n);

       axcuda(gpu_dom.d_w, gpu_dom.d_p, gpu_dom.d_g, gpu_dom.d_dxm1, gpu_dom.d_dxtm1, gpu_dom.nx1, gpu_dom.ny1, gpu_dom.nz1, gpu_dom.nelt, gpu_dom.ldim, gpu_dom.nid);

       pap = glsc3_cuda(gpu_dom.d_w, gpu_dom.d_c, gpu_dom.d_p, n);
#ifdef DEBUG
       printf("pap = %12.8g\n",pap);
#endif
       alpha=rtz1/pap;
       double alphm = -alpha;

       add2s2_cuda(gpu_dom.d_x, gpu_dom.d_p, alpha, n);
       add2s2_cuda(gpu_dom.d_r, gpu_dom.d_w, alphm, n);

       double rtr = glsc3_cuda(gpu_dom.d_r, gpu_dom.d_c, gpu_dom.d_r, n);

       rnorm = sqrt(rtr);
    }

    if (gpu_dom.nid==0) 
      printf("cg: %d %12.8g %12.8g %12.8g %12.8g\n",iter,rnorm,alpha,beta,pap);

    if ((*copy_to_cpu)==1)
      cudaMemcpy(r, gpu_dom.d_r, n*sizeof(double), cudaMemcpyDeviceToHost);

    float elapsed_time;
    cudaEventRecord( timer_stop );
    cudaEventSynchronize( timer_stop);
    cudaEventElapsedTime( &elapsed_time, timer_start, timer_stop );
    elapsed_time*=1.e-3f;
    printf("GPU Elapsed Time        =  %8.4e seconds\n",elapsed_time);
  }

  void cg_cuda_free_() 
  {
    /* free GPU memory for u, w, ur, us, ut, g, dxm1, dxtm1  */

    cudaCheckError();

    cudaFree(gpu_dom.d_map_offsets[0]);
    cudaFree(gpu_dom.d_map_indices_from[0]);
    cudaFree(gpu_dom.d_map_indices_from_COO[0]);
    cudaFree(gpu_dom.d_map_indices_to[0]);

    cudaFree(gpu_dom.d_map_offsets[1]);
    cudaFree(gpu_dom.d_map_indices_from[1]);
    cudaFree(gpu_dom.d_map_indices_from_COO[1]);
    cudaFree(gpu_dom.d_map_indices_to[1]);

    cudaFree(gpu_dom.d_w);
    cudaFree(gpu_dom.d_p);
    cudaFree(gpu_dom.d_g);
    cudaFree(gpu_dom.d_dxm1);
    cudaFree(gpu_dom.d_dxtm1);

    cudaCheckError();

    return;
  }

} // extern C





