
#include <stdio.h>
#include "cuda_runtime.h"

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


  /*
   * Matrix-vector kernel
  */
  extern "C" 
  {
    void axcuda_(int *nx1,int *ny1,int *nz1,int *nelt,int *ldim, double *w, double *u,
                 double *g, double *dxm1, double *dxtm1) {

      int nxyz = (*nx1)*(*ny1)*(*nz1);
      int m1 = (*nx1);
      int m2 = m1*m1;

      int thrdsPBlck = 128;
      int numBlocks = (nxyz*(*nelt)+thrdsPBlck-1)/thrdsPBlck;

      /* copy u to the gpu */

      copyTo(*nelt, nxyz, u);

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

      /* copy w back to host */

      copyFrom(*nelt, nxyz, w);

      return;
    }
  }


  /*
   * Initialize GPU operations
  */
  extern "C" 
  {
    void axcuda_init_(int *nx1, int *ny1, int *nz1, int *nelt, int *ldim, double *g, double *dxm1, double *dxtm1) {

      int nxyz = (*nx1)*(*ny1)*(*nz1);

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
    void axcuda_free_() {

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
