
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
   * Matrix multiply
  */
  __global__ void gmxm(double *a, int n1, double *b, int n2, double *c, int n3) {

    int id= blockDim.x*blockIdx.x+ threadIdx.x;
    int row = id % n1;
    int col = id / n1;

    if (id < n1*n3) {
      c[id] = 0.0;

      int k;
      for (k = 0; k < n2; k++) {
        c[id] += a[k*n1+row]*b[col*n2+k];
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
  __global__ void geom(int nxyz, double *ur, double *us, double *ut, double *g) {
    int i= blockDim.x*blockIdx.x+ threadIdx.x;

    if (i < nxyz) {
      double wr = g[i*6+0]*ur[i] + g[i*6+1]*us[i] + g[i*6+2]*ut[i];
      double ws = g[i*6+1]*ur[i] + g[i*6+3]*us[i] + g[i*6+4]*ut[i];
      double wt = g[i*6+2]*ur[i] + g[i*6+4]*us[i] + g[i*6+5]*ut[i];
      ur[i] = wr;
      us[i] = ws;
      ut[i] = wt;
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
      int m3 = m1*m1*m1;

      int thrdsPBlck = 128;

      /* copy u to the gpu */

      copyTo(*nelt, nxyz, u);

      /* loop over each element */

      int e, k;
      for (e = 0; e < *nelt; e++) {

        /* Calculate ur, us, and ut */

        gmxm<<<16, thrdsPBlck>>>(d_dxm1, m1, d_u+e*nxyz, m1, d_ur+e*nxyz, m2);
        for (k = 0; k < *nz1; k++) {
          gmxm<<<16,thrdsPBlck>>>(d_u+e*nxyz+(*nx1)*(*ny1)*k, m1, d_dxtm1, m1, d_us+e*nxyz+(*nx1)*(*ny1)*k, m1);
        }
        gmxm<<<16,thrdsPBlck>>>(d_u+e*nxyz, m2, d_dxtm1, m1, d_ut+e*nxyz, m1);

        /* calculate geom effects */

        geom<<<16, thrdsPBlck>>>(nxyz, d_ur+e*nxyz, d_us+e*nxyz, d_ut+e*nxyz, d_g+e*nxyz*6);


        /* calculate u from ur, us, ut */

        gmxm<<<16,thrdsPBlck>>>(d_dxtm1, m1, d_ur+e*nxyz, m1, d_w+e*nxyz, m2);

        for (k = 0; k < *nz1; k++) {
          gmxm<<<16,thrdsPBlck>>>(d_us+e*nxyz+(*nx1)*(*ny1)*k, m1, d_dxm1, m1, d_u+e*nxyz+(*nx1)*(*ny1)*k, m1);
        }
        gadd2<<<16,thrdsPBlck>>>(d_w+e*nxyz, d_u+e*nxyz, m3);

        gmxm<<<16,thrdsPBlck>>>(d_ut+e*nxyz, m2, d_dxm1, m1, d_u+e*nxyz, m1);
        gadd2<<<16,thrdsPBlck>>>(d_w+e*nxyz, d_u+e*nxyz, m3);
      }

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
