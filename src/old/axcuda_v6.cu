
#include <stdio.h>

  /* GPU data pointers */
  static double *d_ur, *d_us, *d_ut, *d_g;


  __global__ void geom(int nelt, int nxyz, double *ur, double *us, double *ut, double *g) {
    int i= blockDim.x*blockIdx.x+ threadIdx.x;

    if (i < nelt*nxyz) {
      double wr = g[i*6+0]*ur[i] + g[i*6+1]*us[i] + g[i*6+2]*ut[i];
      double ws = g[i*6+1]*ur[i] + g[i*6+3]*us[i] + g[i*6+4]*ut[i];
      double wt = g[i*6+2]*ur[i] + g[i*6+4]*us[i] + g[i*6+5]*ut[i];
      ur[i] = wr;
      us[i] = ws;
      ut[i] = wt;
    }

    return;
  }

  void cmxm(double *a, int *n1, double *b, int *n2, double *c, int *n3) {
    double (*am)[*n2][*n1] = (double (*)[*n2][*n1]) a;
    double (*bm)[*n3][*n2] = (double (*)[*n3][*n2]) b;
    double (*cm)[*n3][*n1] = (double (*)[*n3][*n1]) c;

    int i,j,k;
    for (j = 0; j < *n3; j++) {
      for (i = 0; i < *n1; i++) {
        (*cm)[j][i] = 0.0;

        for (k = 0; k < *n2; k++) {
          (*cm)[j][i] += (*am)[k][i]*(*bm)[j][k];
        }
      }
    }

    return;
  }


  void cadd2(double *a, double *b, int *n) {

    int i;
    for (i=0; i < *n; i++) {
      a[i] += b[i];
    }

    return;
  }


  void copyTo(int nelt, int nxyz, double *ur, double *us, double *ut) {

      cudaError_t err = cudaMemcpy(d_ur, ur, nelt*nxyz*sizeof(double), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        printf("cudaMemcpy error\n");
        exit(1);
      }

      err = cudaMemcpy(d_us, us, nelt*nxyz*sizeof(double), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        printf("cudaMemcpy error\n");
        exit(1);
      }

      err = cudaMemcpy(d_ut, ut, nelt*nxyz*sizeof(double), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        printf("cudaMemcpy error\n");
        exit(1);
      }
  }


  void copyFrom(int nelt, int nxyz, double *ur, double *us, double *ut) {

    cudaError_t err = cudaMemcpy(ur, d_ur, nelt*nxyz*sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      printf("cudaMemcpy error\n");
      exit(1);
    }

    err = cudaMemcpy(us, d_us, nelt*nxyz*sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      printf("cudaMemcpy error\n");
      exit(1);
    }

    err = cudaMemcpy(ut, d_ut, nelt*nxyz*sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      printf("cudaMemcpy error\n");
      exit(1);
    }

    return;
  }


  extern "C" 
  {
    void axcuda_(int *nx1,int *ny1,int *nz1,int *nelt,int *ldim, double *w, double *u,
                 double *g, double *dxm1, double *dxtm1) {

      int nxyz = (*nx1)*(*ny1)*(*nz1);
      int m1 = (*nx1);
      int m2 = m1*m1;
      int m3 = m1*m1*m1;

      double (*wm)[*nelt][nxyz] = (double (*)[*nelt][nxyz]) w;
      double (*um)[*nelt][nxyz] = (double (*)[*nelt][nxyz]) u;

      double ur[(*nelt)*nxyz],us[(*nelt)*nxyz],ut[(*nelt)*nxyz];

      /* Calculate ur, us, and ut */

      int e, k;
      for (e = 0; e < *nelt; e++) {
        cmxm(dxm1, &m1, &((*um)[e][0]), &m1, &(ur[e*nxyz]), &m2);
        for (k = 0; k < *nz1; k++) {
          cmxm(&((*um)[e][(*nx1)*(*ny1)*k]), &m1, dxtm1, &m1, &(us[e*nxyz+ (*nx1)*(*ny1)*k]), &m1);
        }
        cmxm(&((*um)[e][0]), &m2, dxtm1, &m1, &(ut[e*nxyz]), &m1);
      }

      /* copy ur, us, ut to the gpu */

      copyTo(*nelt, nxyz, ur, us, ut);

      /* calculate geom effects */

      int threadsPerBlock = 128;
      int numBlocks = ((*nelt)*nxyz + threadsPerBlock-1)/threadsPerBlock;
      geom<<<numBlocks, threadsPerBlock>>>(*nelt, nxyz, d_ur, d_us, d_ut, d_g);

      /* copy ur, us, ut back to host */

      copyFrom(*nelt, nxyz, ur, us, ut);

      /* calculate u from ur, us, ut */

      for (e = 0; e < *nelt; e++) {
        cmxm(dxtm1, &m1, &(ur[e*nxyz]), &m1, &((*wm)[e][0]), &m2);

        double wa[*nz1][*ny1][*nx1];
        for (k = 0; k < *nz1; k++) {
          cmxm(&(us[e*nxyz+(*nx1)*(*ny1)*k]), &m1, dxm1, &m1, &wa[k][0][0], &m1);
        }
        cadd2(&((*wm)[e][0]), (double*)wa, &m3);

        cmxm(&(ut[e*nxyz]), &m2, dxm1, &m1, (double*)wa, &m1);
        cadd2(&((*wm)[e][0]), (double*)wa, &m3);
      }

      return;
    }
  }


  extern "C" 
  {
    void axcuda_init_(int *nx1, int *ny1, int *nz1, int *nelt, int *ldim, double *g) {

      int nxyz = (*nx1)*(*ny1)*(*nz1);

      /* malloc GPU memory for ur, us, ut, and g  */

      cudaError_t err = cudaMalloc((void **)&d_ur, (*nelt)*nxyz*sizeof(double));
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

      /* copy g to the gpu */

      err = cudaMemcpy(d_g, g, (*nelt)*nxyz*2*(*ldim)*sizeof(double), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        printf("cudaMemcpy error\n");
        exit(1);
      }

      return;
    }
  }


  extern "C" 
  {
    void axcuda_free_() {

      /* free GPU memory for ur, us, ut, and g  */

      cudaError_t err = cudaFree(d_ur);
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


      return;
    }
  }
