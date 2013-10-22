#include <stdio.h>
  void cmxm(double *a, int n1, double *b, int n2, double *c, int n3) {
    double (*am)[n2][n1] = (double (*)[n2][n1]) a;
    double (*bm)[n3][n2] = (double (*)[n3][n2]) b;
    double (*cm)[n3][n1] = (double (*)[n3][n1]) c;

    int i,j,k;
    for (j = 0; j < n3; j++) {
      for (i = 0; i < n1; i++) {
        (*cm)[j][i] = 0.0;

        for (k = 0; k < n2; k++) {
          (*cm)[j][i] += (*am)[k][i]*(*bm)[j][k];
        }
      }
    }

    return;
  }


  void cadd2(double *a, double *b, int n) {

    int i;
    for (i=0; i < n; i++) {
      a[i] += b[i];
    }

    return;
  }


  void axcuda_(int *nx1,int *ny1,int *nz1,int *nelt,int *ldim,
              double w[*nelt][(*nx1)*(*ny1)*(*nz1)],
              double u[*nelt][(*nx1)*(*ny1)*(*nz1)],
              double g[*nelt][(*nx1)*(*ny1)*(*nz1)][2*(*ldim)],
              double dxm1[*nx1][*nx1], double dxtm1[*nx1][*nx1]) {

    int nxyz = (*nx1)*(*ny1)*(*nz1);
    int m1 = (*nx1);
    int m2 = m1*m1;
    int m3 = m1*m1*m1;

    double ur[nxyz],us[nxyz],ut[nxyz];

    int e, k, i;
    for (e = 0; e < *nelt; e++) {
      cmxm((double *) dxm1, m1, &(u[e][0]), m1, (double *) ur, m2);
      for (k = 0; k < *nz1; k++) {
        cmxm((double*) &(u[e][(*nx1)*(*ny1)*k]), m1, (double*) dxtm1, m1, &us[(*nx1)*(*ny1)*k], m1);
      }
      cmxm(&(u[e][0]), m2, (double*)dxtm1, m1, (double*)ut, m1);

      for (i = 0; i < nxyz; i++) {
        double wr = g[e][i][0]*ur[i] + g[e][i][1]*us[i] + g[e][i][2]*ut[i];
        double ws = g[e][i][1]*ur[i] + g[e][i][3]*us[i] + g[e][i][4]*ut[i];
        double wt = g[e][i][2]*ur[i] + g[e][i][4]*us[i] + g[e][i][5]*ut[i];
        ur[i] = wr;
        us[i] = ws;
        ut[i] = wt;
      }

      cmxm((double*)dxtm1, m1, (double*)ur, m1, &(w[e][0]), m2);

      double wa[*nz1][*ny1][*nx1];
      for (k = 0; k < *nz1; k++) {
        cmxm(&us[(*nx1)*(*ny1)*k], m1, (double*)dxm1, m1, &(wa[k][0][0]), m1);
      }

      cadd2(&(w[e][0]), (double*)wa, m3);
      cmxm((double*)ut, m2, (double*)dxm1, m1, (double*)wa, m1);
      cadd2(&(w[e][0]), (double*)wa, m3);

    }

    return;
  }
