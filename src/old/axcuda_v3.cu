
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
      double (*gm)[*nelt][nxyz][2*(*ldim)] = (double (*)[*nelt][nxyz][2*(*ldim)]) g;

      double ur[(*nelt)*nxyz],us[(*nelt)*nxyz],ut[(*nelt)*nxyz];

      int e, k, i;
      for (e = 0; e < *nelt; e++) {
        cmxm(dxm1, &m1, &((*um)[e][0]), &m1, &(ur[e*nxyz]), &m2);
        for (k = 0; k < *nz1; k++) {
          cmxm(&((*um)[e][(*nx1)*(*ny1)*k]), &m1, dxtm1, &m1, &(us[e*nxyz+ (*nx1)*(*ny1)*k]), &m1);
        }
        cmxm(&((*um)[e][0]), &m2, dxtm1, &m1, &(ut[e*nxyz]), &m1);
      }

      for (e = 0; e < *nelt; e++) {
        int off = e*nxyz;
        for (i = 0; i < nxyz; i++) {
          double wr = (*gm)[e][i][0]*ur[off+i] + (*gm)[e][i][1]*us[off+i] + (*gm)[e][i][2]*ut[off+i];
          double ws = (*gm)[e][i][1]*ur[off+i] + (*gm)[e][i][3]*us[off+i] + (*gm)[e][i][4]*ut[off+i];
          double wt = (*gm)[e][i][2]*ur[off+i] + (*gm)[e][i][4]*us[off+i] + (*gm)[e][i][5]*ut[off+i];
          ur[off+i] = wr;
          us[off+i] = ws;
          ut[off+i] = wt;
        }
      }

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
