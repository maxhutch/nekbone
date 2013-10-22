c-------------------------------------------------------------------------
c     subroutine axcuda(w,u,g,dxm1,dxtm1,nelt,ldim,nx1,ny1,nz1) ! Local matrix-vector product
      subroutine axcuda(nx1,ny1,nz1,nelt,ldim,w,u,g,dxm1,dxtm1) ! Local matrix-vector product

      implicit none

      integer nelt, ldim, nx1, ny1, nz1
      real w(nx1*ny1*nz1,nelt),u(nx1*ny1*nz1,nelt)
      real g(2*ldim,nx1*ny1*nz1,nelt)
      real dxm1(nx1,nx1), dxtm1(nx1,nx1)

      integer e, i, n, nxyz, m1, m2, m3, k
      real ur(nx1*ny1*nz1),us(nx1*ny1*nz1),ut(nx1*ny1*nz1)
      real wr, ws, wt
      real wa(nx1,ny1,nz1)
     
      nxyz = nx1*ny1*nz1
      n    = nx1-1
      m1 = n+1
      m2 = m1*m1
      m3 = m1*m1*m1

      do e=1, nelt
        call mxm(dxm1,m1,u(1,e),m1,ur,m2)
        do k=1,nz1
          call mxm(u(nx1*ny1*(k-1)+1,e),m1,dxtm1,m1,
     c             us(nx1*ny1*(k-1)+1),m1)
        enddo
        call mxm(u(1,e),m2,dxtm1,m1,ut,m1)

        do i=1,nxyz
           wr = g(1,i,e)*ur(i) + g(2,i,e)*us(i) + g(3,i,e)*ut(i)
           ws = g(2,i,e)*ur(i) + g(4,i,e)*us(i) + g(5,i,e)*ut(i)
           wt = g(3,i,e)*ur(i) + g(5,i,e)*us(i) + g(6,i,e)*ut(i)
           ur(i) = wr
           us(i) = ws
           ut(i) = wt
        enddo

        call mxm(dxtm1,m1,ur,m1,w(1,e),m2)

        do k=1,nz1
          call mxm(us(nx1*ny1*(k-1)+1),m1,dxm1,m1,wa(1,1,k),m1)
        enddo
        call add2(w(1,e),wa,m3)

        call mxm(ut,m2,dxm1,m1,wa,m1)
        call add2(w(1,e),wa,m3)
      end do

      return
      end
