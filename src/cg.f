c-----------------------------------------------------------------------
      subroutine cg(x,f,g,c,r,w,p,z,n,niter,flop_cg)
      include 'SIZE'

c     Solve Ax=f where A is SPD and is invoked by ax()
c
c     Output:  x - vector of length n
c
c     Input:   f - vector of length n
c     Input:   g - geometric factors for SEM operator
c     Input:   c - inverse of the counting matrix
c
c     Work arrays:   r,w,p,z  - vectors of length n
c
c     User-provided ax(w,z,n) returns  w := Az,  
c
c     User-provided solveM(z,r,n) ) returns  z := M^-1 r,  
c
      parameter (lt=lx1*ly1*lz1*lelt)
      real ur(lt),us(lt),ut(lt),wk(lt)

c     parameter (lxyz=lx1*ly1*lz1)
c     real ur(lxyz),us(lxyz),ut(lxyz),wk(lxyz)

      real x(n),f(n),r(n),w(n),p(n),z(n),g(1),c(n)

      character*1 ans

      pap = 0.0

c     set machine tolerances
      one = 1.
      eps = 1.e-20
      if (one+eps .eq. one) eps = 1.e-14
      if (one+eps .eq. one) eps = 1.e-7

      rtz1=1.0

      call rzero(x,n)
      call copy (r,f,n)
      call mask (r)   ! Zero out Dirichlet conditions

      rnorm = sqrt(glsc3(r,c,r,n))
      iter = 0
      if (nid.eq.0)write(6,6) iter,rnorm

      miter = niter
      do iter=1,miter
         call solveM(z,r,n)    ! preconditioner here

         rtz2=rtz1                                                       ! OPS
         rtz1=glsc3(r,c,z,n)   ! parallel weighted inner product r^T C z ! 3n

         beta = rtz1/rtz2
         if (iter.eq.1) beta=0.0
         call add2s1(p,z,beta,n)                                         ! 2n

         call ax(w,p,g,ur,us,ut,wk,n)                                    ! flopa
         pap=glsc3(w,c,p,n)                                              ! 3n

         alpha=rtz1/pap
         alphm=-alpha
         call add2s2(x,p,alpha,n)                                        ! 2n
         call add2s2(r,w,alphm,n)                                        ! 2n

         rtr = glsc3(r,c,r,n)                                            ! 3n
         if (iter.eq.1) rlim2 = rtr*eps**2
         if (iter.eq.1) rtr0  = rtr
         rnorm = sqrt(rtr)
c        if (nid.eq.0.and.mod(iter,100).eq.0) 
c    $      write(6,6) iter,rnorm,alpha,beta,pap
    6    format('cg:',i4,1p4e12.4)
c        if (rtr.le.rlim2) goto 1001

      enddo

 1001 continue

      if (nid.eq.0) write(6,6) iter,rnorm,alpha,beta,pap

      flop_cg = flop_cg + iter*15*n

      return
      end
c-----------------------------------------------------------------------
      subroutine solveM(z,r,n)
      real z(n),r(n)

      call copy(z,r,n)

      return
      end
c-----------------------------------------------------------------------
      subroutine ax(w,u,gxyz,ur,us,ut,wk,n) ! Matrix-vector product: w=A*u

      include 'SIZE'
      include 'TOTAL'

      parameter (lxyz=lx1*ly1*lz1)
      real w(lxyz,lelt),u(lxyz,lelt),gxyz(2*ldim,lxyz,lelt)
      parameter (lt=lx1*ly1*lz1*lelt)
      real ur(lt),us(lt),ut(lt),wk(lt)

c      real ur(lxyz),us(lxyz),ut(lxyz),wk(lxyz)

      integer e
#ifdef USE_CUDA
      call ax_e_cuda(w, u,dxm1,dxtm1,gxyz,nx1-1,nelt)
#else
      do e=1,nelt                                ! ~
         call ax_e( w(1,e),u(1,e),gxyz(1,1,e)    ! w   = A  u
     $                             ,ur,us,ut,wk) !  L     L  L
      enddo                                      ! 
#endif
      call gs_op(gsh,w,1,1,0)  ! Gather-scatter operation  ! w   = QQ  w
                                                           !            L
      call add2s2(w,u,.1,n)
      call mask(w)             ! Zero out Dirichlet conditions

      nxyz=nx1*ny1*nz1
      flop_a = flop_a + (15*nxyz+12*nx1*nxyz)*nelt

      return
      end
c-------------------------------------------------------------------------
      subroutine ax1(w,u,n)
      include 'SIZE'
      real w(n),u(n)
      real h2i
  
      h2i = (n+1)*(n+1)  
      do i = 2,n-1
         w(i)=h2i*(2*u(i)-u(i-1)-u(i+1))
      enddo
      w(1)  = h2i*(2*u(1)-u(2  ))
      w(n)  = h2i*(2*u(n)-u(n-1))

      return
      end
c-------------------------------------------------------------------------
      subroutine ax_e(w,u,g,ur,us,ut,wk) ! Local matrix-vector product
      include 'SIZE'
      include 'TOTAL'

      parameter (lxyz=lx1*ly1*lz1)
      real w(lxyz),u(lxyz),g(2*ldim,lxyz)

      real ur(lxyz),us(lxyz),ut(lxyz),wk(lxyz)

      nxyz = nx1*ny1*nz1
      n    = nx1-1
#ifdef USE_CUDA
      call local_grad3_comb_cuda(w, u,dxm1,dxtm1,g,n)
#else
      call local_grad3(ur,us,ut,u,n,dxm1,dxtm1)
      do i=1,nxyz
         wr = g(1,i)*ur(i) + g(2,i)*us(i) + g(3,i)*ut(i)
         ws = g(2,i)*ur(i) + g(4,i)*us(i) + g(5,i)*ut(i)
         wt = g(3,i)*ur(i) + g(5,i)*us(i) + g(6,i)*ut(i)
         ur(i) = wr
         us(i) = ws
         ut(i) = wt
      enddo
      call local_grad3_t(w,ur,us,ut,n,dxm1,dxtm1,wk)
#endif
      return
      end
c-------------------------------------------------------------------------
      subroutine local_grad3(ur,us,ut,u,n,D,Dt)
c     Output: ur,us,ut         Input:u,n,D,Dt
      real ur(0:n,0:n,0:n),us(0:n,0:n,0:n),ut(0:n,0:n,0:n)
      real u (0:n,0:n,0:n)
      real D (0:n,0:n),Dt(0:n,0:n)

      m1 = n+1
      m2 = m1*m1

      call mxm(D ,m1,u,m1,ur,m2)
      do k=0,n
         call mxm(u(0,0,k),m1,Dt,m1,us(0,0,k),m1)
      enddo
      call mxm(u,m2,Dt,m1,ut,m1)

      return
      end
c-----------------------------------------------------------------------
      subroutine local_grad3_t(u,ur,us,ut,N,D,Dt,w)
c     Output: ur,us,ut         Input:u,N,D,Dt
      real u (0:N,0:N,0:N)
      real ur(0:N,0:N,0:N),us(0:N,0:N,0:N),ut(0:N,0:N,0:N)
      real D (0:N,0:N),Dt(0:N,0:N)
      real w (0:N,0:N,0:N)
      integer e

      m1 = N+1
      m2 = m1*m1
      m3 = m1*m1*m1

      call mxm(Dt,m1,ur,m1,u,m2)

      do k=0,N
         call mxm(us(1,0,k),m1,D ,m1,w(0,0,k),m1)
      enddo
      call add2(u,w,m3)

      call mxm(ut,m2,D ,m1,w,m1)
      call add2(u,w,m3)

      return
      end
c-----------------------------------------------------------------------
      subroutine mask(w)   ! Zero out Dirichlet conditions
      include 'SIZE'
      real w(1)

      if (nid.eq.0) w(1) = 0.  ! suitable for solvability

      return
      end
c-----------------------------------------------------------------------
