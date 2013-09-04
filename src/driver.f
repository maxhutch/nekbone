c-----------------------------------------------------------------------
      program nekbone
      
      include 'SIZE'
      include 'TOTAL'
      include 'mpif.h'

      parameter (lxyz = lx1*ly1*lz1)
      parameter (lt=lxyz*lelt)

      real ah(lx1*lx1),bh(lx1),ch(lx1*lx1),dh(lx1*lx1)
     $    ,zpts(2*lx1),wght(2*lx1)
      
      real x(lt),f(lt),r(lt),w(lt),p(lt),z(lt),c(lt)
      real g(6,lt)

      logical ifbrick
      integer iel0,ielN   ! element range per proc.
      integer nx0,nxN     ! poly. order range

#ifdef USE_CUDA
      call setup_cuda()
#endif

      call iniproc(mpi_comm_world)    ! has nekmpi common block
      call read_param(ifbrick,iel0,ielN,nx0,nxN)

c     GET PLATFORM CHARACTERISTICS
c     iverbose = 1
c     call platform_timer(iverbose)   ! iverbose=0 or 1

c     SET UP and RUN NEKBONE
      do nx1=nx0,nxN
         call init_dim
         do nelt=iel0,ielN
           call init_mesh(ifbrick)
           call proxy_setupds    (gsh)     ! Has nekmpi common block
           call set_multiplicity (c)       ! Inverse of counting matrix

           call proxy_setup(ah,bh,ch,dh,zpts,wght,g) 

           niter = 100
           n     = nx1*ny1*nz1*nelt

           call set_f(f,c,n)
           call cg(x,f,g,c,r,w,p,z,n,niter,flop_cg)

           call nekgsync()

           call set_timer_flop_cnt(0)
           call cg(x,f,g,c,r,w,p,z,n,niter,flop_cg)
           call set_timer_flop_cnt(1)
           call gs_free(gsh)
         enddo
      enddo

c     TEST BANDWIDTH BISECTION CAPACITY
c     call xfer(np,cr_h)

#ifdef USE_CUDA
      call teardown_cuda()
#endif USE_CUDA

      call exitt0

      end
c--------------------------------------------------------------
      subroutine set_f(f,c,n)
      real f(n),c(n)

      do i=1,n
         arg  = 1.e9*(i*i)
         arg  = 1.e9*cos(arg)
         f(i) = sin(arg)
      enddo

      call dssum(f)
      call col2 (f,c,n)

      return
      end
c-----------------------------------------------------------------------
      subroutine init_dim

C     Transfer array dimensions to common

      include 'SIZE'
      include 'INPUT'
 
      ny1=nx1
      nz1=nx1
 
      ndim=ldim

      return
      end
c-----------------------------------------------------------------------
      subroutine init_mesh(ifbrick)
      include 'SIZE'
      include 'TOTAL'
      logical ifbrick
      integer e,eg,offs
 

      if(.not.ifbrick) then   ! A 1-D array of elements of length P*lelt
  10     continue
         nelx = nelt*np
         nely = 1
         nelz = 1
   
         do e=1,nelt
            eg = e + nid*nelt
            lglel(e) = eg
         enddo
      else              ! A 3-D block of elements 
        call cubic(npx,npy,npz,np)  !xyz distribution of total proc
        call cubic(mx,my,mz,nelt)   !xyz distribution of elements per proc
      
        if(mx.eq.nelt) goto 10

        nelx = mx*npx
        nely = my*npy 
        nelz = mz*npz

        e = 1
        offs = (mod(nid,npx)*mx) + npx*(my*mx)*(mod(nid/npx,npy)) 
     $      + (npx*npy)*(mx*my*mz)*(nid/(npx*npy))
        do k = 0,mz-1
        do j = 0,my-1
        do i = 0,mx-1
           eg = offs+i+(j*nelx)+(k*nelx*nely)+1
           lglel(e) = eg
           e        = e+1
        enddo
        enddo
        enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine cubic(mx,my,mz,np)

        rp = np**(1./3.)
        mz = rp*(1.01)

        do iz=mz,1,-1
           myx = np/iz
           nrem = np-myx*iz
           if (nrem.eq.0) goto 10
        enddo
   10   mz = iz
        rq = myx**(1./2.)
        my = rq*(1.01)
        do iy=my,1,-1
           mx = myx/iy
           nrem = myx-mx*iy
           if (nrem.eq.0) goto 20
        enddo
   20   my = iy

        mx = np/(mz*my)

      return
      end
c-----------------------------------------------------------------------
      subroutine set_multiplicity (c)       ! Inverse of counting matrix
      include 'SIZE'
      include 'TOTAL'

      real c(1)

      n = nx1*ny1*nz1*nelt

      call rone(c,n)
      call gs_op(gsh,c,1,1,0)  ! Gather-scatter operation  ! w   = QQ  w

      do i=1,n
         c(i) = 1./c(i)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine set_timer_flop_cnt(iset)
      include 'SIZE'
      include 'TOTAL'

      real time0,time1
      save time0,time1

      if (iset.eq.0) then
         flop_a  = 0
         flop_cg = 0
         time0   = dnekclock()
      else
        time1   = dnekclock()-time0
        if (time1.gt.0) mflops = (flop_a+flop_cg)/(1.e6*time1)
        if (nid.eq.0) write(6,1) mflops,flop_a,flop_cg,time1,nelt,np,nx1
    1   format(1p4e12.4,' flops',3i7)
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine xfer(np,gsh)
      include 'SIZE'
      parameter(npts_max = lx1*ly1*lz1*lelt)

      real buffer(2,npts_max)
      integer ikey(npts_max)


      nbuf = 800
      npts = 1
      do itest=1,200
         npoints = npts*np

         call load_points(buffer,nppp,npoints,npts,nbuf)
         iend   = mod1(npoints,nbuf)
         istart = 1
         if(nid.ne.0)istart = iend+(nid-1)*nbuf+1
         do i = 1,nppp
            icount=istart+(i-1)
            ikey(i)=mod(icount,np)
         enddo

         call nekgsync
         time0 = dnekclock()
         do loop=1,50
            call crystal_tuple_transfer(gsh,nppp,npts_max,
     $                ikey,1,ifake,0,buffer,2,1)
         enddo
         time1 = dnekclock()
         etime = (time1-time0)/50

         if (nid.eq.0) write(6,1) np,npts,npoints,etime
   1     format(2i7,i10,1p1e12.4,' bandwidth' )
         npts = 1.02*(npts+1)
         if (npts.gt.npts_max) goto 100
      enddo
 100  continue

      return
      end
c-----------------------------------------------------------------------
      subroutine load_points(buffer,nppp,npoints,npts,nbuf)
      include 'SIZE'
      include 'PARALLEL'

      real buffer(2,nbuf)

      nppp=0
      if(nbuf.gt.npts) then
       npass = 1+npoints/nbuf

       do ipass = 1,npass
          if(nid.eq.ipass.and.ipass.ne.npass) then
            do i = 1,nbuf
             buffer(1,i)=i
             buffer(2,i)=nid
            enddo
            nppp=nbuf
          elseif (npass.eq.ipass.and.nid.eq.0) then
            mbuf=mod1(npoints,nbuf)
            do i=1,mbuf
               buffer(1,i)=i
               buffer(2,i)=nid
            enddo
            nppp=mbuf
          endif
       enddo
      else
       do i = 1,npts
          buffer(1,i)=i
          buffer(2,i)=nid
       enddo
       nppp=npts
      endif

      return
      end
c----------------------------------------------------------------------
      subroutine read_param(ifbrick,iel0,ielN,nx0,nxN)
      include 'SIZE'
      logical ifbrick
      integer iel0,ielN,nx0,nxN

      !open .rea
      if(nid.eq.0) then
         open(unit=9,file='data.rea',status='old') 
         read(9,*,err=100) ifbrick
         read(9,*,err=100) iel0,ielN
c        read(9,*,err=100) nx0,nxN
         close(9)
      endif
      call bcast(ifbrick,4)
      call bcast(iel0,4)
      call bcast(ielN,4)
      nx0=lx1
      nxN=lx1
c     call bcast(nx0,4)
c     call bcast(nxN,4)
      if(iel0.gt.ielN.or.nx0.gt.nxN) goto 200

      return

  100 continue
      write(6,*) "ERROR READING data.rea....ABORT"
      call exitt0

  200 continue
      write(6,*) "ERROR data.rea :: iel0 > ielN or nx0 > nxN :: ABORT"
      call exitt0
  
      return
      end
c-----------------------------------------------------------------------
