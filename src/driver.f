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
      real r_gpu(lt)
      real g(6,lt)
      real mfloplist(1024), avmflop
      integer icount

      logical ifbrick
      integer iel0,ielN,ielD   ! element range per proc.
      integer nx0,nxN,nxD      ! poly. order range
      integer npx,npy,npz      ! poly. order range
      integer mx,my,mz         ! poly. order range

      real*8 time_beg
      real*8 time_end

      call iniproc(mpi_comm_world)    ! has nekmpi common block
      call read_param(ifbrick,iel0,ielN,ielD,nx0,nxN,nxD,
     +                npx,npy,npz,mx,my,mz)

c     GET PLATFORM CHARACTERISTICS
c     iverbose = 1
c     call platform_timer(iverbose)   ! iverbose=0 or 1

      icount = 0

c     SET UP and RUN NEKBONE
      do nx1=nx0,nxN,nxD
         call init_dim
         do nelt=iel0,ielN,ielD
           call init_mesh(ifbrick,npx,npy,npz,mx,my,mz)
           call proxy_setupds    (gsh)     ! Has nekmpi common block
           call set_multiplicity (c)       ! Inverse of counting matrix

           call proxy_setup(ah,bh,ch,dh,zpts,wght,g) 

           niter = 100
           n     = nx1*ny1*nz1*nelt
           call set_f(f,c,n)

           ! Initialized the GPU arrays
           call cg_cuda_init(x,f,g,c,r,w,p,z,nx1,ny1,nz1,nelt,ldim,
     &                       dxm1,dxtm1,niter,flop_cg,gsh,nid)

           !Run cg iterations on host
           call cg(x,f,g,c,r,w,p,z,n,niter,flop_cg)

           ! Copy gpu results for future comparison
           do it=1,n
             r_gpu(it) = r(it)
           end do

           !Run cg iterations on device and copy back to host r vector
           call cg_cuda(r,1,flop_cg,flop_a)

           ! Compare the CPU and GPU solutions
           do it=1,n
             if (abs(r_gpu(it) - r(it)) .ge. 1e-12) then
               write(6,*) "CPU and GPU results don't match, i=",it
             endif 
           end do

           ! Rerun with timing, without copying back to host
           call nekgsync()

           if (nid .eq. 0) then
              write(6,*) '---------------------------------------------'
              write(6,*) 'CPU RESULTS'
              write(6,*) '---------------------------------------------'
           endif 
              
           ! Run same computation on host
           call set_timer_flop_cnt(0)
           time_beg = dnekclock()
           call cg(x,f,g,c,r,w,p,z,n,niter,flop_cg)
           time_end = dnekclock()
           if (nid .eq. 0) then
              write(6,*) "CPU time: ",time_end-time_beg
              write(6,*) "---------------------------------------------"
           endif 
           call set_timer_flop_cnt(1)

           call set_timer_flop_cnt(0)
c          call summary_start()

           if (nid .eq. 0) then
              write(6,*) '---------------------------------------------'
              write(6,*) 'GPU RESULTS'
              write(6,*) '---------------------------------------------'
           endif 
           !Run cg iterations on device
           time_beg = dnekclock()
           call cg_cuda(r,0,flop_cg,flop_a)
           time_end = dnekclock()
           if (nid .eq. 0) then
              write(6,*) "GPU time: ",time_end-time_beg
              write(6,*) "---------------------------------------------"
           endif 
           call set_timer_flop_cnt(1)

c          call summary_stop()
           call gs_free(gsh)

           call cg_cuda_free()
           icount = icount + 1
           mfloplist(icount)= mflops*np
         enddo
      enddo

      avmflop = 0.0
      do i = 1, icount
        avmflop = avmflop + mfloplist(i)
      end do

      if (icount .ne. 0) then
        avmflop = avmflop/icount
      end if

      if (nid .eq. 0) then
        write(6,1) avmflop
      end if
    1 format('Av MFlops = ', 1pe12.4)

c     TEST BANDWIDTH BISECTION CAPACITY
c     call xfer(np,cr_h)

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
      subroutine init_mesh(ifbrick,npx,npy,npz,mx,my,mz)
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
        if (npx*npy*npz .ne. np) then
          call cubic(npx,npy,npz,np)  !xyz distribution of total proc
        end if 
        if (mx*my*mz .ne. nelt) then
          call cubic(mx,my,mz,nelt)   !xyz distribution of elements per proc
        end if 
     
       write(6,*) "inside init" 
c       if(mx.eq.nelt) goto 10

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

      write(6,*) "done init" 
      if (nid.eq.0) then
        write(6,*) "Processes: npx= ", npx, " npy= ", npy, " npz= ", npz
        write(6,*) "Local Elements: mx= ", mx, " my= ", my, " mz= ", mz
        write(6,*) "Elements: nelx= ", nelx, " nely= ", nely,
     &             " nelz= ", nelz
      end if

      return
      end
c-----------------------------------------------------------------------
      subroutine cubic(mx,my,mz,np)

      mx = np
      my = 1
      mz = 1
      ratio = np

      iroot3 = np**(1./3.) + 0.000001
      do i= iroot3,1,-1
        iz = i
        myx = np/iz
        nrem = np-myx*iz

        if (nrem.eq.0) then
          iroot2 = myx**(1./2.) + 0.000001
          do j=iroot2,1,-1
            iy = j
            ix = myx/iy
            nrem = myx-ix*iy
            if (nrem.eq.0) goto 20
          enddo
   20     continue

          if (ix < iy) then
            it = ix
            ix = iy
            iy = it
          end if      

          if (ix < iz) then
            it = ix
            ix = iz
            iz = it
          end if      

          if (iy < iz) then
            it = iy
            iy = iz
            iz = it
          end if      

          if ( REAL(ix)/iz < ratio) then
            ratio = REAL(ix)/iz
            mx = ix
            my = iy
            mz = iz
          end if 

        end if
      enddo

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
         ax_time= 0.0
         axe_time= 0.0
         gsop_time= 0.0
         axmisc_time= 0.0
         time0   = dnekclock()
      else
        time1   = dnekclock()-time0
        if (time1.gt.0) mflops = (flop_a+flop_cg)/(1.e6*time1)
        if (nid.eq.0) then
          write(6,1) nelt,np,nx1, nelt*np
          write(6,2) mflops*np, mflops
          write(6,3) flop_a,flop_cg,time1
          write(6,4) ax_time, axe_time, gsop_time, axmisc_time
        endif
    1   format('nelt = ', i7, ', np = ', i9, ', nx1 = ', i7,
     &         ', elements =', i10 )
    2   format('Tot MFlops = ', 1pe12.4, ', MFlops = ', e12.4)
    3   format('Ax FOp = ', 1pe12.4, ', CG FOp = ', e12.4,
     &         ', Solve Time = ', e12.4)
    4   format('Ax time = ', 1pe12.4, ', AxE time = ', e12.4,
     &         ', gsop time = ', e12.4, ', Ax misc time = ', e12.4)
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
      subroutine read_param(ifbrick,iel0,ielN,ielD,nx0,nxN,nxD,
     +                      npx,npy,npz,mx,my,mz)
      include 'SIZE'
      logical ifbrick
      integer iel0,ielN,ielD,nx0,nxN,nxD,npx,npy,npz,mx,my,mz

      !open .rea
      if(nid.eq.0) then
         open(unit=9,file='data.rea',status='old') 
         read(9,*,err=100) ifbrick
         read(9,*,err=100) iel0,ielN,ielD
         read(9,*,err=100) nx0,nxN,nxD
         read(9,*,err=100) npx,npy,npz
         read(9,*,err=100) mx,my,mz
         close(9)
      endif
      call bcast(ifbrick,4)
      call bcast(iel0,4)
      call bcast(ielN,4)
      call bcast(ielD,4)
c     nx0=lx1
c     nxN=lx1
      call bcast(nx0,4)
      call bcast(nxN,4)
      call bcast(nxD,4)
      call bcast(npx,4)
      call bcast(npy,4)
      call bcast(npz,4)
      call bcast(mx,4)
      call bcast(my,4)
      call bcast(mz,4)
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
