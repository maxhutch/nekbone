### makefile automatically created by makenek 10/02/2013 12:57:33 ###
BINNAME=nekbone
CASENAME:=
CASEDIR:=/home/pcastonguay/Perforce_workspace_sw/sw/devrel/internal/miniapps/nekbone/nekbone-2.2.x.cuda/build
S:=/home/pcastonguay/Perforce_workspace_sw/sw/devrel/internal/miniapps/nekbone/nekbone-2.2.x.cuda/src/src
J:=$S/jl
OPT_INCDIR:=./
OBJDIR=obj
IFMPI:=true
F77:=mpif77
CC:=nvcc -arch=sm_35 --ptxas-options=-v
P:=-fcray-pointer -fdefault-real-8 -x f77-cpp-input
PPPO=
PPS= PTRSIZE8 MPI LONGINT8 UNDERSCORE GLOBAL_LONG_LONG
G:=
OPT_FLAGS_STD=
USR:=
USR_LFLAGS:=-L/usr/local/cuda-5.5/lib64 -lcudart -lcublas
USR_IFLAGS=
################################################################################

lFLAGS = $(USR_LFLAGS)
#iFLAGS = $(USR_IFLAGS) 
iFLAGS=-I/usr/lib/openmpi/include/ -I/usr/local/cuda5.5/include

PPS_F = $(patsubst %,$(PPPO)-D%,$(PPS))
PPS_C = $(patsubst %,-D%,$(PPS))

#NEW #########################################################################
EXTRA = driver.o mxm_wrapper.o prox_dssum.o prox_setup.o semhat.o \
speclib.o

################################################################################
# MXM 
MXM=mxm_std.o blas.o

# JL Routines ###################################################################
JO  = jl_
JL := -DPREFIX=jl_

JLCORE = $(JO)gs.o $(JO)sort.o $(JO)sarray_transfer.o $(JO)sarray_sort.o \
$(JO)gs_local.o $(JO)crystal.o $(JO)comm.o $(JO)tensor.o $(JO)fail.o \
$(JO)fcrystal.o 

COMM_MPI := comm_mpi.o
ifeq ($(IFMPI),false)
  COMM_MPI := ${COMM_MPI} mpi_dummy.o
endif

ifeq ($(IFMPI),false)
	DUMMY:= $(shell cp $S/mpi_dummy.h $S/mpif.h) 
else
	DUMMY:= $(shell rm -rf $S/mpif.h) 
endif

#####################################################################################
TMP0 = $(EXTRA) $(COMM_MPI) $(MXM)
NOBJS_F0 = $(patsubst %,$(OBJDIR)/%,$(TMP0))
TMP0c = $(JLCORE)
NOBJS_C0 = $(patsubst %,$(OBJDIR)/%,$(TMP0c))

NOBJS0 = $(NOBJS_F0) $(NOBJS_C0)
ORIG = $(patsubst %,$(OBJDIR)/%,cg.o math.o)
CUDACG = $(patsubst %,$(OBJDIR)/%,cg-cuda.o cgcuda.o math.o)

##############################################################################

L0=$(G) -O0
L2=$(G) -O2
L3=$(G) -O3
L4=$(L3)

FL0   = $(L0) $(P) $(PPS_F) -I$(CASEDIR) -I$S -I$(OPT_INCDIR)	
FL2i4 = $(L0)      $(PPS_F) -I$(CASEDIR) -I$S -I$(OPT_INCDIR)	
FL2   = $(L2) $(P) $(PPS_F) -I$(CASEDIR) -I$S -I$(OPT_INCDIR)	
FL3   = $(L3) $(P) $(PPS_F) -I$(CASEDIR) -I$S -I$(OPT_INCDIR)	
FL4   = $(L4) $(P) $(PPS_F) -I$(CASEDIR) -I$S -I$(OPT_INCDIR)	

cFL0   = $(L0) $(PPS_C) $(iFLAGS) -I$S
cFL2   = $(L2) $(PPS_C) $(iFLAGS) -I$S
cFL3   = $(L3) $(PPS_C) $(iFLAGS) -I$S
cFL4   = $(L4) $(PPS_C) $(iFLAGS) -I$S

##############################################################################
all : nekbone-cgcuda

objdir: 
	@mkdir $(OBJDIR) 2>/dev/null; cat /dev/null 

nekbone-orig: 	objdir $(NOBJS0) $(ORIG)
	$(F77) -o $@ $G $(NOBJS0) $(ORIG) $(lFLAGS)
	@if test -f $@; then \
	echo "#############################################################"; \
	echo "#                  Compilation successful!                  #"; \
	echo "#############################################################"; \
        size $@; \
        echo ""; \
	else \
	echo -e "\033[1;31;38m" "ERROR: Compilation failed!"; \
	echo -e "\033[0m"; \
	fi
ifeq ($(IFMPI),false) 
	@rm -rf $S/mpif.h
endif

nekbone-ax-f: 	objdir $(NOBJS0) $(CUDAF) 
	$(F77) -o $@ $G $(NOBJS0) $(CUDAF) $(lFLAGS)
	@if test -f $@; then \
	echo "#############################################################"; \
	echo "#                  Compilation successful!                  #"; \
	echo "#############################################################"; \
        size $@; \
        echo ""; \
	else \
	echo -e "\033[1;31;38m" "ERROR: Compilation failed!"; \
	echo -e "\033[0m"; \
	fi
ifeq ($(IFMPI),false) 
	@rm -rf $S/mpif.h
endif

nekbone-cgcuda: 	objdir $(NOBJS0) $(CUDACG) 
	$(F77) -o $@ $G $(NOBJS0) $(CUDACG) $(lFLAGS)
	@if test -f $@; then \
	echo "#############################################################"; \
	echo "#                  Compilation successful!                  #"; \
	echo "#############################################################"; \
        size $@; \
        echo ""; \
	else \
	echo -e "\033[1;31;38m" "ERROR: Compilation failed!"; \
	echo -e "\033[0m"; \
	fi
ifeq ($(IFMPI),false) 
	@rm -rf $S/mpif.h
endif


nekbone-ax-fomp: objdir $(NOBJS0) $(CUDAFOMP) 
	$(F77) -fopenmp -o $@ $G $(NOBJS0) $(CUDAFOMP) $(lFLAGS)
	@if test -f $@; then \
	echo "#############################################################"; \
	echo "#                  Compilation successful!                  #"; \
	echo "#############################################################"; \
        size $@; \
        echo ""; \
	else \
	echo -e "\033[1;31;38m" "ERROR: Compilation failed!"; \
	echo -e "\033[0m"; \
	fi
ifeq ($(IFMPI),false) 
	@rm -rf $S/mpif.h
endif

nekbone-ax-c: 	objdir $(NOBJS0) $(CUDAC) 
	$(F77) -o $@ $G $(NOBJS0) $(CUDAC) $(lFLAGS)
	@if test -f $@; then \
	echo "#############################################################"; \
	echo "#                  Compilation successful!                  #"; \
	echo "#############################################################"; \
        size $@; \
        echo ""; \
	else \
	echo -e "\033[1;31;38m" "ERROR: Compilation failed!"; \
	echo -e "\033[0m"; \
	fi
ifeq ($(IFMPI),false) 
	@rm -rf $S/mpif.h
endif

clean:
	rm -rf ./obj nekbone-* compiler.out makefile
ifeq ($(IFMPI),false) 
	@rm -rf $S/mpif.h
endif

$(NOBJS_F0) : SIZE
# CORE      ############################################################################
$(OBJDIR)/cg.o		 :$S/cg.f;			$(F77) -c $(FL4) $< -o $@
$(OBJDIR)/cg-cuda.o	 :$S/cg.f;			$(F77) -c $(FL4) $< -o $@ -DCUDA -DCUDAINIT
$(OBJDIR)/ax-f.o	 :$S/ax.f;			$(F77) -c $(FL4) $< -o $@
$(OBJDIR)/ax-fomp.o	 :$S/ax-omp.f;			$(F77) -c $(FL4) $< -o $@ -fopenmp
$(OBJDIR)/driver.o	 :$S/driver.f;			$(F77) -c $(FL2) $< -o $@ -DCUDA
$(OBJDIR)/math.o	 :$S/math.f;			$(F77) -c $(FL4) $< -o $@
$(OBJDIR)/math-omp.o	 :$S/math.f;			$(F77) -c $(FL4) $< -o $@ -fopenmp
$(OBJDIR)/prox_dssum.o   :$S/prox_dssum.f;		$(F77) -c $(FL2) $< -o $@
$(OBJDIR)/prox_setup.o	 :$S/prox_setup.f;		$(F77) -c $(FL4) $< -o $@
$(OBJDIR)/semhat.o	 :$S/semhat.f;			$(F77) -c $(FL4) $< -o $@
$(OBJDIR)/speclib.o	 :$S/speclib.f;			$(F77) -c $(FL2) $< -o $@
$(OBJDIR)/blas.o         :$S/blas.f; 		        $(F77) -c $(FL2i4) $< -o $@
$(OBJDIR)/byte_mpi.o	 :$S/byte_mpi.f;			$(F77) -c $(FL2) $< -o $@
$(OBJDIR)/comm_mpi.o	 :$S/comm_mpi.f;			$(F77) -c $(FL2) $< -o $@
$(OBJDIR)/mpi_dummy.o	 :$S/mpi_dummy.f;		$(F77) -c $(FL2) $< -o $@
# MXM       ############################################################################
$(OBJDIR)/mxm_wrapper.o	  :$S/mxm_wrapper.f;		$(F77) -c $(FL2) $< -o $@ 
$(OBJDIR)/mxm_std.o	  :$S/mxm_std.f;		$(F77) -c $(FL4) $< -o $@
$(OBJDIR)/bg_aligned3.o	  :$S/bg_aligned3.s;		$(CC) -c $< -o $@
$(OBJDIR)/bg_mxm3.o	  :$S/bg_mxm3.s;		$(CC) -c $< -o $@
$(OBJDIR)/bg_mxm44.o	  :$S/bg_mxm44.s;		$(CC) -c $< -o $@
$(OBJDIR)/bg_mxm44_uneven.o :$S/bg_mxm44_uneven.s;	$(CC) -c $< -o $@
$(OBJDIR)/k10_mxm.o	  :$S/k10_mxm.c;		$(CC)  -c $(cFL2) $(JL) $< -o $@
# C Files ##################################################################################
$(OBJDIR)/ax-c.o                 :$S/ax.c;		  $(CC) -c $(cFL2) $(JL) $< -o $@
$(OBJDIR)/cgcuda.o               :$S/cgcuda.cu;	  $(CC) -c $(cFL2) $(JL) $< -o $@
$(OBJDIR)/byte.o                 :$S/byte.c;              $(CC) -c $(cFL2) $(JL) $< -o $@
$(OBJDIR)/chelpers.o             :$S/chelpers.c;          $(CC) -c $(cFL2) $(JL) $< -o $@
$(OBJDIR)/$(JO)fail.o            :$(J)/fail.c;            $(CC) -c $(cFL2) $(JL) $< -o $@
$(OBJDIR)/$(JO)tensor.o          :$(J)/tensor.c;          $(CC) -c $(cFL2) $(JL) $< -o $@
$(OBJDIR)/$(JO)sort.o            :$(J)/sort.c;            $(CC) -c $(cFL2) $(JL) $< -o $@
$(OBJDIR)/$(JO)sarray_sort.o     :$(J)/sarray_sort.c;     $(CC) -c $(cFL2) $(JL) $< -o $@
$(OBJDIR)/$(JO)comm.o            :$(J)/comm.c;            $(CC) -c $(cFL2) $(JL) $< -o $@
$(OBJDIR)/$(JO)crystal.o         :$(J)/crystal.c;         $(CC) -c $(cFL2) $(JL) $< -o $@
$(OBJDIR)/$(JO)sarray_transfer.o :$(J)/sarray_transfer.c; $(CC) -c $(cFL2) $(JL) $< -o $@
$(OBJDIR)/$(JO)fcrystal.o        :$(J)/fcrystal.c;        $(CC) -c $(cFL2) $(JL) $< -o $@
$(OBJDIR)/$(JO)gs.o              :$(J)/gs.c;              $(CC) -c $(cFL2) $(JL) $< -o $@ -DCUDA
$(OBJDIR)/$(JO)gs_cuda.o         :$(J)/gs_cuda.cu;        $(CC) -c $(cFL2) $(JL) $< -o $@ -DCUDA
$(OBJDIR)/$(JO)gs_local.o        :$(J)/gs_local.c;        $(CC) -c $(cFL2) $(JL) $< -o $@ -DCUDA
