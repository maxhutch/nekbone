/** \file cuda_main.cu
 * \brief Contains global initialization/destruction routines for cuda contexts. 
 */

/** \mainpage notitle  
 * \n\n\n
 * 
 * We will incorporate the use of GPUs in VASP to replace existing
 * FFTs and certain linear algebra algorithms in the "subspace diagonalization" portion
 * of code. These parts of VASP constitute computational bottlenecks for ab initio
 * simulations currently being performed in the DoD. It is hoped that the application
 * of GPUs will offer a significant increase in speed for one of the most heavily used
 * scientific codes in the DoD. This work is part of a suite of proposals that will enhance
 * VASP in several respects and profile the new code on recently purchased DoD
 * machines. The resulting improvements will be available to all DoD license holders with
 * DSRC accounts.
 */

#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cublas_v2.h"
#include <iostream>
#include <unistd.h>
#ifdef __PARA
#include "mpi.h" 
#endif

#define CUDA_CALL(function) {\
cudaError_t err = function; \
if (err != cudaSuccess) \
  fprintf(stderr, "CURROR [%s,%d]: %s \n", \
  __FILE__,  __LINE__, cudaGetErrorString(err)); \
}

using namespace std;

/** Simple, custom representation of a cuda device (for a list).
 * 
 * Only called in main.F
 */
typedef struct cuda_device{
  int num;
  int used;
  int avail;
  cudaDeviceProp properties;
} cuda_device;


int nPE, myPE;
int this_device;
int already_setup = 0;

cudaStream_t* streams;
cublasHandle_t cublas_ctx;

/** Initialize the cuda context, which includes picking a device */
extern "C" int setup_cuda_(void){
  if (already_setup  == 1) return EXIT_FAILURE;
  already_setup = 1;


  int deviceCount;
  CUDA_CALL(cudaGetDeviceCount(&deviceCount));
  
  cuda_device* device_list = (cuda_device*) malloc(deviceCount * sizeof(cuda_device));
  int device;
  int best_device;
  int best_used, used;
  size_t best_mem;
  int i, PE;
  FILE* dev_file;
  char buffer[100];

  /* Get MPI Information */
#ifdef __PARA
  MPI_Comm_size(MPI_COMM_WORLD, &nPE);
  MPI_Comm_rank(MPI_COMM_WORLD, &myPE);
#else
  nPE = 1; myPE = 0;
#endif

/* Distribute GPU resources sequentially */
  for (PE = 0; PE < nPE; PE++){
    if (myPE == PE){

  /* Init the device list */
  for (device = 0; device < deviceCount; ++device) {
    device_list[device].num = device;
    device_list[device].used = 0;
    cudaGetDeviceProperties(&(device_list[device].properties), device);

    if (device == 0 && device_list[device].properties.major == 9999){
      fprintf(stderr, "There is no device supporting CUDA.\n");
      return EXIT_FAILURE;
    }
  }

  /* Check to see if there is a CUDA device file */
  if(access("/tmp/cuda_devices", F_OK) == -1) {
    /* File doesn't exist; all devices are available */
  } else {
    /* File exists; read used devices */
    dev_file = fopen("/tmp/cuda_devices", "r");
    while(fscanf(dev_file, "%d %d", &device, &used) != EOF){
      device_list[device].used = used;
    }
    fclose(dev_file);
  }

  /* Select a device.  We'll require arch 2.* and pick by memory */
  best_device = -1;  best_mem = 0; best_used = 10000000;
  for (device = 0; device < deviceCount; device++){
    if (device_list[device].used < best_used &&
        device_list[device].properties.major >= 2){
      best_device = device;
      best_mem = device_list[device].properties.totalGlobalMem;
      best_used = device_list[device].used;
    } else if (device_list[device].used == best_used &&
               device_list[device].properties.totalGlobalMem > best_mem &&
               device_list[device].properties.major >= 2){
      best_device = device;
      best_mem = device_list[device].properties.totalGlobalMem;
      best_used = device_list[device].used;
    }
  }
  if (best_device == -1){
    fprintf(stderr, "There is no suitable device.\n");
    this_device = -1;
    return EXIT_FAILURE;
  }

  cudaSetDevice(best_device);
  device_list[best_device].used++; 
  this_device = best_device;

  /* Re-write the device file */
  dev_file = fopen("/tmp/cuda_devices", "w");
  for (device = 0; device < deviceCount; device++)
    fprintf(dev_file, "%d %d\n", device_list[device].num, device_list[device].used);
  fclose(dev_file);  
 
  /* Now print some things to screen */
  fprintf(stdout,"#======================================#\n");
  fprintf(stdout,"#                                      #\n");
  fprintf(stdout,"#  Devices (* = used):                 #\n");
  for (device = 0; device < deviceCount; ++device) {
    if (device_list[device].used == 0 || device == this_device)
      fprintf(stdout, "#             %-20s - %d #\n", device_list[device].properties.name, device);
    else
      fprintf(stdout, "#          *  %-20s - %d #\n", device_list[device].properties.name, device);
  } 
  fprintf(stdout,"#                                      #\n");
  fprintf(stdout,"#  Chose to use:                       #\n");
  fprintf(stdout,"#   %-30s - %d #\n", device_list[best_device].properties.name, best_device);
  fprintf(stdout,"#                                      #\n");
  fprintf(stdout,"#======================================#\n");
  fflush(stdout);

#ifdef __PARA
  MPI_Barrier(MPI_COMM_WORLD);
#endif

    }
  }

  free(device_list);
 
  streams = (cudaStream_t*) malloc(32 * sizeof(cudaStream_t));
  for (i = 0; i < 32; i++)
      cudaStreamCreate(streams + i);
  cublasCreate(&cublas_ctx);

  return EXIT_SUCCESS;
}

/** Destroy the cuda context. 
 *
 * Only called in main.F
 */
extern "C" int teardown_cuda_(void){
  cublasDestroy(cublas_ctx);
  int i;
  for (i = 0; i < 32; i++)
    cudaStreamDestroy(streams[i]);
  free(streams);


  int deviceCount;
  CUDA_CALL(cudaGetDeviceCount(&deviceCount));
  
  cuda_device* device_list = (cuda_device*) malloc(deviceCount * sizeof(cuda_device));
  int device;
  int best_device;
  int best_used, used;
  size_t best_mem;
  int PE;
  FILE* dev_file;
  char buffer[100];

  /* Remove device from the device file */
  for (PE = 0; PE < nPE; PE++){
    if (myPE == PE){

  /* Init the device list */
  for (device = 0; device < deviceCount; ++device) {
    device_list[device].num = device;
    device_list[device].used = 0;
    cudaGetDeviceProperties(&(device_list[device].properties), device);

    if (device == 0 && device_list[device].properties.major == 9999){
      fprintf(stderr, "There is no device supporting CUDA.\n");
      return EXIT_FAILURE;
    }
  }

  /* Check to see if there is a CUDA device file */
  if(access("/tmp/cuda_devices", F_OK) == -1) {
    /* File doesn't exist; something is wrong */
    return EXIT_FAILURE;
  } else {
    /* File exists; read used devices */
    dev_file = fopen("/tmp/cuda_devices", "r");
    while(fscanf(dev_file, "%d %d", &device, &used) != EOF){
      device_list[device].used = used;
    }
    fclose(dev_file);
  }

  /* Reduce run count on our device */
  device_list[this_device].used--;

  /* Re-write the device file */
  dev_file = fopen("/tmp/cuda_devices", "w");
  for (device = 0; device < deviceCount; device++)
    fprintf(dev_file, "%d %d\n", device_list[device].num, device_list[device].used);
  fclose(dev_file);  
 
#ifdef __PARA
  MPI_Barrier(MPI_COMM_WORLD);
#endif

    }
  }

  free(device_list);

  return EXIT_SUCCESS;
}

