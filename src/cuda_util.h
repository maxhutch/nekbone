#define CUDA_ERROR_CHECK
 
#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaCheckError() __cudaCheckError( __FILE__, __LINE__ )

#include "cuda_runtime.h"
#include "cuda.h"
 
inline void __cudaSafeCall( cudaError_t err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
  if ( cudaSuccess != err )
  {
    fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }
#endif
   
  return;
}
 
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
  cudaError_t err = cudaGetLastError();
  if ( cudaSuccess != err )
  {
    fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }
   
  // More careful checking. However, this will affect performance.
  // // Comment away if needed.
  //err = cudaDeviceSynchronize();
/*   if( cudaSuccess != err ) */
/*   { */
/*     fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", */
/*     file, line, cudaGetErrorString( err ) ); */
/*     exit( -1 ); */
/*   } */
#endif
    
  return;
}

// Double precision atomicAdd in software
static 
__device__ 
__forceinline__
double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = 
      atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val +
          __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);

}


