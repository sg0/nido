#ifndef CUDA_WRAPPER_HPP_ 
#define CUDA_WRAPPER_HPP_

#include <iostream>
#ifdef debug
#define CudaCall(ans) { CudaAssert((ans), __FILE__, __LINE__); }
inline void CudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) 
                << " " << file << " " << line << std::endl;
      if (abort) 
          exit(code);
   }
}
#else
#define CudaCall(ans) {ans;}
#endif

#ifdef debug
#define CudaLaunch(kernel) \
{ \
    kernel; \
    CudaCall(cudaPeekAtLastError()); \
    CudaCall(cudaDeviceSynchronize()); \
}
#else
#define CudaLaunch(kernel) \
{ \
    kernel; \
}
#endif

#define CudaMalloc(ptr,size) \
{\
    CudaCall(cudaMalloc((void**)&ptr,size));\
}

#define CudaMallocHost(ptr, size)\
{\
    CudaCall(cudaMallocHost((void**)&ptr, size)); \
}

#define CudaMemcpyHtoD(dest, src, size)\
CudaCall(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice))

#define CudaMemcpyDtoH(dest, src, size)\
CudaCall(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost))

#define CudaMemcpyAsyncHtoD(dest, src, size, stream) \
CudaCall(cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice, stream))

#define CudaMemcpyAsyncDtoH(dest, src, size, stream) \
CudaCall(cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, stream))

#define CudaFree(ptr) \
{\
    if(ptr != nullptr) \
        CudaCall(cudaFree(ptr));\
    ptr = nullptr; \
}

#define CudaFreeHost(ptr) \
{\
    if(ptr != nullptr) \
        CudaCall(cudaFreeHost(ptr)); \
    ptr = nullptr; \
}
  
#define CudaDeviceSynchronize()\
CudaCall(cudaDeviceSynchronize())

#define CudaMemset(src, val, size) \
{\
    CudaCall(cudaMemset(src, val, size));\
}

#define CudaHostRegister(ptr, size, flag)\
{\
    CudaCall(cudaHostRegister(ptr, size, flag)); \
}

#define CudaHostUnregister(ptr)\
{\
    CudaCall(cudaHostUnregister(ptr));\
} 

//define some cuda kernel configuration
#define TILESIZE01	32
#define TILESIZE02	16
#define TILESIZE03	8
#define TILESIZE04	4
#define TILESIZE05	2
#define WARPSIZE	32
#define BLOCKDIM01	64
#define BLOCKDIM02	128
#define BLOCKDIM03	256
#define BLOCKDIM04      512
#define MAX_BLOCKDIM	1024
#define MAX_GRIDDIM	65535

#ifdef USE_32BIT_GRAPH
#define MAX_FLOAT	1.0E+38
#else
#define MAX_FLOAT       1.0E+308
#endif

#ifdef USE_32BIT_GRAPH
struct less_int2
{
    __host__ __device__ bool operator()(const int2& a, const int2& b)
    {
        return (a.x != b.x) ? (a.x < b.x) : (a.y < b.y);
    };
};
#else
struct less_int2
{
    __host__ __device__ bool operator() (const longlong2& a, const longlong2& b)
    {
        return (a.x != b.x) ? (a.x < b.x) : (a.y < b.y);
    };
};
#endif

#endif
