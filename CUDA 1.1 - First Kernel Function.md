GPU functions are called kernels

To use CUDA, data values must be transferred from the host to the device. These transfers are costly in terms of performance and should be minimized. (See [Data Transfer Between Host and Device](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#data-transfer-between-host-and-device).) This cost has several ramifications:

- The complexity of operations should justify the cost of moving data to and from the device. Code that transfers data for brief use by a small number of threads will see little or no performance benefit. The ideal scenario is one in which many threads perform a substantial amount of work.
    
    For example, transferring two matrices to the device to perform a matrix addition and then transferring the results back to the host will not realize much performance benefit. The issue here is the number of operations performed per data element transferred. For the preceding procedure, assuming matrices of size NxN, there are N2 operations (additions) and 3N2 elements transferred, so the ratio of operations to elements transferred is 1:3 or O(1). Performance benefits can be more readily achieved when this ratio is higher. For example, a matrix multiplication of the same matrices requires N3 operations (multiply-add), so the ratio of operations to elements transferred is O(N), in which case the larger the matrix the greater the performance benefit. The types of operations are an additional factor, as additions have different complexity profiles than, for example, trigonometric functions. It is important to include the overhead of transferring data to and from the device in determining whether operations should be performed on the host or on the device.

Compiler - [[CUDA - NVCC]]
#### Basic CPU function vs GPU function in CUDA

```cpp
void CPUFunction()
{
  printf("This function is defined to run on the CPU.\n");
}

//global == can be invoked by GPU or CPU : must return void 
__global__ void GPUFunction()
{
  printf("This function is defined to run on the GPU.\n");
}

int main()
{
  CPUFunction();
	//specify the exec config 
  GPUFunction<<<1, 1>>>();
  //only resume action on CPU once GPU is complete since GPUfunction is asynchronous
  cudaDeviceSynchronize();
}
```

#### Exec config 
```cpp 
<<<A, B>>>
// where A is the number of blocks (threadgroups)
// where B is the number of threads per block 
```

A collection of blocks is a grid --> Grid of blocks of threads 
1024 Threads per block at MAX

**Thread Hierarchy**
```cpp

gridDim.x //number of blocks within grid
blockIdx.x //index of current block within grid
blockDim.x //number of threads in a block
threadIdx.x //index of current thread

```

Production code should systematically check the error code returned by each API call and check for failures in kernel launches by calling `cudaGetLastError()`