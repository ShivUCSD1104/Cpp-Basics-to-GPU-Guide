### Streaming Multiprocessors and Warps

The GPUs that CUDA applications run on have processing units called **streaming multiprocessors**, or **SMs**. During kernel execution, blocks of threads are given to SMs to execute. In order to support the GPU's ability to perform as many parallel operations as possible, performance gains can often be had by:

**choosing a grid size that has a number of blocks that is a multiple of the number of SMs on a given GPU**

Additionally, SMs create, manage, schedule, and execute groupings of 32 threads from within a block called **warps**.  
Performance gains can also be had by:

**choosing a block size that has a number of threads that is a multiple of 32**

In order to support portability, since the number of SMs on a GPU can differ depending on the specific GPU being used, the number of SMs should not be hard-coded into a code bases. Rather, this information should be acquired programatically.

The following shows how, in CUDA C/C++, to obtain a C struct which contains many properties about the currently active GPU device, including its number of SMs:

```cpp
int deviceId;
cudaGetDevice(&deviceId); // `deviceId` now points to the id of the currently active GPU.

cudaDeviceProp props;
cudaGetDeviceProperties(&props, deviceId); 
// `props` now has many useful properties about the active GPU device.

//example
props.major
props.minor
```

**How to get the number of SMs**
```cpp
cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
```

Optimized example: [[CUDA 1.5.5 - Optimized Vector Addition]]