`nsys profile` will generate a report file which can be used in a variety of manners, including for use in visual profiling with Nsight Systems, which we will look at in more detail in the following section.

Here we use the `--stats=true` flag to indicate we would like summary statistics printed. 

- Operating System Runtime Summary (`osrt_sum`)
- **CUDA API Summary (`cuda_api_sum`)**
- **CUDA Kernel Summary (`cuda_gpu_kern_sum`)**
- **CUDA Memory Time Operation Summary (`cuda_gpu_mem_time_sum`)**
- **CUDA Memory Size Operation Summary (`cuda_gpu_mem_size_sum`)**

