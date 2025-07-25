**CUDA FUNCTIONS NOT KERNELS**

Most CUDA functions return ```cudaError_t``` 
```cpp
cudaError_t err;
err = cudaMallocManaged(&a, N) // Assume the existence of `a` and `N`.

if (err != cudaSuccess) // `cudaSuccess` is provided by CUDA.
{
  printf("Error: %s\n", cudaGetErrorString(err)); 
  // `cudaGetErrorString` is provided by CUDA.
}
```

#### Macro For Error Handling

```cpp
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

int main()
{

/*
 * The macro can be wrapped around any function returning
 * a value of type `cudaError_t`.
 */

  checkCuda( cudaDeviceSynchronize() )
}
```

