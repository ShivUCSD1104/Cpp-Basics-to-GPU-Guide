
```cpp
#include <stdio.h>

#define N 2048 * 2048 // Number of elements in each vector
__global__ void saxpy(int * a, int * b, int * c)
{
    int tid = blockIdx.x * blockDim.x * threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N; i += stride)
        c[i] = 2 * a[i] + b[i];
}

int main()
{
    int deviceId;
    int SMs;
    
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&SMs, cudaDevAttrMultiProcessorCount, deviceId);

    int *a, *b, *c;

    int size = N * sizeof (int); // The total number of bytes per vector

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    // Initialize memory
    for( int i = 0; i < N; ++i )
    {
        a[i] = 2;
        b[i] = 1;
        c[i] = 0;
    }

    cudaMemPrefetchAsync(a, size, deviceId);
    cudaMemPrefetchAsync(b, size, deviceId);
    cudaMemPrefetchAsync(c, size, deviceId);

    size_t threads_per_block = 256;
    size_t number_of_blocks = 32 * SMs;

    saxpy <<< number_of_blocks, threads_per_block >>> ( a, b, c );

    cudaDeviceSynchronize();

    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");

    cudaFree( a ); cudaFree( b ); cudaFree( c );
}

```

