#### Printing digits 0 to 9 
```cpp
__global__ void loop(int N)
{
	printf("This is iteration number %d\n", threadIdx.x);
}

int main()
{
	int N = 10;
	loop<<<1,N>>>(N);
	cudaDeviceSynchronize();
}
```

#### Embarrassing Parallelization 

**Mapping Threads to Individual Data**
Essentially mapping a 1 dim array to multiple blocks and threads
```cpp
threadIdx.x + blockIdx.x * blockDim.x = dataIndex
```

```cpp
__global__ void loop(int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("This is iteration number %d\n", idx);
}

int main()
{
  int N = 10;
  loop<<<5,2>>>(N);
  cudaDeviceSynchronize();
}
```

#### Malloc and Free in CUDA
1. malloc --> ``` cudaMallocManaged(ref , size) ```
2. free --> ```cudaFree(ref)```

```cpp
__global__ void initializeElementsTo(int initialValue, int *a, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < N){
        a[i] = initialValue;
    }
}

int main()
{
  int N = 1000;

  int *a;
  size_t size = N * sizeof(int);

  cudaMallocManaged(&a, size);
  
  size_t threads = 256;
  size_t blocks = 4;

  int initVal = 6;

  initializeElementsTo<<<blocks, threads>>>(initialValue, a, N);
  cudaDeviceSynchronize();

  for (int i = 0; i < N; ++i)
  {
    if(a[i] != initVal)
    {
      printf("FAILURE: target value: %d\t a[%d]: %d\n", initVal, i, a[i]);
      cudaFree(a);
      exit(1);
    }
  }
  printf("SUCCESS!\n");

  cudaFree(a);
}
```

#### Grid Stride Loops

```cpp
void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

__global__
void doubleElements(int *a, int N)
{
    int idx, gridStride;
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    gridStride = gridDim.x * blockDim.x; 
    if (idx < N)
    {
        for(int i = idx; i < N; i += gridStride){
            a[i] *= 2;
        }
    }
}

bool checkElementsAreDoubled(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    if (a[i] != i*2) return false;
  }
  return true;
}

int main()
{
  // N is greater than the size of the grid (see below).

  int N = 10000;
  int *a;

  size_t size = N * sizeof(int);
  cudaMallocManaged(&a, size);

  init(a, N);

  //The size of this grid is 256*32 = 8192

  size_t threads_per_block = 256;
  size_t number_of_blocks = 32;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  cudaDeviceSynchronize();

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  cudaFree(a);
}

```
