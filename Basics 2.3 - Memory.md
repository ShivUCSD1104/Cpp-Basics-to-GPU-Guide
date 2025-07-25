two main areas of memory commonly used in C++: the **stack** and the **heap**.

### **Differences in Memory Management (Stack vs. Heap)**

|**Feature**|**Stack**|**Heap**|
|---|---|---|
|**Scope**|Local variables within functions|Dynamically allocated variables|
|**Lifetime**|Automatically managed (when function exits)|Must be manually freed (e.g., using `free()` or `delete`)|
|**Memory Allocation**|Automatic (compile-time)|Dynamic (runtime)|
|**Size**|Limited size (often smaller)|Much larger, but prone to fragmentation|
|**Speed**|Faster (because of automatic management)|Slower (due to manual management)|
|**Example**|`int x = 3;`|`int* ptr = malloc(sizeof(int));`|

- The **stack** is a highly-ordered region of memory that handles function calls and local variables. When we call a function, its local variables and data are stored in the stack. This memory is automatically managed—when our function completes, its stack memory is released.
    
- The heap is an unordered region of dynamically allocated memory, which we manage manually using operators like `new` and `delete`. Unlike stack memory, our heap memory persists until we explicitly release it, giving us more flexibility but requiring more careful management.

In C/C++, when you need memory that **outlives** the scope of a function (for example, if you want to allocate a large amount of memory or data structures at runtime), you can allocate memory **dynamically** on the **heap**. There are various methods to allocate and deallocate memory.

#### **Using `malloc()` and `free()`**:

- **`malloc(size)`**: Allocates `size` bytes of memory on the heap and returns a pointer to the allocated memory. The memory is **uninitialized**.
    
- **`free(ptr)`**: Frees the memory previously allocated by `malloc`. This is important to prevent **memory leaks**.

    ``` cpp 
    int* ptr = (int*)malloc(sizeof(int));  // Allocates memory on the heap
    *ptr = 3;  // Sets value at allocated memory 
    free(ptr);  // Frees the allocated memory
    ```
    

#### **Using `new` and `delete`**:

- **`new`**: Allocates memory on the heap and returns a pointer to the allocated memory. The memory is **initialized** to zero for basic types.
    
- **`delete`**: Frees the memory allocated with `new`.
    
    ``` cpp 
    int* ptr = new int;  // Allocates memory on the heap and initializes
    *ptr = 3; 
    delete ptr;  // Frees the allocated memory
    ```

#### **Using `new[]` and `delete[]` for Arrays**:

- For arrays, use `new[]` and `delete[]` to allocate and free memory for multiple elements.
    
    ``` cpp
    int* arr = new int[10];  // Allocates an array of 10 integers on the heap 
    arr[0] = 3; delete[] arr;  // Frees the array
    ```
    