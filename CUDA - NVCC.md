`nvcc` will be very familiar to experienced `gcc` users. Compiling, for example, a some-CUDA.cu file, is simply:

`nvcc -o out some-CUDA.cu -run`

- `nvcc` is the command line command for using the `nvcc` compiler.
- some-CUDA.cu is passed as the file to compile.
- The `o` flag is used to specify the output file for the compiled program.
- As a matter of convenience, providing the `run` flag will execute the successfully compiled binary.

