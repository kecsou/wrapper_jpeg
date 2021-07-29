
all: program

program: libwrapperjpeg.so
	gcc main.c -L. -lwrapperjpeg -lwnvgpu -o c_program
	g++ main.cpp -L. -lwrapperjpeg -lwnvgpu -o cpp_program

libwrapperjpeg.so: libwnvgpu.so
	gcc -shared -fPIC ../wrapperjpeg.cpp -L. -lwnvgpu -o libwrapperjpeg.so

libwnvgpu.so:
	nvcc --compiler-options '-fPIC' ../GPU_NVIDIA/compress.cu --shared -o libwnvgpu.so -lnvjpeg
#	nvcc -c --compiler-options '-fPIC' GPU_NVIDIA/compress.cu -o nvgpu.o -lnvjpeg
# nvcc -Xcompiler -static -o nvgpu.o GPU_NVIDIA/compress.cu -lnvjpeg

clean:
	rm -f *.o *.so program exec a.out output.jpg c_program cpp_program
