ifndef CC
	CC=gcc
endif

all:
	mkdir -p bin
	$(CC) src/opencl-stream.c -o bin/opencl-stream -std=gnu99 -O3 -Wall -pedantic -lrt -lm -lOpenCL

clean:
	rm -r bin
