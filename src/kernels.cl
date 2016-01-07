// enable extension for OpenCL 1.1 and lower
#if __OPENCL_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// Initialize arrays
__kernel void initialiseArraysKernel(__global double * restrict A,
                                     __global double * restrict B,
                                     __global double * restrict C)
{
	size_t tid = get_global_id(0);

	A[tid] = 1.0;
	B[tid] = 2.0;
	C[tid] = 0.0;

}



// Copy kernels
__kernel void copyKernel1(__global const double * restrict A,
                          __global double * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid];
}
__kernel void copyKernel2(__global const double2 * restrict A,
                          __global double2 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid];
}
__kernel void copyKernel4(__global const double4 * restrict A,
                          __global double4 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid];
}
__kernel void copyKernel8(__global const double8 * restrict A,
                          __global double8 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid];
}
__kernel void copyKernel16(__global const double16 * restrict A,
                          __global double16 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid];
}



// Scale kernels
__kernel void scaleKernel1(const double scalar,
                           __global double * restrict B,
                           __global const double * restrict C)
{
	size_t tid = get_global_id(0);

	B[tid] = scalar*C[tid];
}
__kernel void scaleKernel2(const double scalar,
                           __global double2 * restrict B,
                           __global const double2 * restrict C)
{
	size_t tid = get_global_id(0);

	B[tid] = scalar*C[tid];
}
__kernel void scaleKernel4(const double scalar,
                           __global double4 * restrict B,
                           __global const double4 * restrict C)
{
	size_t tid = get_global_id(0);

	B[tid] = scalar*C[tid];
}
__kernel void scaleKernel8(const double scalar,
                           __global double8 * restrict B,
                           __global const double8 * restrict C)
{
	size_t tid = get_global_id(0);

	B[tid] = scalar*C[tid];
}
__kernel void scaleKernel16(const double scalar,
                           __global double16 * restrict B,
                           __global const double16 * restrict C)
{
	size_t tid = get_global_id(0);

	B[tid] = scalar*C[tid];
}



// Add kernels
__kernel void addKernel1(__global const double * restrict A,
                         __global const double * restrict B,
                         __global double * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid] + B[tid];
}
__kernel void addKernel2(__global const double2 * restrict A,
                         __global const double2 * restrict B,
                         __global double2 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid] + B[tid];
}
__kernel void addKernel4(__global const double4 * restrict A,
                         __global const double4 * restrict B,
                         __global double4 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid] + B[tid];
}
__kernel void addKernel8(__global const double8 * restrict A,
                         __global const double8 * restrict B,
                         __global double8 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid] + B[tid];
}
__kernel void addKernel16(__global const double16 * restrict A,
                         __global const double16 * restrict B,
                         __global double16 * restrict C)
{
	size_t tid = get_global_id(0);

	C[tid] = A[tid] + B[tid];
}



// Triad kernels
__kernel void triadKernel1(const double scalar,
                           __global double * restrict A,
                           __global const double * restrict B,
                           __global const double * restrict C)
{
	size_t tid = get_global_id(0);

	A[tid] = B[tid]*scalar + C[tid];
}
__kernel void triadKernel2(const double scalar,
                           __global double2 * restrict A,
                           __global const double2 * restrict B,
                           __global const double2 * restrict C)
{
	size_t tid = get_global_id(0);

	A[tid] = B[tid]*scalar + C[tid];
}
__kernel void triadKernel4(const double scalar,
                           __global double4 * restrict A,
                           __global const double4 * restrict B,
                           __global const double4 * restrict C)
{
	size_t tid = get_global_id(0);

	A[tid] = B[tid]*scalar + C[tid];
}
__kernel void triadKernel8(const double scalar,
                           __global double8 * restrict A,
                           __global const double8 * restrict B,
                           __global const double8 * restrict C)
{
	size_t tid = get_global_id(0);

	A[tid] = B[tid]*scalar + C[tid];
}
__kernel void triadKernel16(const double scalar,
                           __global double16 * restrict A,
                           __global const double16 * restrict B,
                           __global const double16 * restrict C)
{
	size_t tid = get_global_id(0);

	A[tid] = B[tid]*scalar*2.0 + C[tid];
}
