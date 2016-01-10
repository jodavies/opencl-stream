// TODO: - improve verification. Currently only the last iteration of the double16 kernels leaves
//         permenent data in the arrays. Stream interleaves all of the tests... not too hard to do here?
//       - configurable float/double?
//       - automatically run up to max work group size for current device?

#include <stdio.h>
#include <stdlib.h>
#include <time.h>        // clock_gettime()
#include <float.h>       // DBL_MIN

/* clCreateCommandQueue with 2.0 headers gives a warning about it being deprecated, avoid it */
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/opencl.h>


// Array size for tests. Needs to be big to sufficiently load device.
// Must be divisible by 16 (the largest vector type) and 256 (the largest local workgroup size tested)
#define TRYARRAYSIZE (0.5 * 1024*1024*1024/8)

// Number of times to run tests
#define NTIMES 50

// Print per-local-size results during test?
//#define VERBOSE

// Function prototypes
double GetWallTime(void);
void RunTest(cl_command_queue * queue, cl_kernel * kernel, size_t vecWidth, char * testName, int memops, int flops, size_t arraySize);
void VerifyResults(cl_command_queue * queue, cl_mem * device_A, double scalar, size_t arraySize);
// OpenCL Stuff
int InitialiseCLEnvironment(cl_platform_id**, cl_device_id***, cl_context*, cl_command_queue*, cl_program*, cl_ulong*, cl_ulong*);
void CleanUpCLEnvironment(cl_platform_id**, cl_device_id***, cl_context*, cl_command_queue*, cl_program*);
void CheckOpenCLError(cl_int err, int line);

char *kernelFileName = "src/kernels.cl";



int main(void)
{
	// Disable caching of binaries by nvidia implementation
	setenv("CUDA_CACHE_DISABLE", "1", 1);

	// Set up OpenCL environment
	cl_platform_id    *platform;
	cl_device_id      **device_id;
	cl_context        context;
	cl_command_queue  queue;
	cl_ulong          maxAlloc, globalMemSize;
	cl_program        program;
	cl_kernel         initialiseArraysKernel;
	cl_kernel         copyKernel1,copyKernel2,copyKernel4,copyKernel8,copyKernel16;
	cl_kernel         scaleKernel1,scaleKernel2,scaleKernel4,scaleKernel8,scaleKernel16;
	cl_kernel         addKernel1,addKernel2,addKernel4,addKernel8,addKernel16;
	cl_kernel         triadKernel1,triadKernel2,triadKernel4,triadKernel8,triadKernel16;
	cl_int            err;
	cl_mem            device_A, device_B, device_C;

	if (InitialiseCLEnvironment(&platform, &device_id, &context, &queue, &program, &maxAlloc, &globalMemSize) == EXIT_FAILURE) {
		printf("Error initialising OpenCL environment\n");
		return EXIT_FAILURE;
	}

	// Create Kernels. We have a kernel of each vector size (scalar(1), 2, 4, 8, 16) for the
	// stream functions copy, scale, add, triad.
	initialiseArraysKernel = clCreateKernel(program, "initialiseArraysKernel", &err);
	CheckOpenCLError(err, __LINE__);
	copyKernel1 = clCreateKernel(program, "copyKernel1", &err);
	copyKernel2 = clCreateKernel(program, "copyKernel2", &err);
	copyKernel4 = clCreateKernel(program, "copyKernel4", &err);
	copyKernel8 = clCreateKernel(program, "copyKernel8", &err);
	copyKernel16 = clCreateKernel(program, "copyKernel16", &err);
	CheckOpenCLError(err, __LINE__);
	scaleKernel1 = clCreateKernel(program, "scaleKernel1", &err);
	scaleKernel2 = clCreateKernel(program, "scaleKernel2", &err);
	scaleKernel4 = clCreateKernel(program, "scaleKernel4", &err);
	scaleKernel8 = clCreateKernel(program, "scaleKernel8", &err);
	scaleKernel16 = clCreateKernel(program, "scaleKernel16", &err);
	CheckOpenCLError(err, __LINE__);
	addKernel1 = clCreateKernel(program, "addKernel1", &err);
	addKernel2 = clCreateKernel(program, "addKernel2", &err);
	addKernel4 = clCreateKernel(program, "addKernel4", &err);
	addKernel8 = clCreateKernel(program, "addKernel8", &err);
	addKernel16 = clCreateKernel(program, "addKernel16", &err);
	CheckOpenCLError(err, __LINE__);
	triadKernel1 = clCreateKernel(program, "triadKernel1", &err);
	triadKernel2 = clCreateKernel(program, "triadKernel2", &err);
	triadKernel4 = clCreateKernel(program, "triadKernel4", &err);
	triadKernel8 = clCreateKernel(program, "triadKernel8", &err);
	triadKernel16 = clCreateKernel(program, "triadKernel16", &err);
	CheckOpenCLError(err, __LINE__);

	// Allocate device memory
	size_t sizeBytes = TRYARRAYSIZE * sizeof(double);
	if (sizeBytes > maxAlloc) sizeBytes = maxAlloc;
	while (3*sizeBytes > globalMemSize) {
		printf("Adjusting array size from %zuMB to %zuMB\n", sizeBytes, sizeBytes/2);
		sizeBytes /= 2;
	}
	// Ensure new array size is a multiple of 256, the largest local workgroup size tested
	size_t arraySize = sizeBytes/sizeof(double);
	if ( arraySize % 256 != 0) {
		// round down to multiple of 256
		printf("Adjusting array size from %zuMB to %zuMB\n", arraySize*sizeof(double), ((arraySize/256)*256)*sizeof(double));
		arraySize = (arraySize/256)*256;
		sizeBytes = arraySize*sizeof(double);
	}
	device_A = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytes, NULL, &err);
	device_B = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytes, NULL, &err);
	device_C = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytes, NULL, &err);
	CheckOpenCLError(err, __LINE__);

	// Set kernel arguemnts. Scale and Triad kernels involve multiplication by a scalar:
	const double scalar = 3.0;

	err  = clSetKernelArg(initialiseArraysKernel, 0, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(initialiseArraysKernel, 1, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(initialiseArraysKernel, 2, sizeof(cl_mem), &device_C);

	err |= clSetKernelArg(copyKernel1, 0, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(copyKernel1, 1, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(copyKernel2, 0, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(copyKernel2, 1, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(copyKernel4, 0, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(copyKernel4, 1, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(copyKernel8, 0, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(copyKernel8, 1, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(copyKernel16, 0, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(copyKernel16, 1, sizeof(cl_mem), &device_C);

	err |= clSetKernelArg(scaleKernel1, 0, sizeof(double), &scalar);
	err |= clSetKernelArg(scaleKernel1, 1, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(scaleKernel1, 2, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(scaleKernel2, 0, sizeof(double), &scalar);
	err |= clSetKernelArg(scaleKernel2, 1, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(scaleKernel2, 2, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(scaleKernel4, 0, sizeof(double), &scalar);
	err |= clSetKernelArg(scaleKernel4, 1, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(scaleKernel4, 2, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(scaleKernel8, 0, sizeof(double), &scalar);
	err |= clSetKernelArg(scaleKernel8, 1, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(scaleKernel8, 2, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(scaleKernel16, 0, sizeof(double), &scalar);
	err |= clSetKernelArg(scaleKernel16, 1, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(scaleKernel16, 2, sizeof(cl_mem), &device_C);

	err |= clSetKernelArg(addKernel1, 0, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(addKernel1, 1, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(addKernel1, 2, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(addKernel2, 0, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(addKernel2, 1, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(addKernel2, 2, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(addKernel4, 0, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(addKernel4, 1, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(addKernel4, 2, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(addKernel8, 0, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(addKernel8, 1, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(addKernel8, 2, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(addKernel16, 0, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(addKernel16, 1, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(addKernel16, 2, sizeof(cl_mem), &device_C);

	err |= clSetKernelArg(triadKernel1, 0, sizeof(double), &scalar);
	err |= clSetKernelArg(triadKernel1, 1, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(triadKernel1, 2, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(triadKernel1, 3, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(triadKernel2, 0, sizeof(double), &scalar);
	err |= clSetKernelArg(triadKernel2, 1, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(triadKernel2, 2, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(triadKernel2, 3, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(triadKernel4, 0, sizeof(double), &scalar);
	err |= clSetKernelArg(triadKernel4, 1, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(triadKernel4, 2, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(triadKernel4, 3, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(triadKernel8, 0, sizeof(double), &scalar);
	err |= clSetKernelArg(triadKernel8, 1, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(triadKernel8, 2, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(triadKernel8, 3, sizeof(cl_mem), &device_C);
	err |= clSetKernelArg(triadKernel16, 0, sizeof(double), &scalar);
	err |= clSetKernelArg(triadKernel16, 1, sizeof(cl_mem), &device_A);
	err |= clSetKernelArg(triadKernel16, 2, sizeof(cl_mem), &device_B);
	err |= clSetKernelArg(triadKernel16, 3, sizeof(cl_mem), &device_C);
	CheckOpenCLError(err, __LINE__);


	// Initialize arrays
	size_t initLocalSize = 32;
	size_t initGlobalSize = arraySize;
	err = clEnqueueNDRangeKernel(queue, initialiseArraysKernel, 1, NULL, &initGlobalSize, &initLocalSize, 0, NULL, NULL);
	clFinish(queue);


	// Fourth argument is the number of memory operations per output array item. Used in bandwidth calculation.
	// Fifth argument is the number of flops per output array item. Used in flops calculation.
	printf("---------------------------------------------------------------------------------------------------\n");
	printf("Function        Best Rate GB/s   Avg time   Min time   Max time   Best Workgroup Size   Best GFLOPS\n");
	printf("---------------------------------------------------------------------------------------------------\n");
	RunTest(&queue, &copyKernel1,  1,  "copyKernel1",  2, 0, arraySize);
	RunTest(&queue, &copyKernel2,  2,  "copyKernel2",  2, 0, arraySize);
	RunTest(&queue, &copyKernel4,  4,  "copyKernel4",  2, 0, arraySize);
	RunTest(&queue, &copyKernel8,  8,  "copyKernel8",  2, 0, arraySize);
	RunTest(&queue, &copyKernel16, 16, "copyKernel16", 2, 0, arraySize);
	printf("---------------------------------------------------------------------------------------------------\n");
	RunTest(&queue, &scaleKernel1,  1,  "scaleKernel1",  2, 1, arraySize);
	RunTest(&queue, &scaleKernel2,  2,  "scaleKernel2",  2, 1, arraySize);
	RunTest(&queue, &scaleKernel4,  4,  "scaleKernel4",  2, 1, arraySize);
	RunTest(&queue, &scaleKernel8,  8,  "scaleKernel8",  2, 1, arraySize);
	RunTest(&queue, &scaleKernel16, 16, "scaleKernel16", 2, 1, arraySize);
	printf("---------------------------------------------------------------------------------------------------\n");
	RunTest(&queue, &addKernel1,  1,  "addKernel1",  2, 1, arraySize);
	RunTest(&queue, &addKernel2,  2,  "addKernel2",  2, 1, arraySize);
	RunTest(&queue, &addKernel4,  4,  "addKernel4",  2, 1, arraySize);
	RunTest(&queue, &addKernel8,  8,  "addKernel8",  2, 1, arraySize);
	RunTest(&queue, &addKernel16, 16, "addKernel16", 2, 1, arraySize);
	printf("---------------------------------------------------------------------------------------------------\n");
	RunTest(&queue, &triadKernel1,  1,  "triadKernel1",  3, 2, arraySize);
	RunTest(&queue, &triadKernel2,  2,  "triadKernel2",  3, 2, arraySize);
	RunTest(&queue, &triadKernel4,  4,  "triadKernel4",  3, 2, arraySize);
	RunTest(&queue, &triadKernel8,  8,  "triadKernel8",  3, 2, arraySize);
	RunTest(&queue, &triadKernel16, 16, "triadKernel16", 3, 2, arraySize);
	printf("---------------------------------------------------------------------------------------------------\n");

	// Check results are correct
	VerifyResults(&queue, &device_A, scalar, arraySize);

	CleanUpCLEnvironment(&platform, &device_id, &context, &queue, &program);
	return 0;
}



void RunTest(cl_command_queue * queue, cl_kernel * kernel, size_t vecWidth, char * testName, int memops, int flops, size_t arraySize)
{
	size_t localSize;
	size_t bestLocalSize;
	size_t globalSize = arraySize/vecWidth;
	double bestTime = DBL_MAX, worstTime = DBL_MIN, totalTime = 0.0;
	int err;

	// Test local sizes from 2 to to 256, in powers of 2
	for (localSize = 1; localSize <= 256; localSize *= 2) {

		if (globalSize % localSize != 0) {
			printf("Error, localSize must divide globalSize! (%zu %% %zu = %zu)\n",
			       globalSize, localSize, globalSize%localSize);
		}

		double time = GetWallTime();

		for (int n = 0; n < NTIMES; n++) {
			err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		}
		clFinish(*queue);
		CheckOpenCLError(err, __LINE__);

		time = GetWallTime() - time;
		if (time < bestTime) {
			bestTime = time;
			bestLocalSize = localSize;
		}
		if (time > worstTime) {
			worstTime = time;
		}
		totalTime += time;

#ifdef VERBOSE
		printf("------------- localSize = %3zu, bandwidth = %7.3lf GB/s\n",
		        localSize, memops*NTIMES*arraySize*sizeof(double)/1024.0/1024.0/1024.0/(time));
#endif

	}

	printf("%13s   %14.3lf   %8.6lf   %8.6lf   %8.6lf   %19zu   %11.3lf\n",
	       testName, memops*NTIMES*arraySize*sizeof(double)/1024.0/1024.0/1024.0/bestTime, totalTime/NTIMES,
	       bestTime, worstTime, bestLocalSize, flops*NTIMES*arraySize/1.0e9/bestTime);
}



void VerifyResults(cl_command_queue *queue, cl_mem *device_A, double scalar, size_t arraySize)
{
	// Triad puts final values in array A, so retrieve it from the card. Allocate memory to recieve:
	size_t sizeBytes = arraySize * sizeof(double);
	double *checkA;
	checkA = malloc(sizeBytes);
	clEnqueueReadBuffer(*queue, *device_A, CL_TRUE, 0, sizeBytes, checkA, 0, NULL, NULL);

	// Unlike the original stream benchmark, we don't interleave the functions.
	// The initial values were: a = 1.0, b = 2.0, c = 0.0.
	double a = 1.0;
	double b = 2.0;
	double c = 0.0;
	// Copy
	c = a;
	b = scalar*c;
	c = a + b;
	a = b*scalar + c;

	int errors = 0;
	for (int i = 0; i < arraySize; i++) {
		if (checkA[i] != a) {
			errors++;
		}
	}
	if (errors != 0) {
		printf("Error in result!\n");
	}

}



// Return ns accurate walltime
double GetWallTime(void)
{
	struct timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	return (double)tv.tv_sec + 1e-9*(double)tv.tv_nsec;
}



// OpenCL functions
int InitialiseCLEnvironment(cl_platform_id **platform, cl_device_id ***device_id, cl_context *context, cl_command_queue *queue, cl_program *program, cl_ulong *maxAlloc, cl_ulong *globalMemSize)
{
	//error flag
	cl_int err;
	char infostring[1024];

	//get kernel from file
	FILE* kernelFile = fopen(kernelFileName, "rb");
	fseek(kernelFile, 0, SEEK_END);
	long fileLength = ftell(kernelFile);
	rewind(kernelFile);
	char *kernelSource = malloc(fileLength*sizeof(char));
	long read = fread(kernelSource, sizeof(char), fileLength, kernelFile);
	if (fileLength != read) printf("Error reading kernel file, line %d\n", __LINE__);
	fclose(kernelFile);

	//get platform and device information
	cl_uint numPlatforms;
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	*platform = calloc(numPlatforms, sizeof(cl_platform_id));
	*device_id = calloc(numPlatforms, sizeof(cl_device_id*));
	err |= clGetPlatformIDs(numPlatforms, *platform, NULL);
	CheckOpenCLError(err, __LINE__);
	cl_uint *numDevices;
	numDevices = calloc(numPlatforms, sizeof(cl_uint));

	for (int i = 0; i < numPlatforms; i++) {
		clGetPlatformInfo((*platform)[i], CL_PLATFORM_VENDOR, sizeof(infostring), infostring, NULL);
		printf("\n---OpenCL: Platform Vendor %d: %s\n", i, infostring);

		err = clGetDeviceIDs((*platform)[i], CL_DEVICE_TYPE_ALL, 0, NULL, &(numDevices[i]));
		if (err == CL_DEVICE_NOT_FOUND)
			continue;
		CheckOpenCLError(err, __LINE__);
		(*device_id)[i] = malloc(numDevices[i] * sizeof(cl_device_id));
		err = clGetDeviceIDs((*platform)[i], CL_DEVICE_TYPE_ALL, numDevices[i], (*device_id)[i], NULL);
		CheckOpenCLError(err, __LINE__);
		for (int j = 0; j < numDevices[i]; j++) {
			char deviceName[200];
			cl_device_fp_config doublePrecisionSupport = 0;
			clGetDeviceInfo((*device_id)[i][j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
			printf("---OpenCL:    Device found %d. %s\n", j, deviceName);
//			cl_ulong maxAlloc;
//			clGetDeviceInfo((*device_id)[i][j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAlloc), &maxAlloc, NULL);
//			printf("---OpenCL:       CL_DEVICE_MAX_MEM_ALLOC_SIZE: %lu MB\n", maxAlloc/1024/1024);
//			cl_uint cacheLineSize;
//			clGetDeviceInfo((*device_id)[i][j], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cacheLineSize), &cacheLineSize, NULL);
//			printf("---OpenCL:       CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: %u B\n", cacheLineSize);
			clGetDeviceInfo((*device_id)[i][j], CL_DEVICE_DOUBLE_FP_CONFIG,
					sizeof(doublePrecisionSupport), &doublePrecisionSupport, NULL);
			if ( doublePrecisionSupport == 0 )
				printf("---OpenCL:        Device %d does not support double precision!\n", j);
		}
	}

	// Get platform and device from user:
	int chosenPlatform = -1, chosenDevice = -1;
	if (numPlatforms == 1) {
		chosenPlatform = 0;
		printf("Auto-selecting platform %u.\n", chosenPlatform);
	} else while (chosenPlatform < 0) {
		printf("\nChoose a platform: ");
		scanf("%d", &chosenPlatform);
		if (chosenPlatform > (numPlatforms-1) || chosenPlatform < 0) {
			chosenPlatform = -1;
			printf("Invalid platform.\n");
		}
		if (numDevices[chosenPlatform] < 1) {
			chosenPlatform = -1;
			printf("Platform has no devices.\n");
		}
	}
	if (numDevices[chosenPlatform] == 1) {
		chosenDevice = 0;
		printf("Auto-selecting device %u.\n", chosenDevice);
	} else while (chosenDevice < 0) {
		printf("Choose a device: ");
		scanf("%d", &chosenDevice);
		if (chosenDevice > (numDevices[chosenPlatform]-1) || chosenDevice < 0) {
			chosenDevice = -1;
			printf("Invalid device.\n");
		}
	}
	printf("\n");

	//store global mem size and max allocation size
	clGetDeviceInfo((*device_id)[chosenPlatform][chosenDevice], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(*globalMemSize), globalMemSize, NULL);
	clGetDeviceInfo((*device_id)[chosenPlatform][chosenDevice], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(*maxAlloc), maxAlloc, NULL);

	//create a context
	*context = clCreateContext(NULL, 1, &((*device_id)[chosenPlatform][chosenDevice]), NULL, NULL, &err);
	CheckOpenCLError(err, __LINE__);
	//create a queue
	*queue = clCreateCommandQueue(*context, (*device_id)[chosenPlatform][chosenDevice], 0, &err);
	CheckOpenCLError(err, __LINE__);

	//create the program with the source above
//	printf("Creating CL Program...\n");
	*program = clCreateProgramWithSource(*context, 1, (const char**)&kernelSource, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error in clCreateProgramWithSource: %d, line %d.\n", err, __LINE__);
		return EXIT_FAILURE;
	}

	//build program executable
//	printf("Building CL Executable...\n");
	err = clBuildProgram(*program, 0, NULL, "-I. -I src/", NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error in clBuildProgram: %d, line %d.\n", err, __LINE__);
		char buffer[5000];
		clGetProgramBuildInfo(*program, (*device_id)[chosenPlatform][chosenDevice], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
		printf("%s\n", buffer);
		return EXIT_FAILURE;
	}

	// dump ptx
	size_t binSize;
	clGetProgramInfo(*program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binSize, NULL);
	unsigned char *bin = malloc(binSize);
	clGetProgramInfo(*program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &bin, NULL);
	FILE *fp = fopen("openclPTX.ptx", "wb");
	fwrite(bin, sizeof(char), binSize, fp);
	fclose(fp);
	free(bin);

	free(numDevices);
	free(kernelSource);
	return EXIT_SUCCESS;
}



void CleanUpCLEnvironment(cl_platform_id **platform, cl_device_id ***device_id, cl_context *context, cl_command_queue *queue, cl_program *program)
{
	//release CL resources
	clReleaseProgram(*program);
	clReleaseCommandQueue(*queue);
	clReleaseContext(*context);

	cl_uint numPlatforms;
	clGetPlatformIDs(0, NULL, &numPlatforms);
	for (int i = 0; i < numPlatforms; i++) {
		free((*device_id)[i]);
	}
	free(*platform);
	free(*device_id);
}



void CheckOpenCLError(cl_int err, int line)
{
	if (err != CL_SUCCESS) {
		char * errString;

		switch(err) {
			case   0: errString = "CL_SUCCESS"; break;
			case  -1: errString = "CL_DEVICE_NOT_FOUND"; break;
			case  -2: errString = "CL_DEVICE_NOT_AVAILABLE"; break;
			case  -3: errString = "CL_COMPILER_NOT_AVAILABLE"; break;
			case  -4: errString = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
			case  -5: errString = "CL_OUT_OF_RESOURCES"; break;
			case  -6: errString = "CL_OUT_OF_HOST_MEMORY"; break;
			case  -7: errString = "CL_PROFILING_INFO_NOT_AVAILABLE"; break;
			case  -8: errString = "CL_MEM_COPY_OVERLAP"; break;
			case  -9: errString = "CL_IMAGE_FORMAT_MISMATCH"; break;
			case -10: errString = "CL_IMAGE_FORMAT_NOT_SUPPORTED"; break;
			case -11: errString = "CL_BUILD_PROGRAM_FAILURE"; break;
			case -12: errString = "CL_MAP_FAILURE"; break;
			case -13: errString = "CL_MISALIGNED_SUB_BUFFER_OFFSET"; break;
			case -14: errString = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; break;
			case -15: errString = "CL_COMPILE_PROGRAM_FAILURE"; break;
			case -16: errString = "CL_LINKER_NOT_AVAILABLE"; break;
			case -17: errString = "CL_LINK_PROGRAM_FAILURE"; break;
			case -18: errString = "CL_DEVICE_PARTITION_FAILED"; break;
			case -19: errString = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"; break;
			case -30: errString = "CL_INVALID_VALUE"; break;
			case -31: errString = "CL_INVALID_DEVICE_TYPE"; break;
			case -32: errString = "CL_INVALID_PLATFORM"; break;
			case -33: errString = "CL_INVALID_DEVICE"; break;
			case -34: errString = "CL_INVALID_CONTEXT"; break;
			case -35: errString = "CL_INVALID_QUEUE_PROPERTIES"; break;
			case -36: errString = "CL_INVALID_COMMAND_QUEUE"; break;
			case -37: errString = "CL_INVALID_HOST_PTR"; break;
			case -38: errString = "CL_INVALID_MEM_OBJECT"; break;
			case -39: errString = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; break;
			case -40: errString = "CL_INVALID_IMAGE_SIZE"; break;
			case -41: errString = "CL_INVALID_SAMPLER"; break;
			case -42: errString = "CL_INVALID_BINARY"; break;
			case -43: errString = "CL_INVALID_BUILD_OPTIONS"; break;
			case -44: errString = "CL_INVALID_PROGRAM"; break;
			case -45: errString = "CL_INVALID_PROGRAM_EXECUTABLE"; break;
			case -46: errString = "CL_INVALID_KERNEL_NAME"; break;
			case -47: errString = "CL_INVALID_KERNEL_DEFINITION"; break;
			case -48: errString = "CL_INVALID_KERNEL"; break;
			case -49: errString = "CL_INVALID_ARG_INDEX"; break;
			case -50: errString = "CL_INVALID_ARG_VALUE"; break;
			case -51: errString = "CL_INVALID_ARG_SIZE"; break;
			case -52: errString = "CL_INVALID_KERNEL_ARGS"; break;
			case -53: errString = "CL_INVALID_WORK_DIMENSION"; break;
			case -54: errString = "CL_INVALID_WORK_GROUP_SIZE"; break;
			case -55: errString = "CL_INVALID_WORK_ITEM_SIZE"; break;
			case -56: errString = "CL_INVALID_GLOBAL_OFFSET"; break;
			case -57: errString = "CL_INVALID_EVENT_WAIT_LIST"; break;
			case -58: errString = "CL_INVALID_EVENT"; break;
			case -59: errString = "CL_INVALID_OPERATION"; break;
			case -60: errString = "CL_INVALID_GL_OBJECT"; break;
			case -61: errString = "CL_INVALID_BUFFER_SIZE"; break;
			case -62: errString = "CL_INVALID_MIP_LEVEL"; break;
			case -63: errString = "CL_INVALID_GLOBAL_WORK_SIZE"; break;
			case -64: errString = "CL_INVALID_PROPERTY"; break;
			case -65: errString = "CL_INVALID_IMAGE_DESCRIPTOR"; break;
			case -66: errString = "CL_INVALID_COMPILER_OPTIONS"; break;
			case -67: errString = "CL_INVALID_LINKER_OPTIONS"; break;
			case -68: errString = "CL_INVALID_DEVICE_PARTITION_COUNT"; break;
			case -1000: errString = "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR"; break;
			case -1001: errString = "CL_PLATFORM_NOT_FOUND_KHR"; break;
			case -1002: errString = "CL_INVALID_D3D10_DEVICE_KHR"; break;
			case -1003: errString = "CL_INVALID_D3D10_RESOURCE_KHR"; break;
			case -1004: errString = "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR"; break;
			case -1005: errString = "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR"; break;
			default: errString = "Unknown OpenCL error";
		}
		printf("OpenCL Error %d (%s), line %d\n", err, errString, line);
	}
}
