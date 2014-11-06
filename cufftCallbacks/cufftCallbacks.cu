/**
* cufftCallbacks.cu
*
* A. Thall
* Alma College
* 5/28/2014
*
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cufftXt.h> 
#include <helper_cuda.h>
#include <helper_timer.h>

// Create ThreadsPerBlock constant
const int T_PER_B = 1024;
const int SIGNAL_SIZE = 65536;

// Complex data type
typedef cufftDoubleComplex dbComplex;
typedef cufftDoubleReal dbReal;
#define CUFFT_TYPEFORWARD CUFFT_D2Z
#define CUFFT_TYPEINVERSE CUFFT_Z2D
#define CUFFT_EXECFORWARD cufftExecD2Z
#define CUFFT_EXECINVERSE cufftExecZ2D

void callbackTest(int signalSize);
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id) );
	}
	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
		fprintf(stderr, "There is no device supporting CUDA\n");
	else
		fprintf(stderr, "Found %d CUDA Capable device(s)\n", deviceCount);

	int dev;
	for (dev = 0; dev < deviceCount; ++dev) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		fprintf(stderr, "\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	}
	fprintf(stderr, "\n and deviceID of max GFLOPS device is %d\n", gpuGetMaxGflopsDeviceId());
	fprintf(stderr, "but we're going to use device 0 by default.\n");
	cudaSetDevice(0);//gpuGetMaxGflopsDeviceId());

	printf("  NUM_BLOCKS = %d, T_PER_B = %d\n", SIGNAL_SIZE/T_PER_B, T_PER_B);

	callbackTest(SIGNAL_SIZE);

	cudaThreadExit();	
	exit(0);
}

/**
 * HERE BEGINS THE HOST AND KERNEL CODE TO SUPPORT THE APPLICATION
 *   NOTE:  some changed, moved to IrrBaseBalanced11.cu
 */

// Complex pointwise multiplication...divide by signal size to get normalized
static __global__ void dbcPointwiseSqr(dbComplex* cval, int size)
{
	dbComplex c, temp;
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < size) {
		temp = cval[tid];
		c.y = 2.0*temp.x*temp.y/SIGNAL_SIZE;
		//c.x = (temp.x + temp.y)*(temp.x - temp.y);  xxAT ??
		c.x = (temp.x*temp.x - temp.y*temp.y)/SIGNAL_SIZE;
		cval[tid] = c;
	}
} 


/** 
 * LET'S do the above with a cuFFT callback using the new CUDA6.5
 * callback protocols
 */

__device__ void dbcPointwiseSqrCB(void *dataOut, size_t offset,
									  dbComplex element, void *callerInfo,
									  void *sharedPointer) {
   dbComplex temp = element;
   element.y = 2.0*temp.x*temp.y/SIGNAL_SIZE;
   element.x = (temp.x*temp.x - temp.y*temp.y)/SIGNAL_SIZE;
   ((dbComplex *)dataOut)[offset] = element;
}

__device__ cufftCallbackStoreZ csquareCBptr = dbcPointwiseSqrCB;

__device__ dbComplex dbcPointwiseSqrLoadCB(void *dataIn, size_t offset,
										  void *callerInfo, void *sharedPointer) {
   dbComplex ret, element = ((dbComplex *) dataIn)[offset];
   ret.x = element.x*element.x - element.y*element.y;
   ret.y = 2.0*element.x*element.y;
   return ret;
}

__device__ cufftCallbackLoadZ csquareLOADCBptr = (cufftCallbackLoadZ) dbcPointwiseSqrLoadCB;

// load values of int array into double array for FFT.  Low-order 2 bytes go in lowest numbered
//     position in dArr
static __global__ void loadValue4ToFFTarray(double *dArr, int size) {

	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid < 2)
		dArr[tid] = 1.0;
	else
		dArr[tid] = 0.0;
}

/**
* callbackTest() -- test the cufftCallback functionality in CUDA 6.5 pre-release
*/
void callbackTest(int signalSize) {

	// We assume throughout that signalSize is divisible by T_PER_B
	const int numBlocks = signalSize/T_PER_B;
	const int numFFTblocks = (signalSize/2 + 1)/T_PER_B + 1;

	// Allocate device memory for signal
	dbReal *d_signal;
	dbComplex *z_signal;

	int d_size = sizeof(dbReal)*signalSize;
	int z_size = sizeof(dbComplex)*(signalSize/2 + 1);

	checkCudaErrors(cudaMallocManaged(&d_signal, d_size));
	checkCudaErrors(cudaMallocManaged(&z_signal, z_size));

	// allocate device memory for DWT weights and base values
	// CUFFT plan
	cufftHandle plan1, plan2;
	checkCudaErrors(cufftPlan1d(&plan1, signalSize, CUFFT_TYPEFORWARD, 1));
	checkCudaErrors(cufftPlan1d(&plan2, signalSize, CUFFT_TYPEINVERSE, 1));

	/** xxAT ** get callbackPtr for fftCallback squaring */

	cufftCallbackStoreZ hostCopyPtr;
   	checkCudaErrors(cudaMemcpyFromSymbol(&hostCopyPtr, csquareCBptr,
										 sizeof(hostCopyPtr)));
	cufftCallbackStoreZ pters[1];
	pters[0] = hostCopyPtr;
	fprintf(stderr, "The host pointer to the device function is %d\n", hostCopyPtr);
	fflush(stderr);
   	checkCudaErrors(cufftXtSetCallback(plan1, (void **) pters,
	   								   CUFFT_CB_ST_COMPLEX_DOUBLE, NULL));
   	/*
	cufftCallbackLoadZ hostCopyPtr;
   	checkCudaErrors(cudaMemcpyFromSymbol(&hostCopyPtr, csquareLOADCBptr,
										 sizeof(hostCopyPtr)));
	cufftCallbackLoadZ pters[1];
	pters[0] = hostCopyPtr;
	fprintf(stderr, "The host pointer to the device function is %d\n", hostCopyPtr);
	fflush(stderr);
	//   	checkCudaErrors(cufftXtSetCallback(plan2, (void **) pters,
	//   								   CUFFT_CB_LD_COMPLEX_DOUBLE, NULL));
	*/
	// load the int array to the doubles for FFT
	// This is already balanced, and already multiplied by a_0 = 1 for DWT
	loadValue4ToFFTarray<<<numBlocks, T_PER_B>>>(d_signal, signalSize);
	getLastCudaError("Kernel execution failed [ loadValue4ToFFTarray ]");

	cudaDeviceSynchronize();

	for (int i = 0; i < 20; i++)
		fprintf(stderr, "%f ", d_signal[i]);
	fprintf(stderr, "\n\n");
	fflush(stderr);

	fprintf(stderr, "starting loop\n");
	fflush(stderr);
	// Loop M-2 times

	for (int iter = 0; iter < 4; iter++) {

		// Transform signal
		checkCudaErrors(CUFFT_EXECFORWARD(plan1, (dbReal *)d_signal, (dbComplex *)z_signal));
		getLastCudaError("Kernel execution failed [ CUFFT_EXECFORWARD ]");
		cudaDeviceSynchronize();
		for (int z = 0; z < 20; z++)
			fprintf(stderr, "(%f,%f) ", z_signal[z].x, z_signal[z].y);
		fprintf(stderr, "\n\n");
		fflush(stderr);


		//fprintf(stderr, "Completed one forward fft at iteration %d\n", iter);
		// fflush(stderr);
		// Multiply the coefficients componentwise
   		//dbcPointwiseSqr<<<numFFTblocks, T_PER_B>>>(z_signal, signalSize/2 + 1);
   		//getLastCudaError("Kernel execution failed [ ComplexPointwiseSqr ]");

   		cudaDeviceSynchronize();
		// Transform signal back
		checkCudaErrors(CUFFT_EXECINVERSE(plan2, (dbComplex *)z_signal, (dbReal *)d_signal));
		getLastCudaError("Kernel execution failed [ CUFFT_EXECINVERSE ]");

		cudaDeviceSynchronize();
		for (int i = 0; i < 20; i++)
			fprintf(stderr, "%f ", d_signal[i]);
		fprintf(stderr, "\n\n");
		fflush(stderr);
	}

	fprintf(stderr, "\nTests completed.\n");

	//Destroy CUFFT context
	checkCudaErrors(cufftDestroy(plan1));
	checkCudaErrors(cufftDestroy(plan2));

	// cleanup memory
	checkCudaErrors(cudaFree(d_signal));
	checkCudaErrors(cudaFree(z_signal));
}
