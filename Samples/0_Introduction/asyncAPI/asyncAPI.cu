/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample illustrates the usage of CUDA events for both GPU timing and
 * overlapping CPU and GPU execution.  Events are inserted into a stream
 * of CUDA calls.  Since CUDA stream calls are asynchronous, the CPU can
 * perform computations while GPU is executing (including DMA memcopies
 * between the host and device).  CPU can query CUDA events to determine
 * whether GPU has completed tasks.
 */

// includes, system
#include <stdio.h>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions

// 将传入的整数数组 g_data 中的每个元素增加 inc_value
__global__ void increment_kernel(int *g_data, int inc_value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // [0,32767] * 512 + [0,511]
	g_data[idx] = g_data[idx] + inc_value;
}

// 用于检查结果是否正确。遍历数组data，并检查每个元素是否等于期望值x
bool correct_output(int *data, const int n, const int x)
{
	for (int i = 0; i < n; i++)
		if (data[i] != x)
		{
			printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
			return false;
		}

	return true;
}

int main(int argc, char *argv[])
{
	int devID;
	cudaDeviceProp deviceProps;
	printf("[%s] - Starting...\n", argv[0]);

	// 根据当前环境和设备状态选择最适合的CUDA设备
	devID = findCudaDevice(argc, (const char **)argv);

	// 获取当前id对应的设备属性中的name
	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
	printf("CUDA device [%s]\n", deviceProps.name);
	// n表示数组的元素个数
	int n = 16 * 1024 * 1024;	  // 2 ^ 24  idx ： [0 , (2 ^ 24-1)]
	// nbytes表示数组的空间大小
	int nbytes = n * sizeof(int); // 2 ^ 26
	int value = 26;

	// 分配主机内存
	int *a = 0;
	checkCudaErrors(cudaMallocHost((void **)&a, nbytes));
	memset(a, 0, nbytes);

	// 分配设备内存
	int *d_a = 0;
	checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
	checkCudaErrors(cudaMemset(d_a, 255, nbytes));

	// 设置内核配置
	dim3 threads = dim3(512, 1);  // 512 threads per block
	dim3 blocks = dim3(n / threads.x, 1); // 32768 blocks per grid

	// 创建CUDA事件句柄
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	// 计时器，用于测量CPU时间
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	// 同步设备
	checkCudaErrors(cudaDeviceSynchronize());
	float gpu_time = 0.0f;

	// 开始计时并启动CUDA任务
	checkCudaErrors(cudaProfilerStart()); // 启动 CUDA Profiler。这是用于性能分析的工具
	sdkStartTimer(&timer); // CPU 发出指令的时间，但这些指令是异步的，不代表 GPU 的实际执行时间
	cudaEventRecord(start, 0); // 记录一个 CUDA 事件，并将该事件插入到指定的 CUDA 流（此处是默认的流 0）中
	// 异步执行数据拷贝和内核调用
	cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
	increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
	cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);

	// 停止事件计时
	cudaEventRecord(stop, 0);
	sdkStopTimer(&timer); // 停止 CPU 端的计时器，记录 CPU 端的时间
	checkCudaErrors(cudaProfilerStop());

	// CPU在等待GPU完成时进行一些工作
	unsigned long int counter = 0;
	while (cudaEventQuery(stop) == cudaErrorNotReady)
	{
		counter++;
	}
	
	// 获取GPU执行时间
	checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

	// 打印CPU和GPU时间
	printf("time spent executing by the GPU: %.2f\n", gpu_time);
	printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer)); //  表示从 CPU 启动计时段内的 CUDA 操作（包括数据传输和内核执行）到 CPU 完成所有 CUDA 调用的时间
	printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

	// 验证结果
	bool bFinalResults = correct_output(a, n, value);

	// 释放资源
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaFreeHost(a));
	checkCudaErrors(cudaFree(d_a));

	exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}
