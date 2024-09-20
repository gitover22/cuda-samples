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

#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <helper_cuda.h>

/////////////////////////////////////////////////////////////////
// Some utility code to define grid_stride_range
// Normally this would be in a header but it's here
// for didactic purposes. Uses
#include "range.hpp"
using namespace util::lang;

// type alias to simplify typing...
template <typename T>
using step_range = typename range_proxy<T>::step_range_proxy;


/**
 * @brief 生成一个基于grid-stride的范围对象
 * 这个函数在设备端执行，计算线程的起始索引，并返回步长为blockDim.x * gridDim.x的范围
 * 这种范围的好处是，允许线程跨越整个网格迭代处理数据，而不仅限于当前block
 * 
 * @param begin 起始值
 * @param end 结束值
 * @return step_range<T> 一个范围对象，表示线程的索引步进
 */
template <typename T>
__device__ step_range<T> grid_stride_range(T begin, T end) {
  begin += blockDim.x * blockIdx.x + threadIdx.x; // 计算每个线程的起始索引 256* [0 - 7] + [0 - 255], begin: [0 - 2047]
  return range(begin, end).step(gridDim.x * blockDim.x); // 为这个范围设置了一个步长，步长的大小是 gridDim.x * blockDim.x,即 8 * 256
}
/////////////////////////////////////////////////////////////////
/**
 * @brief 设备端的count_if函数，用于根据给定的谓词函数(p)来统计数组中符合条件的元素个数
 * 
 * @param count 计数器指针，用于存储符合条件的元素个数
 * @param data 要统计的数组
 * @param n 数组的长度
 * @param p 谓词函数，判断元素是否符合条件
 */
template <typename T, typename Predicate>
__device__ void count_if(int *count, T *data, int n, Predicate p) {
  for (auto i : grid_stride_range(0, n)) {
    // 如果元素满足条件，使用原子操作累加计数器
    if (p(data[i])) atomicAdd(count, 1);
  }
}

/**
 * @brief 使用count_if函数，统计文本中字符'x', 'y', 'z', 'w'出现的次数
 * 使用lambda函数来定义谓词
 * 
 * @param count 存储计数结果的设备内存指针
 * @param text 文本数据
 * @param n 文本长度
 */
__global__ void xyzw_frequency(int *count, char *text, int n) {
  const char letters[]{'x', 'y', 'z', 'w'};

  count_if(count, text, n, [&](char c) {
    for (const auto x : letters)
      if (c == x) return true;
    return false;
  });
}


/**
 * @brief 使用Thrust库的count_if函数在设备端统计字符'x', 'y', 'z', 'w'的出现次数
 * Thrust库简化了在CUDA中的并行编程，减少了对手动定义内核的需求
 * 
 * @param count 存储计数结果的设备内存指针
 * @param text 文本数据
 * @param n 文本长度
 */
__global__ void xyzw_frequency_thrust_device(int *count, char *text, int n) {
  const char letters[]{'x', 'y', 'z', 'w'};
  *count = thrust::count_if(thrust::device, text, text + n, [=](char c) {
    for (const auto x : letters)
      if (c == x) return true;
    return false;
  });
}

// a bug in Thrust 1.8 causes warnings when this is uncommented
// so commented out by default -- fixed in Thrust master branch
#if 0 
void xyzw_frequency_thrust_host(int *count, char *text, int n)
{
  const char letters[] { 'x','y','z','w' };
  *count = thrust::count_if(thrust::host, text, text+n, [&](char c) {
    for (const auto x : letters) 
      if (c == x) return true;
    return false;
  });
}
#endif

int main(int argc, char **argv) {
  const char *filename = sdkFindFilePath("warandpeace.txt", argv[0]); // 查找名为 "warandpeace.txt" 的文件，并返回其路径

  // 分配16MB的主机内存，用于存储文本数据
  int numBytes = 16 * 1048576;
  char *h_text = (char *)malloc(numBytes);
  // 查找并初始化CUDA设备
  int devID = findCudaDevice(argc, (const char **)argv);

  // 分配设备内存，用于存储文本数据
  char *d_text;
  checkCudaErrors(cudaMalloc((void **)&d_text, numBytes));

  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Cannot find the input text file\n. Exiting..\n");
    return EXIT_FAILURE;
  }
  int len = (int)fread(h_text, sizeof(char), numBytes, fp); // 从文件中读取数据到主机端缓存区
  fclose(fp);
  std::cout << "Read " << len << " byte corpus from " << filename << std::endl;

  checkCudaErrors(cudaMemcpy(d_text, h_text, len, cudaMemcpyHostToDevice)); // copy到设备缓冲区

  int count = 0;
  int *d_count; // 设备端计数器
  checkCudaErrors(cudaMalloc(&d_count, sizeof(int)));
  checkCudaErrors(cudaMemset(d_count, 0, sizeof(int)));

  // Try uncommenting one kernel call at a time
  xyzw_frequency<<<8, 256>>>(d_count, d_text, len); // 使用了手写的CUDA内核
  xyzw_frequency_thrust_device<<<1, 1>>>(d_count, d_text, len); // 使用了 Thrust 库，它提供了更高级的并行编程接口，通过 thrust::count_if 来完成同样的任务
  checkCudaErrors(
      cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

  // xyzw_frequency_thrust_host(&count, h_text, len);

  std::cout << "counted " << count
            << " instances of 'x', 'y', 'z', or 'w' in \"" << filename << "\""
            << std::endl;

  checkCudaErrors(cudaFree(d_count));
  checkCudaErrors(cudaFree(d_text));

  return EXIT_SUCCESS;
}
