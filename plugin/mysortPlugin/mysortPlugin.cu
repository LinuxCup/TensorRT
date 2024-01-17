/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <cstring>
#include <vector>
#include <cub/cub.cuh>

#include "NvInfer.h"
// #include "common/bertCommon.h"
#include "common/common.cuh"
#include "common/serialize.hpp"
// #include "geluPlugin.h"
#include "mysortPlugin.h"

// using namespace nvinfer1;
#define TOTAL_NUM (152064)
#define kGPUBlockSize (512)

namespace nvinfer1
{
namespace plugin
{

__global__ void find_objects_kernel(int total_num, int valid_num)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int i = iy + ix;
    if (i > valid_num - 1) return;

}

template<typename T>
void swap(T *a, T*b)
{
    T t= *a;
    *a = *b;
    *b = t;
}


int partition(int *arr, float *coor_array,  int l, int h)
{
    float x = coor_array[h];
    int i = l - 1;

    for(int j = l; j<= h-1; j++)
    {
        if(coor_array[j] >= x)
        {
            i++;
            swap(&coor_array[i],&coor_array[j]);
            swap(&arr[i],&arr[j]);
        }
    }
    swap(&coor_array[i+1],&coor_array[h]);
    swap(&arr[i+1],&arr[h]);
    return (i+1);
}

void quickSortIterative(int *arr, float *coor_array, int l, int h)
{
    // create an auxiliary stack
    int stack[10000];

    // init top of stack
    int top = -1;

    // push init values of l and h to stack
    stack[++top] = l;
    stack[++top] = h;

    while(top>=0)
    {
         h = stack[top--];
         l = stack[top--];
         
         int p = partition(arr,coor_array,l,h);

         if(p-1>l)
         {
            stack[++top] = l;
            stack[++top] = p -1;
         }

         if(p+1 <h)
         {
            stack[++top] = p+1;
            stack[++top] = h;
         }
    }
}

inline double getTime(void) {
        const auto t = std::chrono::system_clock::now();
        const auto t_sec = std::chrono::duration_cast<std::chrono::duration<double>>(t.time_since_epoch());
        return (double) t_sec.count();
    }

pluginStatus_t Mysort::sort_inference(cudaStream_t stream, void const* const* inputs, void* const* outputs)
{
    double t0 = getTime();
    const void* src = inputs[0];
    const void* src_coor = inputs[1];
    const int* valid_num = (int*)inputs[2];
    void* ret = outputs[0];
    int valid_num_h;
    double t1 = getTime();
    cudaMemcpyAsync(
            &valid_num_h,
            valid_num,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream
        );
    std::cout << "sort valid num: " << valid_num_h << std::endl;
    double t2 = getTime();
    

    // typedef cub::BlockRadixSort<int, 128, 512> BlockRadixSort;
    // __shared__ typename BlockRadixSort::TempStorage storageSort;
    cudaMemcpyAsync(
            ret,
            src_coor,
            valid_num_h * sizeof(int),
            cudaMemcpyDeviceToDevice,
            stream
        );

    double t3 = getTime();
    float* src_h = new float[valid_num_h];
    int* ret_h = new int[valid_num_h];
    double t4 = getTime();
    
    cudaMemcpyAsync(
            src_h,
            src,
            valid_num_h * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );
    cudaMemcpyAsync(
            ret_h,
            ret,
            valid_num_h * sizeof(int),
            cudaMemcpyDeviceToHost,
            stream
        );

    double t5 = getTime();
    quickSortIterative(ret_h, src_h, 0, valid_num_h-1);
    double t6 = getTime();

    // for (int i = 0; i < 50; i++)
    // {
    //     std::cout << ret_h[i] << " " << src_h[i] << std::endl;
    // }

    cudaMemcpyAsync(
            ret,
            ret_h,
            valid_num_h * sizeof(int),
            cudaMemcpyHostToDevice,
            stream
        );
    free(src_h);
    free(ret_h);
    double t7 = getTime();
    
    // int grid_size = (TOTAL_NUM + kGPUBlockSize - 1) / kGPUBlockSize;
    // find_objects_kernel<<<grid_size, kGPUBlockSize>>>(TOTAL_NUM, valid_num);

    // std::cout << "t0:" << t1 - t0 << std::endl;
    // std::cout << "t1:" << t2 - t1 << std::endl;
    // std::cout << "t2:" << t3 - t2 << std::endl;
    // std::cout << "t3:" << t4 - t3 << std::endl;
    // std::cout << "t4:" << t5 - t4 << std::endl;
    // std::cout << "t5:" << t6 - t5 << std::endl;
    // std::cout << "t6:" << t7 - t6 << std::endl;
    // std::cout << "total:" << t7 - t0 << std::endl;

    return STATUS_SUCCESS;
}





} // namespace plugin
} // namespace nvinfer1

