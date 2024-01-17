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
#ifndef TRT_MYSORT_PLUGIN_H
#define TRT_MYSORT_PLUGIN_H
#include "NvInferPlugin.h"
#include "common/kernel.h"
#include "common/plugin.h"
#include <iostream>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{


pluginStatus_t sort_inference(cudaStream_t stream, void const* const* inputs, void* const* outputs);

class Mysort : public nvinfer1::IPluginV2DynamicExt
{
public:
    Mysort();
    Mysort(float negSlope);

    Mysort(void const* buffer, size_t length);

    // Mysort() = delete;

    ~Mysort() override = default;

    int32_t getNbOutputs() const noexcept override;


    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
    // Dims getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    // size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;
    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;

    // int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
    //     cudaStream_t stream) noexcept override;
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    void configureWithFormat(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType type, PluginFormat format, int32_t maxBatchSize) noexcept override;

    // bool supportsFormat(DataType type, PluginFormat format) const noexcept override;
    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    // IPluginV2* clone() const noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;
    


    void setPluginNamespace(char const* libNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    // IPluginV2Ext methods
    DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputType, int32_t nbInputs) const noexcept override;
    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;
    void detachFromContext() noexcept override;
    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    pluginStatus_t sort_inference(cudaStream_t stream, void const* const* inputs, void* const* outputs);



private:
    float mNegSlope;
    int32_t mBatchDim;
    
    std::string mPluginNamespace;
    std::string mNamespace;
};

class MysortPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    MysortPluginCreator();

    ~MysortPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;


private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};


} // namespace plugin
} // namespace nvinfer1

#endif
