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
#include "mysortPlugin.h"
#include "common/checkMacrosPlugin.h"
#include "common/kernel.h"


using namespace nvinfer1::plugin;

namespace nvinfer1
{
namespace plugin
{
static char const* const kMysort_PLUGIN_VERSION{"1"};
static char const* const kMysort_PLUGIN_NAME{"MysortPlugin"};
PluginFieldCollection MysortPluginCreator::mFC{};
std::vector<PluginField> MysortPluginCreator::mPluginAttributes;

// LeakyReLU {{{
Mysort::Mysort()
{
;
}

Mysort::Mysort(float negSlope)
    : mNegSlope(negSlope)
    , mBatchDim(1)
{
    PLUGIN_VALIDATE(negSlope >= 0.0F);
}

Mysort::Mysort(void const* buffer, size_t length)
{
    char const *d = reinterpret_cast<char const*>(buffer), *a = d;
    mNegSlope = read<float>(d);
    mBatchDim = read<int32_t>(d);
    PLUGIN_VALIDATE(d == a + length);
}

int32_t Mysort::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs Mysort::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // std::cout << "_________getOutputDimensions " << inputs[0].nbDims << " " << inputs[1].nbDims << " " << inputs[2].nbDims << std::endl;
    try
    {
        PLUGIN_ASSERT(nbInputs == 3);
        PLUGIN_ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());

        // Shape of boxes input should be
        // Constant shape: [batch_size, num_boxes, num_classes, 4] or [batch_size, num_boxes, 1, 4]
        //           shareLocation ==              0               or          1
        // or
        // Dynamic shape: some dimension values may be -1
        PLUGIN_ASSERT(inputs[0].nbDims == 2);

        // Shape of scores input should be
        // Constant shape: [batch_size, num_boxes, num_classes] or [batch_size, num_boxes, num_classes, 1]
        // or
        // Dynamic shape: some dimension values may be -1
        PLUGIN_ASSERT(inputs[1].nbDims == 2 || inputs[1].nbDims == 2);

        nvinfer1::DimsExprs output(inputs[0]);
        return output;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}


void Mysort::setPluginNamespace(char const* pluginNamespace) noexcept
{
    std::cout << "setPluginNamespace " << pluginNamespace  << std::endl;
    mPluginNamespace = pluginNamespace;
}


char const* Mysort::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}


// Return the DataType of the plugin output at the requested index.
DataType Mysort::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    // std::cout << "___________getOutputDataType " << (int)inputTypes[0] << " " << (int)inputTypes[1] << " " << (int)inputTypes[2] << std::endl;
    // Two outputs
    // PLUGIN_ASSERT(index == 0);
    // PLUGIN_ASSERT(inputTypes[0] == inputTypes[1]);
    // // topDetections
    // if (index == 0)
    // {
    //     return inputTypes[0];
    // }
    // // keepCount: use kFLOAT instead as they have same sizeof(type)
    // PLUGIN_ASSERT(sizeof(int32_t) == sizeof(float));
    return inputTypes[1];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Mysort::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
    std::cout << "attachToContext"  << std::endl;
}



void Mysort::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    // std::cout << "configurePlugin"  << std::endl;
    // try
    // {
    //     PLUGIN_ASSERT(nbInputs == 2);
    //     PLUGIN_ASSERT(nbOutputs == 4);

    //     // Shape of boxes input should be
    //     // Constant shape: [batch_size, num_boxes, num_classes, 4] or [batch_size, num_boxes, 1, 4]
    //     //           shareLocation ==              0               or          1
    //     const int32_t numLocClasses = param.shareLocation ? 1 : param.numClasses;
    //     PLUGIN_ASSERT(in[0].desc.dims.nbDims == 4);
    //     PLUGIN_ASSERT(in[0].desc.dims.d[2] == numLocClasses);
    //     PLUGIN_ASSERT(in[0].desc.dims.d[3] == 4);

    //     // Shape of scores input should be
    //     // Constant shape: [batch_size, num_boxes, num_classes] or [batch_size, num_boxes, num_classes, 1]
    //     PLUGIN_ASSERT(in[1].desc.dims.nbDims == 3 || (in[1].desc.dims.nbDims == 4 && in[1].desc.dims.d[3] == 1));

    //     mBoxesSize = in[0].desc.dims.d[1] * in[0].desc.dims.d[2] * in[0].desc.dims.d[3];
    //     mScoresSize = in[1].desc.dims.d[1] * in[1].desc.dims.d[2];
    //     // num_boxes
    //     mNumPriors = in[0].desc.dims.d[1];

    //     mPrecision = in[0].desc.type;
    // }
    // catch (std::exception const& e)
    // {
    //     caughtError(e);
    // }
}


// Detach the plugin object from its execution context.
void Mysort::detachFromContext() noexcept {}


int32_t Mysort::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // std::cout << "__________Mysort::enqueue______________" << std::endl;

    try
    {

        pluginStatus_t status = sort_inference(stream, inputs, outputs);
        return status;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

size_t Mysort::getSerializationSize() const noexcept
{
    // mNegSlope, mBatchDim
    // std::cout << "getSerializationSize"  << std::endl;
    return sizeof(float) + sizeof(int32_t);
}

void Mysort::serialize(void* buffer) const noexcept
{
    // std::cout << "serialize"  << std::endl;
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mNegSlope);
    write(d, mBatchDim);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void Mysort::configureWithFormat(Dims const* inputDims, int32_t /* nbInputs */, Dims const* /* outputDims */,
    int32_t nbOutputs, DataType type, PluginFormat format, int32_t) noexcept
{
    // std::cout << "configureWithFormat"  << std::endl;
    PLUGIN_ASSERT(type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
    PLUGIN_ASSERT(mBatchDim == 1);
    PLUGIN_ASSERT(nbOutputs == 1);
    for (int32_t i = 0; i < inputDims[0].nbDims; ++i)
    {
        mBatchDim *= inputDims[0].d[i];
    }
    std::cout << "configureWithFormat done"  << std::endl;
}


bool Mysort::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // std::cout << "3_______________" << pos << " " << nbInputs << " " << nbOutputs << std::endl;
    PLUGIN_ASSERT(nbInputs <= 3 && nbInputs >= 0);
    PLUGIN_ASSERT(nbOutputs <= 1 && nbOutputs >= 0);
    PLUGIN_ASSERT(pos < 5 && pos >= 0);
    auto const* in = inOut;
    auto const* out = inOut + nbInputs;
    bool const consistentFloatPrecision = in[0].type == in[pos].type;

    switch (pos)
    {
    case 0:
        // std::cout << ""
        return (in[0].type == DataType::kHALF || in[0].type == DataType::kFLOAT)
            && in[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 1:
        return (in[1].type == DataType::kHALF || in[1].type == DataType::kINT32)
            && in[1].format == PluginFormat::kLINEAR;
    case 2: 
        return (in[2].type == DataType::kHALF || in[2].type == DataType::kINT32)
            && in[2].format == PluginFormat::kLINEAR;
    case 3: 
        return (out[0].type == DataType::kHALF || out[0].type == DataType::kINT32)
            && out[0].format == PluginFormat::kLINEAR;
    }
    return false;
}

int32_t Mysort::initialize() noexcept
{
    return 0;
}

void Mysort::terminate() noexcept {}


size_t Mysort::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    int32_t batchSize = inputs[0].dims.d[0];
    int32_t intput0 = inputs[0].dims.d[1];
    int32_t intput1 = inputs[1].dims.d[1];
    int32_t intput2 = inputs[2].dims.d[1];
    int32_t output0 = inputs[0].dims.d[1];

    size_t wss[4];
    wss[0] = batchSize * intput0 * sizeof(float);
    wss[1] = batchSize * intput1 * sizeof(int32_t);
    wss[2] = batchSize * intput2 * sizeof(int32_t);
    wss[3] = batchSize * output0 * sizeof(int32_t);
    return calculateTotalWorkspaceSize(wss, 4);

    // std::cout << "nbInputs:" << nbInputs << " " << nbOutputs << std::endl;
    // std::cout << "___________getWorkspaceSize " << inputs[0].dims.d[0] << " " << inputs[1].dims.d[0] << " " << inputs[2].dims.d[0] << " " << outputs[0].dims.d[0] << std::endl;
    // std::cout << "___________getWorkspaceSize " << inputs[0].dims.d[1] << " " << inputs[1].dims.d[1] << " " << inputs[2].dims.d[1] << " " << outputs[0].dims.d[1] << std::endl;
    // return 0;
}

char const* Mysort::getPluginType() const noexcept
{
    return kMysort_PLUGIN_NAME;
}

char const* Mysort::getPluginVersion() const noexcept
{
    return kMysort_PLUGIN_VERSION;
}

void Mysort::destroy() noexcept
{
    delete this;
}

IPluginV2DynamicExt* Mysort::clone() const noexcept
{
    try
    {
        IPluginV2DynamicExt* plugin = new Mysort(mNegSlope);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}


MysortPluginCreator::MysortPluginCreator()
{
    // mPluginAttributes.clear();
    // mPluginAttributes.emplace_back(PluginField("number", nullptr, PluginFieldType::kFLOAT32, 1));

    // std::cout << "5_______________" << mPluginAttributes.size() << std::endl;
    // mFC.nbFields = mPluginAttributes.size();
    // mFC.fields = mPluginAttributes.data();
}

char const* MysortPluginCreator::getPluginName() const noexcept
{
    return kMysort_PLUGIN_NAME;
}

char const* MysortPluginCreator::getPluginVersion() const noexcept
{
    return kMysort_PLUGIN_VERSION;
}

PluginFieldCollection const* MysortPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* MysortPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    // std::cout << "4_______________" << fc->nbFields << std::endl;
    try
    {
        // PluginField const* fields = fc->fields;
        // PLUGIN_VALIDATE(fc->nbFields == 1);
        // PLUGIN_VALIDATE(fields[0].type == PluginFieldType::kFLOAT32);
        // PLUGIN_VALIDATE(!strcmp(fields[0].name, "negSlope"));
        // float negSlope = *(static_cast<float const*>(fields[0].data));

        // return new Mysort(negSlope);
        return new Mysort();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* MysortPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call NMS::destroy()
        auto* plugin = new Mysort(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
} // namespace plugin
} // namespace nvinfer1
// LeakReLU }}}
