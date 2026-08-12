/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockScheduler_, bool QuantOutput_>
class QuantMatmulNzSwigluWorkspace {
 public:
  using BlockMmad = BlockMmad_;
  using ArchTag = typename BlockMmad::ArchTag;
  using L1TileShape = typename BlockMmad::L1TileShape;
  using ElementA = typename BlockMmad::ElementA;
  using LayoutA = typename BlockMmad::LayoutA;
  using ElementB = typename BlockMmad::ElementB;
  using LayoutB = typename BlockMmad::LayoutB;
  using ElementC = typename BlockMmad::ElementC;
  using BlockScheduler = BlockScheduler_;
  static constexpr bool QuantOutput = QuantOutput_;

  struct Params {
    GemmCoord problem_shape;
    __gm__ ElementA* x;
    LayoutA layout_x;
    __gm__ ElementB* weight;
    LayoutB layout_weight;
    __gm__ float* scale;
    __gm__ int32_t* bias;
    __gm__ float* quant_scale;
    __gm__ float* quant_offset;
    GM_ADDR y;
    GM_ADDR workspace;

    CATLASS_DEVICE
    Params(GemmCoord problem_shape_, GM_ADDR x_, LayoutA layout_x_,
           GM_ADDR weight_, LayoutB layout_weight_, GM_ADDR scale_,
           GM_ADDR bias_, GM_ADDR quant_scale_, GM_ADDR quant_offset_,
           GM_ADDR y_, GM_ADDR workspace_)
        : problem_shape(problem_shape_),
          x(reinterpret_cast<__gm__ ElementA*>(x_)),
          layout_x(layout_x_),
          weight(reinterpret_cast<__gm__ ElementB*>(weight_)),
          layout_weight(layout_weight_),
          scale(reinterpret_cast<__gm__ float*>(scale_)),
          bias(reinterpret_cast<__gm__ int32_t*>(bias_)),
          quant_scale(reinterpret_cast<__gm__ float*>(quant_scale_)),
          quant_offset(reinterpret_cast<__gm__ float*>(quant_offset_)),
          y(y_),
          workspace(workspace_) {}
  };

  CATLASS_DEVICE
  QuantMatmulNzSwigluWorkspace()
      : aic_finish_store_(0),
        aiv_finish_compute_(1) {}

  template <int32_t CoreType = g_coreType>
  CATLASS_DEVICE void operator()(const Params& params);

  template <>
  CATLASS_DEVICE void operator()<AscendC::AIC>(const Params& params) {
    BlockScheduler scheduler;
    scheduler.Update(params.problem_shape,
                     MakeCoord(L1TileShape::M, L1TileShape::N));

    const uint32_t core_index = AscendC::GetBlockIdx();
    const uint32_t core_count = AscendC::GetBlockNum();
    const uint32_t core_loops = scheduler.GetCoreLoops();
    BlockMmad block_mmad(resource_);

    AscendC::GlobalTensor<ElementA> x;
    x.SetGlobalBuffer(params.x);
    AscendC::GlobalTensor<ElementB> weight;
    weight.SetGlobalBuffer(params.weight);
    weight.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    AscendC::GlobalTensor<ElementC> accumulator;
    accumulator.SetGlobalBuffer(
        reinterpret_cast<__gm__ ElementC*>(params.workspace));
    constexpr uint32_t kWorkspaceStages = 2;
    const auto accumulator_layout = layout::RowMajor{
        L1TileShape::M * core_count * kWorkspaceStages, L1TileShape::N};

    for (uint32_t loop = core_index; loop < core_loops;
         loop += core_count) {
      const GemmCoord block_coord = scheduler.GetBlockCoord(loop);
      const GemmCoord actual_shape =
          scheduler.GetActualBlockShape(block_coord);
      const MatrixCoord x_offset{block_coord.m() * L1TileShape::M,
                                 block_coord.k() * L1TileShape::K};
      const MatrixCoord weight_offset{block_coord.k() * L1TileShape::K,
                                      block_coord.n() * L1TileShape::N};
      const uint32_t workspace_stage = block_coord.n() / core_count;
      const MatrixCoord accumulator_offset{
          (workspace_stage * core_count + core_index) * L1TileShape::M, 0};
      const int64_t x_position = params.layout_x.GetOffset(x_offset);
      const int64_t weight_position =
          params.layout_weight.GetOffset(weight_offset);
      const int64_t accumulator_position =
          accumulator_layout.GetOffset(accumulator_offset);

      Callback callback_before_fixpipe{};
      Callback callback_after_fixpipe{};
      if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
        block_mmad(x[x_position], params.layout_x,
                   weight[weight_position], params.layout_weight,
                   accumulator[accumulator_position], accumulator_layout,
                   actual_shape, callback_before_fixpipe,
                   callback_after_fixpipe);
      } else {
        block_mmad(x[x_position], params.layout_x,
                   weight[weight_position], params.layout_weight,
                   accumulator[accumulator_position], accumulator_layout,
                   actual_shape);
        callback_after_fixpipe();
      }
    }

    if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
      block_mmad.SynchronizeBlock();
    }
    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(aic_finish_store_);
    Arch::CrossCoreWaitFlag(aiv_finish_compute_);
  }

  template <>
  CATLASS_DEVICE void operator()<AscendC::AIV>(const Params& params) {
    const uint32_t core_index =
        AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    Arch::CrossCoreWaitFlag(aic_finish_store_);

    constexpr uint32_t kGateCoreCount = 20;
    if (core_index < kGateCoreCount) {
      FusedEpilogue(params, core_index);
    }
    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(aiv_finish_compute_);
  }

 private:
  static constexpr uint32_t kBf16EpilogueTileN = 80;
  static constexpr uint32_t kQuantEpilogueFirstTileN = 64;
  static constexpr uint32_t kMaxEpilogueTileN = 96;
  static constexpr uint32_t kMaxEpilogueRows = 16;
  static constexpr uint32_t kMaxEpilogueElements =
      kMaxEpilogueRows * kMaxEpilogueTileN;
  static constexpr uint32_t kGateSize = 3200;
  static constexpr uint32_t kGateCoreCount = 20;

  CATLASS_DEVICE void FusedEpilogue(const Params& params,
                                    uint32_t core_index) {
    const uint32_t subblock_index = AscendC::GetSubBlockIdx();
    const uint32_t tile_offset = QuantOutput
        ? subblock_index * kQuantEpilogueFirstTileN
        : subblock_index * kBf16EpilogueTileN;
    if (tile_offset >= L1TileShape::N) {
      return;
    }
    const uint32_t remaining_columns = L1TileShape::N - tile_offset;
    const uint32_t requested_tile_size = QuantOutput
        ? (subblock_index == 0 ? kQuantEpilogueFirstTileN
                               : kMaxEpilogueTileN)
        : kBf16EpilogueTileN;
    const uint32_t tile_size = remaining_columns < requested_tile_size
                                   ? remaining_columns
                                   : requested_tile_size;
    const uint32_t column_offset =
        core_index * L1TileShape::N + tile_offset;
    const uint32_t core_count = AscendC::GetBlockNum();
    constexpr uint32_t kWorkspaceStages = 2;
    const auto accumulator_layout = layout::RowMajor{
        L1TileShape::M * core_count * kWorkspaceStages, L1TileShape::N};
    const MatrixCoord gate_offset{core_index * L1TileShape::M, 0};
    const MatrixCoord up_offset{
        (core_count + core_index) * L1TileShape::M, 0};

    AscendC::GlobalTensor<int32_t> accumulator;
    accumulator.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(params.workspace));
    const int64_t gate_position = accumulator_layout.GetOffset(gate_offset);
    const int64_t up_position = accumulator_layout.GetOffset(up_offset);
    auto gate_accumulator = accumulator[gate_position];
    auto up_accumulator = accumulator[up_position];
    AscendC::GlobalTensor<int32_t> bias;
    bias.SetGlobalBuffer(params.bias);
    AscendC::GlobalTensor<float> scale;
    scale.SetGlobalBuffer(params.scale);
    AscendC::GlobalTensor<bfloat16_t> bf16_output;
    AscendC::GlobalTensor<int8_t> quant_output;
    if constexpr (QuantOutput) {
      quant_output.SetGlobalBuffer(
          reinterpret_cast<__gm__ int8_t*>(params.y));
    } else {
      bf16_output.SetGlobalBuffer(
          reinterpret_cast<__gm__ bfloat16_t*>(params.y));
    }

    size_t ub_offset = 0;
    auto gate_int = resource_.ubBuf.template GetBufferByByte<int32_t>(ub_offset);
    ub_offset += kMaxEpilogueElements * sizeof(int32_t);
    auto up_int = resource_.ubBuf.template GetBufferByByte<int32_t>(ub_offset);
    ub_offset += kMaxEpilogueElements * sizeof(int32_t);
    auto gate_bias = resource_.ubBuf.template GetBufferByByte<int32_t>(ub_offset);
    ub_offset += kMaxEpilogueTileN * sizeof(int32_t);
    auto up_bias = resource_.ubBuf.template GetBufferByByte<int32_t>(ub_offset);
    ub_offset += kMaxEpilogueTileN * sizeof(int32_t);
    auto gate_scale = resource_.ubBuf.template GetBufferByByte<float>(ub_offset);
    ub_offset += kMaxEpilogueTileN * sizeof(float);
    auto up_scale = resource_.ubBuf.template GetBufferByByte<float>(ub_offset);
    ub_offset += kMaxEpilogueTileN * sizeof(float);
    auto gate_float = resource_.ubBuf.template GetBufferByByte<float>(ub_offset);
    ub_offset += kMaxEpilogueElements * sizeof(float);
    auto up_float = resource_.ubBuf.template GetBufferByByte<float>(ub_offset);
    ub_offset += kMaxEpilogueElements * sizeof(float);
    auto temporary = resource_.ubBuf.template GetBufferByByte<float>(ub_offset);
    ub_offset += kMaxEpilogueElements * sizeof(float);
    auto gate_bf16 =
        resource_.ubBuf.template GetBufferByByte<bfloat16_t>(ub_offset);
    ub_offset += kMaxEpilogueElements * sizeof(bfloat16_t);
    auto up_bf16 =
        resource_.ubBuf.template GetBufferByByte<bfloat16_t>(ub_offset);
    ub_offset += kMaxEpilogueElements * sizeof(bfloat16_t);
    auto output_bf16 =
        resource_.ubBuf.template GetBufferByByte<bfloat16_t>(ub_offset);
    ub_offset += kMaxEpilogueElements * sizeof(bfloat16_t);
    auto output_fp16 =
        resource_.ubBuf.template GetBufferByByte<half>(ub_offset);
    ub_offset += kMaxEpilogueElements * sizeof(half);
    auto output_int8 =
        resource_.ubBuf.template GetBufferByByte<int8_t>(ub_offset);

    float quant_multiplier = 1.0f;
    float quant_addend = 0.0f;
    if constexpr (QuantOutput) {
      quant_multiplier = 1.0f / params.quant_scale[0];
      quant_addend = params.quant_offset[0];
    }

    const uint32_t row_count = params.problem_shape.m();
    const uint32_t element_count = row_count * tile_size;
    constexpr uint32_t kDataBlockBytes = 32;
    constexpr uint32_t kMin2dBurstRows = 2;
    constexpr uint32_t kVectorRepeatElements = 64;
    const bool use_2d_burst = row_count >= kMin2dBurstRows;
    const uint16_t accumulator_burst_length = static_cast<uint16_t>(
        tile_size * sizeof(int32_t) / kDataBlockBytes);
    const uint16_t accumulator_source_gap = static_cast<uint16_t>(
        (L1TileShape::N - tile_size) * sizeof(int32_t) / kDataBlockBytes);
    AscendC::DataCopy(gate_bias, bias[column_offset], tile_size);
    AscendC::DataCopy(up_bias, bias[kGateSize + column_offset], tile_size);
    AscendC::DataCopy(gate_scale, scale[column_offset], tile_size);
    AscendC::DataCopy(up_scale, scale[kGateSize + column_offset], tile_size);
    if (use_2d_burst) {
      const AscendC::DataCopyParams accumulator_copy_params{
          static_cast<uint16_t>(row_count), accumulator_burst_length,
          accumulator_source_gap, 0};
      AscendC::DataCopy(gate_int, gate_accumulator[tile_offset],
                        accumulator_copy_params);
      AscendC::DataCopy(up_int, up_accumulator[tile_offset],
                        accumulator_copy_params);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

      const uint8_t row_stride =
          static_cast<uint8_t>(accumulator_burst_length);
      const AscendC::BinaryRepeatParams broadcast_params{
          1, 1, 1, row_stride, row_stride, 0};
      AscendC::Add(gate_int, gate_int, gate_bias, kVectorRepeatElements,
                   static_cast<uint8_t>(row_count), broadcast_params);
      AscendC::Add(up_int, up_int, up_bias, kVectorRepeatElements,
                   static_cast<uint8_t>(row_count), broadcast_params);
      if (tile_size > kVectorRepeatElements) {
        const uint32_t tail_size = tile_size - kVectorRepeatElements;
        AscendC::Add(gate_int[kVectorRepeatElements],
                     gate_int[kVectorRepeatElements],
                     gate_bias[kVectorRepeatElements], tail_size,
                     static_cast<uint8_t>(row_count), broadcast_params);
        AscendC::Add(up_int[kVectorRepeatElements],
                     up_int[kVectorRepeatElements],
                     up_bias[kVectorRepeatElements], tail_size,
                     static_cast<uint8_t>(row_count), broadcast_params);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(gate_float, gate_int,
                    AscendC::RoundMode::CAST_RINT, element_count);
      AscendC::Cast(up_float, up_int,
                    AscendC::RoundMode::CAST_RINT, element_count);
      AscendC::PipeBarrier<PIPE_V>();

      AscendC::Mul(gate_float, gate_float, gate_scale,
                   kVectorRepeatElements, static_cast<uint8_t>(row_count),
                   broadcast_params);
      AscendC::Mul(up_float, up_float, up_scale, kVectorRepeatElements,
                   static_cast<uint8_t>(row_count), broadcast_params);
      if (tile_size > kVectorRepeatElements) {
        const uint32_t tail_size = tile_size - kVectorRepeatElements;
        AscendC::Mul(gate_float[kVectorRepeatElements],
                     gate_float[kVectorRepeatElements],
                     gate_scale[kVectorRepeatElements], tail_size,
                     static_cast<uint8_t>(row_count), broadcast_params);
        AscendC::Mul(up_float[kVectorRepeatElements],
                     up_float[kVectorRepeatElements],
                     up_scale[kVectorRepeatElements], tail_size,
                     static_cast<uint8_t>(row_count), broadcast_params);
      }
      AscendC::PipeBarrier<PIPE_V>();
    } else {
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
      for (uint32_t row = 0; row < row_count; ++row) {
        const uint32_t row_offset = row * tile_size;
        auto gate_float_row = gate_float[row_offset];
        auto up_float_row = up_float[row_offset];
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::DataCopy(
            gate_int,
            gate_accumulator[row * L1TileShape::N + tile_offset],
            tile_size);
        AscendC::DataCopy(
            up_int,
            up_accumulator[row * L1TileShape::N + tile_offset],
            tile_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::Add(gate_int, gate_int, gate_bias, tile_size);
        AscendC::Add(up_int, up_int, up_bias, tile_size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(gate_float_row, gate_int,
                      AscendC::RoundMode::CAST_RINT, tile_size);
        AscendC::Cast(up_float_row, up_int,
                      AscendC::RoundMode::CAST_RINT, tile_size);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(gate_float_row, gate_float_row, gate_scale, tile_size);
        AscendC::Mul(up_float_row, up_float_row, up_scale, tile_size);
        AscendC::PipeBarrier<PIPE_V>();
      }
    }

    if constexpr (!QuantOutput) {
      // Preserve the public BF16 path's QuantMatmul -> SwiGLU boundary.
      AscendC::Cast(gate_bf16, gate_float,
                    AscendC::RoundMode::CAST_RINT, element_count);
      AscendC::Cast(up_bf16, up_float,
                    AscendC::RoundMode::CAST_RINT, element_count);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(gate_float, gate_bf16,
                    AscendC::RoundMode::CAST_NONE, element_count);
      AscendC::Cast(up_float, up_bf16,
                    AscendC::RoundMode::CAST_NONE, element_count);
      AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::Muls(temporary, gate_float, -1.0f, element_count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(temporary, temporary, element_count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(temporary, temporary, 1.0f, element_count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(gate_float, gate_float, temporary, element_count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(gate_float, gate_float, up_float, element_count);
    AscendC::PipeBarrier<PIPE_V>();

    if constexpr (QuantOutput) {
      // Match DequantSwigluQuant: keep dequant and SwiGLU in FP32, then use
      // round(x / scale) + offset with INT8 saturation.
      AscendC::Muls(gate_float, gate_float, quant_multiplier, element_count);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Adds(gate_float, gate_float, quant_addend, element_count);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(output_fp16, gate_float,
                    AscendC::RoundMode::CAST_ODD, element_count);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(output_int8, output_fp16,
                    AscendC::RoundMode::CAST_RINT, element_count);
    } else {
      AscendC::Cast(output_bf16, gate_float,
                    AscendC::RoundMode::CAST_RINT, element_count);
      AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    if (use_2d_burst) {
      if constexpr (QuantOutput) {
        const uint16_t output_burst_length = static_cast<uint16_t>(
            tile_size * sizeof(int8_t) / kDataBlockBytes);
        const uint16_t output_destination_gap = static_cast<uint16_t>(
            (kGateSize - tile_size) * sizeof(int8_t) / kDataBlockBytes);
        AscendC::DataCopy(
            quant_output[column_offset], output_int8,
            AscendC::DataCopyParams{static_cast<uint16_t>(row_count),
                                    output_burst_length, 0,
                                    output_destination_gap});
      } else {
        const uint16_t output_burst_length = static_cast<uint16_t>(
            tile_size * sizeof(bfloat16_t) / kDataBlockBytes);
        const uint16_t output_destination_gap = static_cast<uint16_t>(
            (kGateSize - tile_size) * sizeof(bfloat16_t) / kDataBlockBytes);
        AscendC::DataCopy(
            bf16_output[column_offset], output_bf16,
            AscendC::DataCopyParams{static_cast<uint16_t>(row_count),
                                    output_burst_length, 0,
                                    output_destination_gap});
      }
    } else {
      for (uint32_t row = 0; row < row_count; ++row) {
        const uint32_t row_offset = row * tile_size;
        if constexpr (QuantOutput) {
          AscendC::DataCopy(
              quant_output[row * kGateSize + column_offset],
              output_int8[row_offset], tile_size);
        } else {
          AscendC::DataCopy(
              bf16_output[row * kGateSize + column_offset],
              output_bf16[row_offset], tile_size);
        }
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    if (!use_2d_burst) {
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    }
  }

  Arch::CrossCoreFlag aic_finish_store_;
  Arch::CrossCoreFlag aiv_finish_compute_;
  Arch::Resource<ArchTag> resource_;

  CATLASS_DEVICE static float Bf16ToFloat(bfloat16_t value) {
    uint16_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    const uint32_t float_bits = static_cast<uint32_t>(bits) << 16;
    float result;
    __builtin_memcpy(&result, &float_bits, sizeof(result));
    return result;
  }

};

}  // namespace Catlass::Gemm::Kernel
