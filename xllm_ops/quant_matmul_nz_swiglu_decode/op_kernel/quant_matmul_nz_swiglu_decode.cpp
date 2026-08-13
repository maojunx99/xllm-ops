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

#include "kernel_operator.h"

#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "quant_matmul_nz_swiglu_workspace.hpp"

using namespace Catlass;

template <bool QuantOutput>
__aicore__ inline void RunQuantMatmulNzSwigluDecode(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR bias,
    GM_ADDR quant_scale, GM_ADDR quant_offset, GM_ADDR y,
    GM_ADDR workspace, uint32_t m, uint32_t k, uint32_t n) {
  using L1TileShape = GemmShape<16, 160, QuantOutput ? 1024 : 512>;
  using L0TileShape = GemmShape<16, 128, 256>;
  using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsyncWithCallback<
      1, 2, 2, 2, 1, false, QuantOutput>;
  using AType = Gemm::GemmType<int8_t, layout::RowMajor>;
  using BType = Gemm::GemmType<int8_t, layout::zN>;
  using CType = Gemm::GemmType<int32_t, layout::RowMajor>;
  using BlockMmad = Gemm::Block::BlockMmad<
      DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
  using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
  using MatmulKernel = Gemm::Kernel::QuantMatmulNzSwigluWorkspace<
      BlockMmad, BlockScheduler, QuantOutput>;

  GemmCoord problem_shape{m, n, k};
  layout::RowMajor layout_x{m, k};
  auto layout_weight = layout::zN::MakeLayout<int8_t>(k, n);
  typename MatmulKernel::Params params{
      problem_shape, x, layout_x, weight, layout_weight, scale, bias,
      quant_scale, quant_offset, y, AscendC::GetUserWorkspace(workspace)};
  MatmulKernel matmul;
  matmul(params);
}

extern "C" __global__ __aicore__ void quant_matmul_nz_swiglu_decode(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR bias,
    GM_ADDR quant_scale, GM_ADDR quant_offset, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  GET_TILING_DATA(tiling_data, tiling);

  if (TILING_KEY_IS(0)) {
    RunQuantMatmulNzSwigluDecode<false>(
        x, weight, scale, bias, quant_scale, quant_offset, y, workspace,
        tiling_data.m, tiling_data.k, tiling_data.n);
  } else if (TILING_KEY_IS(1)) {
    RunQuantMatmulNzSwigluDecode<true>(
        x, weight, scale, bias, quant_scale, quant_offset, y, workspace,
        tiling_data.m, tiling_data.k, tiling_data.n);
  }
}

#include "lib/matmul_intf.h"

namespace AscendC {

__aicore__ inline void PrepareQuantMatmulNzSwigluWorkspace(
    __gm__ uint8_t*) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
  SetAtomicNone();
  SetMaskNorm();
  SetLoadDataBoundary(static_cast<uint64_t>(0));
  SetLoadDataPaddingValue(static_cast<uint64_t>(0));
  NotifyEvent<PIPE_MTE3>(WORKSPACE_SYNC_ID);
#endif
}

}  // namespace AscendC

#define clearWorkspace(workspace) PrepareQuantMatmulNzSwigluWorkspace(workspace)
