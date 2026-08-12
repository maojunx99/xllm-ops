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

#include "quant_matmul_nz_swiglu_decode_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t kCoreCount = 20;
constexpr uint32_t kTileM = 16;
constexpr uint32_t kTileN = 160;
constexpr uint32_t kWorkspaceStages = 2;
constexpr uint32_t kMaxOptimizedM = kTileM;

ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const auto x_shape = context->GetInputShape(0)->GetOriginShape();
  const auto weight_shape = context->GetInputShape(1)->GetOriginShape();
  const auto scale_shape = context->GetInputShape(2)->GetOriginShape();
  const auto bias_shape = context->GetInputShape(3)->GetOriginShape();
  const auto quant_scale_shape =
      context->GetInputShape(4)->GetOriginShape();
  const auto quant_offset_shape =
      context->GetInputShape(5)->GetOriginShape();
  if (x_shape.GetDimNum() != 2 || weight_shape.GetDimNum() != 2 ||
      scale_shape.GetDimNum() != 1 || bias_shape.GetDimNum() != 1) {
    return ge::GRAPH_FAILED;
  }

  const int64_t m = x_shape.GetDim(0);
  const int64_t k = x_shape.GetDim(1);
  const int64_t n = weight_shape.GetDim(1);
  if (m <= 0 || m > kMaxOptimizedM || k != 5120 ||
      weight_shape.GetDim(0) != k || n != 6400 ||
      scale_shape.GetDim(0) != n || bias_shape.GetDim(0) != n) {
    return ge::GRAPH_FAILED;
  }

  const bool quant_output =
      *context->GetAttrs()->GetAttrPointer<bool>(0);
  if (quant_output &&
      (quant_scale_shape.GetShapeSize() != 1 ||
       quant_offset_shape.GetShapeSize() != 1)) {
    return ge::GRAPH_FAILED;
  }

  QuantMatmulNzSwigluDecodeTilingData tiling;
  tiling.set_m(static_cast<uint32_t>(m));
  tiling.set_k(static_cast<uint32_t>(k));
  tiling.set_n(static_cast<uint32_t>(n));
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(kCoreCount);
  context->SetTilingKey(quant_output ? 1 : 0);

  auto platform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  context->GetWorkspaceSizes(1)[0] =
      platform.GetLibApiWorkSpaceSize() +
      kTileM * kTileN * kCoreCount * kWorkspaceStages * sizeof(int32_t);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(QuantMatmulNzSwigluDecode).Tiling(TilingFunc);
}  // namespace optiling
