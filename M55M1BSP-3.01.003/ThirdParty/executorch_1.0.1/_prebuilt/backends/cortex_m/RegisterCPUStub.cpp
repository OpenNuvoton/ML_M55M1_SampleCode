/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include <torch/library.h>
#include <ATen/Tensor.h>



namespace torch {
namespace executor {
namespace function {


at::Tensor & wrapper_CPU_out_quantize_per_tensor_out(const at::Tensor & input, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max, at::ScalarType dtype, at::Tensor & out) {
    return out;
}


at::Tensor & wrapper_CPU_out_dequantize_per_tensor_out(const at::Tensor & input, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max, at::ScalarType dtype, at::Tensor & out) {
    return out;
}


at::Tensor & wrapper_CPU_out_quantized_add_out(const at::Tensor & self, const at::Scalar & self_zero_point, const at::Scalar & self_multiplier, const at::Scalar & self_shift, const at::Tensor & other, const at::Scalar & other_zero_point, const at::Scalar & other_multiplier, const at::Scalar & other_shift, const at::Scalar & output_zero_point, const at::Scalar & output_multiplier, const at::Scalar & output_shift, at::Tensor & out) {
    return out;
}


at::Tensor & wrapper_CPU_out_quantized_mul_out(const at::Tensor & self, const at::Scalar & self_zero_point, const at::Tensor & other, const at::Scalar & other_zero_point, const at::Scalar & output_zero_point, const at::Scalar & output_multiplier, const at::Scalar & output_shift, at::Tensor & out) {
    return out;
}


at::Tensor & wrapper_CPU_out_minimum_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    return out;
}


at::Tensor & wrapper_CPU_out_maximum_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    return out;
}


at::Tensor & wrapper_CPU_out_quantized_linear_out(const at::Tensor & input, const at::Tensor & weights, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & kernel_sum, const at::Scalar & input_offset, const at::Scalar & filter_offset, const at::Scalar & output_offset, at::IntArrayRef requantize_multipliers, at::IntArrayRef requantize_shifts, const at::Scalar & activation_max, const at::Scalar & activation_min, at::Tensor & out) {
    return out;
}


at::Tensor & wrapper_CPU_out_transpose_out(const at::Tensor & input, at::IntArrayRef perm, at::Tensor & out) {
    return out;
}


at::Tensor & wrapper_CPU_out_quantized_conv2d_out(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t input_offset, int64_t output_offset, const at::Tensor & requantize_multipliers, const at::Tensor & requantize_shifts, int64_t activation_min, int64_t activation_max, at::Tensor & out) {
    return out;
}

// All out variants ops

TORCH_LIBRARY_IMPL(cortex_m, CPU, m) {
m.impl("quantize_per_tensor.out",
TORCH_FN(wrapper_CPU_out_quantize_per_tensor_out));

m.impl("dequantize_per_tensor.out",
TORCH_FN(wrapper_CPU_out_dequantize_per_tensor_out));

m.impl("quantized_add.out",
TORCH_FN(wrapper_CPU_out_quantized_add_out));

m.impl("quantized_mul.out",
TORCH_FN(wrapper_CPU_out_quantized_mul_out));

m.impl("minimum.out",
TORCH_FN(wrapper_CPU_out_minimum_out));

m.impl("maximum.out",
TORCH_FN(wrapper_CPU_out_maximum_out));

m.impl("quantized_linear.out",
TORCH_FN(wrapper_CPU_out_quantized_linear_out));

m.impl("transpose.out",
TORCH_FN(wrapper_CPU_out_transpose_out));

m.impl("quantized_conv2d.out",
TORCH_FN(wrapper_CPU_out_quantized_conv2d_out));

}

namespace cpu {



} // namespace cpu

} // namespace function
} // namespace executor
} // namespace torch
