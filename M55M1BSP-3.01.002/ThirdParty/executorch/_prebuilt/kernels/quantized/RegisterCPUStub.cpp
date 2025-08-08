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


at::Tensor & wrapper_CPU_out_add_out(const at::Tensor & a, double a_scale, int64_t a_zero_point, int64_t a_quant_min, int64_t a_quant_max, const at::Tensor & b, double b_scale, int64_t b_zero_point, int64_t b_quant_min, int64_t b_quant_max, double out_scale, int64_t out_zero_point, int64_t out_quant_min, int64_t out_quant_max, at::Tensor & out) {
    return out;
}
    

::std::tuple<at::Tensor &,at::Tensor &> wrapper_CPU_Tensor_out_choose_qparams_out(const at::Tensor & input, int64_t quant_min, int64_t quant_max, double eps, at::ScalarType dtype, at::Tensor & scale_out, at::Tensor & zero_point_out) {
    return ::std::tuple<at::Tensor &, at::Tensor &>(
                scale_out, zero_point_out
            );
}
    

at::Tensor & wrapper_CPU_out_dequantize_per_tensor_out(const at::Tensor & input, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max, at::ScalarType dtype, ::std::optional<at::ScalarType> out_dtype, at::Tensor & out) {
    return out;
}
    

at::Tensor & wrapper_CPU_Tensor_out_dequantize_per_tensor_out(const at::Tensor & input, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max, at::ScalarType dtype, ::std::optional<at::ScalarType> out_dtype, at::Tensor & out) {
    return out;
}
    

at::Tensor & wrapper_CPU_out_quantize_per_channel_out(const at::Tensor & input, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, int64_t quant_min, int64_t quant_max, at::ScalarType dtype, at::Tensor & out) {
    return out;
}
    

at::Tensor & wrapper_CPU_out_dequantize_per_channel_out(const at::Tensor & input, const at::Tensor & scales, const ::std::optional<at::Tensor> & zero_points, int64_t axis, int64_t quant_min, int64_t quant_max, at::ScalarType dtype, ::std::optional<at::ScalarType> out_dtype, at::Tensor & out) {
    return out;
}
    

at::Tensor & wrapper_CPU_out_embedding_byte_out(const at::Tensor & weight, const at::Tensor & weight_scales, const ::std::optional<at::Tensor> & weight_zero_points, int64_t weight_quant_min, int64_t weight_quant_max, const at::Tensor & indices, at::Tensor & out) {
    return out;
}
    

at::Tensor & wrapper_CPU_dtype_out_embedding_byte_out(const at::Tensor & weight, const at::Tensor & weight_scales, const ::std::optional<at::Tensor> & weight_zero_points, int64_t weight_quant_min, int64_t weight_quant_max, const at::Tensor & indices, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
    return out;
}
    

at::Tensor & wrapper_CPU_out_embedding_2bit_out(const at::Tensor & weight, const at::Tensor & weight_scales, const ::std::optional<at::Tensor> & weight_zero_points, int64_t weight_quant_min, int64_t weight_quant_max, const at::Tensor & indices, at::Tensor & out) {
    return out;
}
    

at::Tensor & wrapper_CPU_dtype_out_embedding_2bit_out(const at::Tensor & weight, const at::Tensor & weight_scales, const ::std::optional<at::Tensor> & weight_zero_points, int64_t weight_quant_min, int64_t weight_quant_max, const at::Tensor & indices, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
    return out;
}
    

at::Tensor & wrapper_CPU_out_embedding_4bit_out(const at::Tensor & weight, const at::Tensor & weight_scales, const ::std::optional<at::Tensor> & weight_zero_points, int64_t weight_quant_min, int64_t weight_quant_max, const at::Tensor & indices, at::Tensor & out) {
    return out;
}
    

at::Tensor & wrapper_CPU_dtype_out_embedding_4bit_out(const at::Tensor & weight, const at::Tensor & weight_scales, const ::std::optional<at::Tensor> & weight_zero_points, int64_t weight_quant_min, int64_t weight_quant_max, const at::Tensor & indices, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
    return out;
}
    

at::Tensor & wrapper_CPU_out_mixed_mm_out(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & weight_scales, const ::std::optional<at::Tensor> & weight_zero_points, at::Tensor & out) {
    return out;
}
    

at::Tensor & wrapper_CPU_out_mixed_linear_out(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & weight_scales, const ::std::optional<at::Tensor> & weight_zero_points, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
    return out;
}
    

at::Tensor & wrapper_CPU_out_quantize_per_tensor_out(const at::Tensor & input, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max, at::ScalarType dtype, at::Tensor & out) {
    return out;
}
    

at::Tensor & wrapper_CPU_Tensor_out_quantize_per_tensor_out(const at::Tensor & input, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max, at::ScalarType dtype, at::Tensor & out) {
    return out;
}
    

::std::tuple<at::Tensor &,at::Tensor &> wrapper_CPU_out_choose_qparams_per_token_asymmetric_out(const at::Tensor & input, at::ScalarType dtype, at::Tensor & scale_out, at::Tensor & zero_point_out) {
    return ::std::tuple<at::Tensor &, at::Tensor &>(
                scale_out, zero_point_out
            );
}
    

at::Tensor & wrapper_CPU_out_quantize_per_token_out(const at::Tensor & input, const at::Tensor & scales, const at::Tensor & zero_points, int64_t quant_min, int64_t quant_max, at::ScalarType dtype, at::Tensor & out) {
    return out;
}
    

at::Tensor & wrapper_CPU_out_dequantize_per_token_out(const at::Tensor & input, const at::Tensor & scales, const at::Tensor & zero_points, int64_t quant_min, int64_t quant_max, at::ScalarType dtype, at::ScalarType output_dtype, at::Tensor & out) {
    return out;
}

// All out variants ops

TORCH_LIBRARY_IMPL(quantized_decomposed, CPU, m) {
m.impl("add.out",
TORCH_FN(wrapper_CPU_out_add_out));

m.impl("choose_qparams.Tensor_out",
TORCH_FN(wrapper_CPU_Tensor_out_choose_qparams_out));

m.impl("dequantize_per_tensor.out",
TORCH_FN(wrapper_CPU_out_dequantize_per_tensor_out));

m.impl("dequantize_per_tensor.Tensor_out",
TORCH_FN(wrapper_CPU_Tensor_out_dequantize_per_tensor_out));

m.impl("quantize_per_channel.out",
TORCH_FN(wrapper_CPU_out_quantize_per_channel_out));

m.impl("dequantize_per_channel.out",
TORCH_FN(wrapper_CPU_out_dequantize_per_channel_out));

m.impl("embedding_byte.out",
TORCH_FN(wrapper_CPU_out_embedding_byte_out));

m.impl("embedding_byte.dtype_out",
TORCH_FN(wrapper_CPU_dtype_out_embedding_byte_out));

m.impl("embedding_2bit.out",
TORCH_FN(wrapper_CPU_out_embedding_2bit_out));

m.impl("embedding_2bit.dtype_out",
TORCH_FN(wrapper_CPU_dtype_out_embedding_2bit_out));

m.impl("embedding_4bit.out",
TORCH_FN(wrapper_CPU_out_embedding_4bit_out));

m.impl("embedding_4bit.dtype_out",
TORCH_FN(wrapper_CPU_dtype_out_embedding_4bit_out));

m.impl("mixed_mm.out",
TORCH_FN(wrapper_CPU_out_mixed_mm_out));

m.impl("mixed_linear.out",
TORCH_FN(wrapper_CPU_out_mixed_linear_out));

m.impl("quantize_per_tensor.out",
TORCH_FN(wrapper_CPU_out_quantize_per_tensor_out));

m.impl("quantize_per_tensor.Tensor_out",
TORCH_FN(wrapper_CPU_Tensor_out_quantize_per_tensor_out));

m.impl("choose_qparams_per_token_asymmetric.out",
TORCH_FN(wrapper_CPU_out_choose_qparams_per_token_asymmetric_out));

m.impl("quantize_per_token.out",
TORCH_FN(wrapper_CPU_out_quantize_per_token_out));

m.impl("dequantize_per_token.out",
TORCH_FN(wrapper_CPU_out_dequantize_per_token_out));

}

namespace cpu {



} // namespace cpu

} // namespace function
} // namespace executor
} // namespace torch
