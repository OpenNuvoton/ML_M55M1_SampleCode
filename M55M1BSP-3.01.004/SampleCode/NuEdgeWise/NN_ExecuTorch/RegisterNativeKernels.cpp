/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/profiler.h>
#include "Portable/NativeFunctions.h" // Generated Function import headers
#include "Quantized/NativeFunctions.h" // Generated Function import headers
#include "cortex_m/NativeFunctions.h" // Generated Function import headers

//Copy CPU native kernel operator from RegisterCodegenUnboxedKernelsEverything.cpp to here. These require by NN model

//using KernelArrayRef = ::torch::executor::ArrayRef<::torch::executor::Kernel>;
using KernelSpan = ::executorch::runtime::Span<
                   const ::executorch::ET_RUNTIME_NAMESPACE::Kernel>;
namespace torch {
namespace executor {

static Kernel kernels_to_register[] = {
    Kernel(
    "cortex_m::quantize_per_tensor.out",
    [](torch::executor::KernelRuntimeContext & context, Span<EValue*> stack) {
        ET_KERNEL_CHECK_MSG(context, stack.size() == 8, InvalidProgram, /*void*/, "Expected %" ET_PRIsize_t "args received %" ET_PRIsize_t, (size_t)8, stack.size());
        EValue& input = *stack[0];
        EValue& scale = *stack[1];
        EValue& zero_point = *stack[2];
        EValue& quant_min = *stack[3];
        EValue& quant_max = *stack[4];
        EValue& dtype = *stack[5];
        EValue& out = *stack[6];
        const torch::executor::Tensor & input_base = input.to<torch::executor::Tensor>();
        double scale_base = scale.to<double>();
        int64_t zero_point_base = zero_point.to<int64_t>();
        int64_t quant_min_base = quant_min.to<int64_t>();
        int64_t quant_max_base = quant_max.to<int64_t>();
        torch::executor::ScalarType dtype_base = dtype.to<torch::executor::ScalarType>();
        torch::executor::Tensor & out_base = out.to<torch::executor::Tensor>();

        internal::EventTracerProfileOpScope event_tracer_op_scope(context.internal_event_tracer(), "native_call_quantize_per_tensor.out");
        EXECUTORCH_SCOPE_PROF("native_call_quantize_per_tensor.out");
        cortex_m::native::quantize_per_tensor_out(context, input_base, scale_base, zero_point_base, quant_min_base, quant_max_base, dtype_base, out_base);
        internal::event_tracer_log_evalue(context.internal_event_tracer(), *stack[6]);

    }
    ),

    Kernel(
        "cortex_m::dequantize_per_tensor.out",
    [](torch::executor::KernelRuntimeContext & context, Span<EValue*> stack) {
        ET_KERNEL_CHECK_MSG(context, stack.size() == 8, InvalidProgram, /*void*/, "Expected %" ET_PRIsize_t "args received %" ET_PRIsize_t, (size_t)8, stack.size());
        EValue& input = *stack[0];
        EValue& scale = *stack[1];
        EValue& zero_point = *stack[2];
        EValue& quant_min = *stack[3];
        EValue& quant_max = *stack[4];
        EValue& dtype = *stack[5];
        EValue& out = *stack[6];
        const torch::executor::Tensor & input_base = input.to<torch::executor::Tensor>();
        double scale_base = scale.to<double>();
        int64_t zero_point_base = zero_point.to<int64_t>();
        int64_t quant_min_base = quant_min.to<int64_t>();
        int64_t quant_max_base = quant_max.to<int64_t>();
        torch::executor::ScalarType dtype_base = dtype.to<torch::executor::ScalarType>();
        torch::executor::Tensor & out_base = out.to<torch::executor::Tensor>();

        internal::EventTracerProfileOpScope event_tracer_op_scope(context.internal_event_tracer(), "native_call_dequantize_per_tensor.out");
        EXECUTORCH_SCOPE_PROF("native_call_dequantize_per_tensor.out");
        cortex_m::native::dequantize_per_tensor_out(context, input_base, scale_base, zero_point_base, quant_min_base, quant_max_base, dtype_base, out_base);
        internal::event_tracer_log_evalue(context.internal_event_tracer(), *stack[6]);

    }
    ),
};

// Explicitly convert to ArrayRef, so that the API can take an empty C array of
// Kernels.
static KernelSpan kernel_span(
    kernels_to_register,
    kernels_to_register + sizeof(kernels_to_register) / sizeof(Kernel));

Error register_all_kernels() {
    Error success_with_kernel_reg = register_kernels(kernel_span);

    if(success_with_kernel_reg != Error::Ok) {
        ET_LOG(Error, "Failed register all kernels");
    }
    return success_with_kernel_reg;
}

}   // namespace executer
}   // namespace torch