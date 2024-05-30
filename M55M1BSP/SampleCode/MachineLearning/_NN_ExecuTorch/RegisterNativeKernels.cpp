/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/profiler.h>
#include "PortableOpsNativeFunctions.h" // Generated Function import headers
#include "QuantizedOpsNativeFunctions.h" // Generated Function import headers

//Copy CPU native kernel operator from  RegisterCodegenUnboxedKernelsEverything.cpp to here. These require by NN model

using KernelArrayRef = ::torch::executor::ArrayRef<::torch::executor::Kernel>;
namespace torch {
namespace executor {

static Kernel kernels_to_register[] = {
    Kernel(
        "aten::max_pool2d_with_indices.out",
        [](torch::executor::KernelRuntimeContext & context, EValue** stack) {
            EValue& self = *stack[0];
    	EValue& kernel_size = *stack[1];
    	EValue& stride = *stack[2];
    	EValue& padding = *stack[3];
    	EValue& dilation = *stack[4];
    	EValue& ceil_mode = *stack[5];
    	EValue& out = *stack[6];
    	EValue& indices = *stack[7];
    	const torch::executor::Tensor & self_base = self.to<torch::executor::Tensor>();
    	
    	    torch::executor::ArrayRef<int64_t> kernel_size_list_out = kernel_size.toIntList();
    	                
    	
    	    torch::executor::ArrayRef<int64_t> stride_list_out = stride.toIntList();
    	                
    	
    	    torch::executor::ArrayRef<int64_t> padding_list_out = padding.toIntList();
    	                
    	
    	    torch::executor::ArrayRef<int64_t> dilation_list_out = dilation.toIntList();
    	                
    	bool ceil_mode_base = ceil_mode.to<bool>();
    	torch::executor::Tensor & out_base = out.to<torch::executor::Tensor>();
    	torch::executor::Tensor & indices_base = indices.to<torch::executor::Tensor>();
    
            internal::EventTracerProfileScope event_tracer_scope(context.internal_event_tracer(), "native_call_max_pool2d_with_indices.out");
            EXECUTORCH_SCOPE_PROF("native_call_max_pool2d_with_indices.out");
            torch::executor::native::max_pool2d_with_indices_out(context, self_base, kernel_size_list_out, stride_list_out, padding_list_out, dilation_list_out, ceil_mode_base, out_base, indices_base);
            internal::event_tracer_log_evalue(context.internal_event_tracer(), *stack[6]);
    internal::event_tracer_log_evalue(context.internal_event_tracer(), *stack[7]);
    
            
        }
    ),

    Kernel(
        "aten::_log_softmax.out",
        [](torch::executor::KernelRuntimeContext & context, EValue** stack) {
            EValue& self = *stack[0];
    	EValue& dim = *stack[1];
    	EValue& half_to_float = *stack[2];
    	EValue& out = *stack[3];
    	const torch::executor::Tensor & self_base = self.to<torch::executor::Tensor>();
    	int64_t dim_base = dim.to<int64_t>();
    	bool half_to_float_base = half_to_float.to<bool>();
    	torch::executor::Tensor & out_base = out.to<torch::executor::Tensor>();
    
            internal::EventTracerProfileScope event_tracer_scope(context.internal_event_tracer(), "native_call__log_softmax.out");
            EXECUTORCH_SCOPE_PROF("native_call__log_softmax.out");
            torch::executor::native::log_softmax_out(context, self_base, dim_base, half_to_float_base, out_base);
            internal::event_tracer_log_evalue(context.internal_event_tracer(), *stack[3]);
    
            
        }
    ),

    Kernel(
        "quantized_decomposed::quantize_per_tensor.out",
        [](torch::executor::KernelRuntimeContext & context, EValue** stack) {
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
    
            internal::EventTracerProfileScope event_tracer_scope(context.internal_event_tracer(), "native_call_quantize_per_tensor.out");
            EXECUTORCH_SCOPE_PROF("native_call_quantize_per_tensor.out");
            torch::executor::native::quantize_per_tensor_out(context, input_base, scale_base, zero_point_base, quant_min_base, quant_max_base, dtype_base, out_base);
            internal::event_tracer_log_evalue(context.internal_event_tracer(), *stack[6]);
    
            
        }
    ),
};

// Explicitly convert to ArrayRef, so that the API can take an empty C array of
// Kernels.
static KernelArrayRef kernel_array_ref(
    kernels_to_register,
    kernels_to_register + sizeof(kernels_to_register) / sizeof(Kernel));

Error register_all_kernels() {
    Error success_with_kernel_reg = register_kernels(kernel_array_ref);
//    Error success_with_kernel_reg = register_kernels(kernels_to_register);

    if(success_with_kernel_reg != Error::Ok) {
        ET_LOG(Error, "Failed register all kernels");
    }
    return success_with_kernel_reg;
}

}   // namespace executer
}   // namespace torch