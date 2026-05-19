#pragma once

#include "common.cuh"

// Fast Walsh-Hadamard Transform on CUDA. Mirrors the CPU implementation
// added in PR #22631 (ggml-cpu/ops.cpp:ggml_compute_forward_fwht_f32).
// Routed from ggml_cuda_mul_mat when the GGML_HINT_SRC0_IS_HADAMARD hint
// is set on the destination tensor's op_params[1]. Operates on src1 only;
// src0 is a shape-marker tensor in the hadamard mul_mat encoding (its
// data is ignored).
//
// src1 must be F32 and contiguous in the inner dimension; ne10 (n) must
// be a power of two. Strides for outer dims are honored.

void ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
