#include "fwht.cuh"

// One CUDA block processes one row of length n. Threads cooperate via
// shared memory across log2(n) butterfly passes. Constraints on n:
//   - n must be a power of two (asserted by caller)
//   - shared-memory budget caps n at FWHT_MAX_N (set so 4 KiB shared mem
//     per block / 4 bytes per float = 1024 elements). For larger rows
//     this kernel would need to spill to a tiled algorithm; the dflash
//     workloads that exercise this path use n=head_dim which is small
//     (typically 64..128).
#define FWHT_BLOCK_SIZE 128
#define FWHT_MAX_N 1024

static __global__ void k_fwht_f32(
        const float * __restrict__ src,
        float * __restrict__ dst,
        const int n,
        const size_t s11, const size_t s12, const size_t s13,
        const size_t d1,  const size_t d2,  const size_t d3,
        const int ne11, const int ne12, const int ne13) {

    extern __shared__ float s_row[];

    // Decompose linear block index back to (i11, i12, i13).
    int row = blockIdx.x;
    const int i13 = row / (ne11 * ne12);
    row -= i13 * ne11 * ne12;
    const int i12 = row / ne11;
    const int i11 = row - i12 * ne11;

    const float * src_row = src + i11 * s11 + i12 * s12 + i13 * s13;
    float       * dst_row = dst + i11 * d1  + i12 * d2  + i13 * d3;

    const int tid = threadIdx.x;
    const float scale = rsqrtf((float) n);

    // Load + scale.
    for (int i = tid; i < n; i += FWHT_BLOCK_SIZE) {
        s_row[i] = src_row[i] * scale;
    }
    __syncthreads();

    // log2(n) butterfly passes. At each pass there are n/2 butterfly
    // pairs total; threads stride through them.
    const int n_pairs = n >> 1;
    for (int len = 1; len < n; len <<= 1) {
        const int two_len = len << 1;
        for (int p = tid; p < n_pairs; p += FWHT_BLOCK_SIZE) {
            const int group = p / len;
            const int inner = p - group * len;
            const int i = group * two_len + inner;
            const int j = i + len;
            const float u = s_row[i];
            const float v = s_row[j];
            s_row[i] = u + v;
            s_row[j] = u - v;
        }
        __syncthreads();
    }

    // Store.
    for (int i = tid; i < n; i += FWHT_BLOCK_SIZE) {
        dst_row[i] = s_row[i];
    }
}

void ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;
    GGML_UNUSED(src0);

    const int n = (int) ne10;
    GGML_ASSERT(n > 0);
    GGML_ASSERT((n & (n - 1)) == 0 && "FWHT requires power-of-two row length");
    GGML_ASSERT(n <= FWHT_MAX_N && "FWHT shared-memory budget exceeded — extend kernel for n > 1024");

    // Inner dimension must be contiguous.
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0  == sizeof(float));

    const int n_rows = (int) (ne11 * ne12 * ne13);
    if (n_rows == 0) {
        return;
    }

    // Strides in elements.
    const size_t s11 = nb11 / sizeof(float);
    const size_t s12 = nb12 / sizeof(float);
    const size_t s13 = nb13 / sizeof(float);
    const size_t d1  = nb1  / sizeof(float);
    const size_t d2  = nb2  / sizeof(float);
    const size_t d3  = nb3  / sizeof(float);

    cudaStream_t stream = ctx.stream();

    const dim3 grid(n_rows, 1, 1);
    const dim3 block(FWHT_BLOCK_SIZE, 1, 1);
    const size_t smem = n * sizeof(float);

    k_fwht_f32<<<grid, block, smem, stream>>>(
        (const float *) src1->data,
        (float *)       dst->data,
        n,
        s11, s12, s13,
        d1, d2, d3,
        (int) ne11, (int) ne12, (int) ne13);
}
