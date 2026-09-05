#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

// ── Typed element load/store helpers ────────────────────────────────────────
__device__ __forceinline__ float load_elem(const void* base,
                                            long long byte_offset,
                                            int dtype_code) {
    const char* ptr = (const char*)base + byte_offset;
    if (dtype_code == 0) return *((const float*)ptr);
    if (dtype_code == 1) return __half2float(*((const __half*)ptr));
    // dtype_code == 2
    return __bfloat162float(*((const __nv_bfloat16*)ptr));
}

__device__ __forceinline__ void store_elem(void* base,
                                            long long byte_offset,
                                            float val,
                                            int dtype_code) {
    char* ptr = (char*)base + byte_offset;
    if (dtype_code == 0) { *((float*)ptr) = val; }
    else if (dtype_code == 1) { *(((__half*)ptr)) = __float2half(val); }
    else { *((__nv_bfloat16*)ptr) = __float2bfloat16(val); }
}

__device__ __forceinline__ int dtype_bytes(int dtype_code) {
    if (dtype_code == 0) return 4;   // float32
    if (dtype_code == 1) return 2;   // float16
    return 2;                         // bfloat16
}

// ── Online softmax kernel ────────────────────────────────────────────────────
// One block per row, block_size threads per block.
// Shared memory layout: [0..block_size-1] = m_shared, [block_size..2*block_size-1] = l_shared
__global__ void online_softmax_kernel(
    const void* input,
    void*       output,
    int         rows,
    int         cols,
    long long   input_row_stride,   // in elements
    long long   input_col_stride,   // in elements
    long long   output_row_stride,  // in elements
    long long   output_col_stride,  // in elements
    int         dtype_code
) {
    extern __shared__ float smem[];   // 2 * block_size floats

    const int row      = blockIdx.x;
    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;

    if (row >= rows) return;

    float* sm_m = smem;               // [0..nthreads-1]
    float* sm_l = smem + nthreads;   // [nthreads..2*nthreads-1]

    const int elem_bytes = dtype_bytes(dtype_code);

    // ── Pass 1: each thread accumulates its strided columns ───────────────
    float m_loc = -FLT_MAX;   // running max
    float l_loc = 0.0f;       // running scaled exp-sum

    for (int col = tid; col < cols; col += nthreads) {
        long long in_byte = ((long long)row * input_row_stride + (long long)col * input_col_stride)
                            * elem_bytes;
        float val = load_elem(input, in_byte, dtype_code);

        // Merge (m_loc, l_loc) with (val, 1)
        if (val > m_loc) {
            l_loc = l_loc * __expf(m_loc - val) + 1.0f;
            m_loc = val;
        } else {
            l_loc += __expf(val - m_loc);
        }
    }

    // Store local summaries
    sm_m[tid] = m_loc;
    sm_l[tid] = l_loc;
    __syncthreads();

    // ── Reduction: merge thread summaries in shared memory ────────────────
    for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float ma = sm_m[tid];
            float la = sm_l[tid];
            float mb = sm_m[tid + stride];
            float lb = sm_l[tid + stride];

            float m_new, l_new;
            if (ma >= mb) {
                m_new = ma;
                l_new = la + lb * __expf(mb - ma);
            } else {
                m_new = mb;
                l_new = la * __expf(ma - mb) + lb;
            }
            sm_m[tid] = m_new;
            sm_l[tid] = l_new;
        }
        __syncthreads();
    }

    // Thread 0 now holds the row's global (m, l)
    float m_global = sm_m[0];
    float l_global = sm_l[0];

    // Guard: if l_global is 0 (all -inf inputs), write 0s and return
    if (l_global == 0.0f) return;

    // ── Pass 2: write normalized outputs ─────────────────────────────────
    for (int col = tid; col < cols; col += nthreads) {
        long long in_byte  = ((long long)row * input_row_stride  + (long long)col * input_col_stride)  * elem_bytes;
        long long out_byte = ((long long)row * output_row_stride + (long long)col * output_col_stride) * elem_bytes;

        float val    = load_elem(input, in_byte, dtype_code);
        float result = __expf(val - m_global) / l_global;
        store_elem(output, out_byte, result, dtype_code);
    }
}

// ── Host entry point ─────────────────────────────────────────────────────────
extern "C" void solve(
    const void* input,
    void*       output,
    int         rows,
    int         cols,
    long long   input_row_stride,
    long long   input_col_stride,
    long long   output_row_stride,
    long long   output_col_stride,
    int         block_size,
    int         dtype_code
) {
    // Validate parameters
    if (rows < 0 || cols < 0 ||
        block_size < 32 || block_size > 1024 ||
        (block_size & (block_size - 1)) != 0 ||
        dtype_code < 0 || dtype_code > 2) {
        cudaDeviceSynchronize();
        return;
    }
    if (rows == 0 || cols == 0) {
        cudaDeviceSynchronize();
        return;
    }

    // 2 arrays of block_size floats in shared memory
    size_t shared_bytes = 2 * (size_t)block_size * sizeof(float);

    online_softmax_kernel<<<rows, block_size, shared_bytes>>>(
        input, output,
        rows, cols,
        input_row_stride, input_col_stride,
        output_row_stride, output_col_stride,
        dtype_code
    );
    cudaDeviceSynchronize();
}
