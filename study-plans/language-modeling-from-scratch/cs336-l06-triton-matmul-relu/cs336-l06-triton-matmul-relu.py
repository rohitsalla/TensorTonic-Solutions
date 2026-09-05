import torch
import triton
import triton.language as tl

@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, out_ptr,
    m_size, n_size, k_size,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Tile coordinates in the output matrix
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column offsets for this tile
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)

    # float32 accumulator for this output tile
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Iterate over K tiles
    for k_start in range(0, k_size, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)     # (BLOCK_K,)

        # Load A tile: (BLOCK_M, BLOCK_K)
        a_mask = (rows[:, None] < m_size) & (k_offs[None, :] < k_size)
        a_tile = tl.load(
            a_ptr + rows[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=a_mask, other=0.0
        ).to(tl.float32)

        # Load B tile: (BLOCK_K, BLOCK_N)
        b_mask = (k_offs[:, None] < k_size) & (cols[None, :] < n_size)
        b_tile = tl.load(
            b_ptr + k_offs[:, None] * stride_bk + cols[None, :] * stride_bn,
            mask=b_mask, other=0.0
        ).to(tl.float32)

        # tl.dot requires fp16/bf16; cast down, accumulate in fp32
        acc += tl.dot(a_tile.to(tl.float16), b_tile.to(tl.float16),
                      out_dtype=tl.float32)

    # ReLU — clamp negatives to zero
    acc = tl.maximum(acc, 0.0)

    # Masked store into the output tile
    out_mask = (rows[:, None] < m_size) & (cols[None, :] < n_size)
    tl.store(
        out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on,
        acc.to(out_ptr.dtype.element_ty),
        mask=out_mask,
    )


def solve(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("matrices must have aligned rank-two shapes")
    if out.shape != (a.shape[0], b.shape[1]):
        raise ValueError("out must have shape (M, N)")
    if not a.is_cuda or not b.is_cuda or not out.is_cuda or a.device != b.device or a.device != out.device:
        raise ValueError("all tensors must be on the same CUDA device")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("a and b must have a matching supported dtype")
    if out.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("out has an unsupported dtype")

    M, K = a.shape
    N    = b.shape[1]

    if M == 0 or N == 0:
        return
    if K == 0:
        out.zero_()
        return

    # tl.dot needs fp16; cast bfloat16 inputs on the host
    ka = a.to(torch.float16) if a.dtype == torch.bfloat16 else a
    kb = b.to(torch.float16) if b.dtype == torch.bfloat16 else b

    # bfloat16 output: accumulate into float32 then copy back
    if out.dtype == torch.bfloat16:
        kout = torch.empty(M, N, dtype=torch.float32, device=out.device)
    else:
        kout = out

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_relu_kernel[grid](
        ka, kb, kout,
        M, N, K,
        ka.stride(0), ka.stride(1),
        kb.stride(0), kb.stride(1),
        kout.stride(0), kout.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    if kout is not out:
        out.copy_(kout.to(out.dtype))