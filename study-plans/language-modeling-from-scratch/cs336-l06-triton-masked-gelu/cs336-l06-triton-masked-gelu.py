import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def masked_gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements

    # Load and upcast to float32 for stable arithmetic
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    # Tanh-approximate GELU: x/2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    SQRT_2_OVER_PI = 0.7978845608028654    # sqrt(2/pi)
    x3   = x * x * x
    inner = SQRT_2_OVER_PI * (x + 0.044715 * x3)
    y    = 0.5 * x * (1.0 + libdevice.tanh(inner))

    # Store in original dtype (tl.store casts automatically)
    tl.store(out_ptr + offsets, y.to(x_ptr.dtype.element_ty), mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    if x.numel() == 0:
        return

    n         = x.numel()
    BLOCK_SIZE = 256
    grid      = (triton.cdiv(n, BLOCK_SIZE),)

    # Work in float32 internally; for non-float32 dtypes, pass float32 views
    if x.dtype == torch.float32:
        masked_gelu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    else:
        x32   = x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int16)
        # Simpler: cast to float32, run kernel, cast result back
        x32   = x.to(torch.float32)
        out32 = torch.empty(n, dtype=torch.float32, device=x.device)
        masked_gelu_kernel[grid](x32, out32, n, BLOCK_SIZE=BLOCK_SIZE)
        out.copy_(out32.to(x.dtype))