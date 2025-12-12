import torch
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies RoPE transform to `x`. Note that `cos`, `sin` need to have a batch
    dimension.

    Args:
        x: Input tensor, `(B, ..., T, head_size)`
        cos: Cached cosines, `(B, T, head_size)` or `(1, T, head_size)`
        sin: Cached sines, `(B, T, head_size)` or `(1, T, head_size)`

    Returns:
        Encoded tensor, `(B, ..., T, head_size)`
    """
    if cos.dim() != 3:
        raise ValueError(f"cos must be three-dimensional, but shape is {cos.shape}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos, sin must have same shape, but cos.shape={cos.shape}, sin.shape={sin.shape}")
    head_size_half = x.size(-1) // 2
    x1 = x[..., :head_size_half]  # (B, ..., T, head_size/2)
    x2 = x[..., head_size_half:]  # (B, ..., T, head_size/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, ..., T, head_size)
    dims_diff = x.dim() - cos.dim()
    if dims_diff > 0:
        # Ensure that shapes of `x`, `cos`, `sin` align
        new_shape = cos.shape[0:1] + (1,) * dims_diff + cos.shape[1:]
        cos = cos.view(*new_shape)
        sin = sin.view(*new_shape)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)

def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
    extra_config: Optional[dict] = None,
    rope_local_base_freq: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced Transformer with Rotary Position Embedding.

    Args:
        seq_len (int): Sequence length.
        n_elem (int): Number of elements (head dimension).
        device (torch.device, optional): Device for tensor allocations.
        base (int, optional): Base for computing inverse frequencies.
        condense_ratio (int, optional): Ratio to condense the position indices.
        extra_config (dict, optional): Configuration parameters for frequency adjustments (used by Llama 3.1 and 3.2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine caches for RoPE.
            Shapes are `(seq_len, n_elem)`.
    """

    # Compute the inverse frequencies theta
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    # Create position indices `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device).float() / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    # If `n_elem` is odd, the final dimension of `idx_theta` has size
    # `n_elem + 1`, so need to cut something off.
    # Due to a current bug in Hugging Face, in the case `n_elem == 1`, we leave
    # `idx_theta`, `cos`, `sin` as is. Things work out in `apply_rope` due to
    # broadcasting. If we shorten `idx_theta`, unit tests comparing to
    # Hugging Face fail.
    # https://github.com/huggingface/transformers/issues/35233
    if idx_theta.shape[-1] > n_elem > 1:
        idx_theta = idx_theta[..., :n_elem]


    return torch.cos(idx_theta), torch.sin(idx_theta)