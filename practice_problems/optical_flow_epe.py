from typing import Optional, Union

try:
    import numpy as np
except Exception:
    np = None

ArrayLike = Union[list, 'np.ndarray']

def _to_numpy(x):
    if np is None:
        raise ImportError('NumPy is required for this problem.')
    return np.asarray(x)

def flow_epe(pred: ArrayLike,
             gt: ArrayLike,
             mask: Optional[ArrayLike] = None,
             max_flow: Optional[float] = None) -> float:
    p = _to_numpy(pred).astype(float)
    g = _to_numpy(gt).astype(float)

    # Shape checks
    if p.shape != g.shape or p.ndim != 3 or p.shape[-1] != 2:
        return -1

    # Per-pixel EPE
    diff = p - g
    epe = np.sqrt(np.sum(diff ** 2, axis=-1))  # (H, W)

    # Clip outliers if requested
    if max_flow is not None:
        epe = np.minimum(epe, float(max_flow))

    # Validity: finite values only
    valid = np.isfinite(epe)

    # Apply mask if provided (broadcast allowed)
    if mask is not None:
        m = _to_numpy(mask)
        try:
            m = np.broadcast_to(m, epe.shape)
        except ValueError:
            return -1
        valid = np.logical_and(valid, m > 0.5)

    vals = epe[valid]
    if vals.size == 0:
        return -1

    return float(vals.mean())