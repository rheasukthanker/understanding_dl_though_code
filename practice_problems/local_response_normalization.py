import numpy as np

def local_response_normalization(x: np.ndarray, n: int = 5, k: float = 2.0, alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    N, C, H, W = x.shape
    half_n = n // 2
    squared = x ** 2
    result = np.zeros_like(x)
    
    for i in range(C):
        start = max(0, i - half_n)
        end = min(C, i + half_n + 1)
        scale = k + alpha * np.sum(squared[:, start:end, :, :], axis=1, keepdims=True)
        result[:, i:i+1, :, :] = x[:, i:i+1, :, :] / (scale ** beta)
    
    return np.round(result, 4)