import numpy as np

def lora_forward(
	x: list[list[float]],
	W: list[list[float]],
	A: list[list[float]],
	B: list[list[float]],
	alpha: float = 1.0
) -> list[list[float]]:
    x = np.array(x)
    W = np.array(W)
    A = np.array(A)
    B = np.array(B)
    rank = B.shape[-1]
    output = x @ W + (alpha / rank) * (x @ B @ A)
    return output