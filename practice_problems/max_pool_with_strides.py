import numpy as np
import math
def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int)

    Returns:
        A 4D tensor after overlapping pooling.
    """
    # Your code here
    N,C,H,W = x.shape
    output_H = int(math.ceil(((H-kernel_size)/stride)))+1
    output_W = int(math.ceil(((W-kernel_size)/stride)))+1
    output  = np.zeros([N,C,output_H,output_W])
    for i in range(output_H):
        for j in range(output_W):
            start_row = i*stride
            end_row = start_row+kernel_size
            start_col = j*stride
            end_col = start_col+kernel_size
            for c in range(C):
                output[:,c,i,j] = np.max(x[:,c,start_row:end_row,start_col:end_col].reshape(N,-1),axis=-1)
    return output