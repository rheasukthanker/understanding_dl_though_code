import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # Pad input
    if padding > 0:
        padded = np.zeros((input_height + 2 * padding,
                           input_width + 2 * padding))
        padded[padding:padding + input_height,
               padding:padding + input_width] = input_matrix
    else:
        padded = input_matrix

    padded_height, padded_width = padded.shape

    # Output shape: (H + 2P - KH)/S + 1
    out_height = (padded_height - kernel_height) // stride + 1
    out_width  = (padded_width  - kernel_width)  // stride + 1

    output = np.zeros((out_height, out_width))

    # Convolution
    for i in range(out_height):
        for j in range(out_width):
            start_row = i * stride
            start_col = j * stride
            patch = padded[start_row:start_row + kernel_height,
                           start_col:start_col + kernel_width]
            output[i, j] = np.sum(patch * kernel)

    return output


import numpy as np

def simple_conv2d_batched(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    H, W = input_matrix.shape
    KH, KW = kernel.shape

    # ----- 1. Pad -----
    if padding > 0:
        padded = np.pad(input_matrix, pad_width=padding, mode='constant')
    else:
        padded = input_matrix

    PH, PW = padded.shape

    # ----- 2. Output shape -----
    out_h = (PH - KH) // stride + 1
    out_w = (PW - KW) // stride + 1

    # ----- 3. Extract all patches (batched) -----
    s0, s1 = padded.strides  # row stride, col stride

    patches = np.lib.stride_tricks.as_strided(
        padded,
        shape=(out_h, out_w, KH, KW),
        strides=(s0 * stride, s1 * stride, s0, s1)
    )
    # patches shape: (out_h, out_w, KH, KW)

    # ----- 4. Convolution via broadcasting -----
    output = np.sum(patches * kernel, axis=(2, 3))

    return output
