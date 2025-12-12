import numpy as np
def sinusoidal_positional_encoding(position, d_model):
    i = np.arange(d_model)
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = position * angle_rates

    pos_encoding = np.zeros(d_model)
    pos_encoding[0::2] = np.sin(angle_rads[0::2])
    pos_encoding[1::2] = np.cos(angle_rads[1::2])
    return pos_encoding