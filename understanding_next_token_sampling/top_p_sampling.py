import numpy as np
import torch
def top_p_sampling_numpy(logits, p):
    logits = np.asarray(logits)          # (B, V)
    B, V = logits.shape

    # sort ascending per row (same as your version)
    sorted_indices = np.argsort(logits, axis=1)          # (B, V)
    sorted_logits = np.take_along_axis(logits, sorted_indices, axis=1)

    # softmax per row (numerically stable)
    sorted_logits = sorted_logits - sorted_logits.max(axis=1, keepdims=True)
    sorted_probs = np.exp(sorted_logits)
    sorted_probs /= sorted_probs.sum(axis=1, keepdims=True)

    # cumulative probability per row
    cum_probs = np.cumsum(sorted_probs, axis=1)

    selected = np.empty(B, dtype=int)

    for i in range(B):
        valid = np.where(cum_probs[i] >= (1 - p))[0]

        if len(valid) > 0:
            start = valid[0]
            mask = sorted_indices[i, start:]
        else:
            mask = sorted_indices[i, -1:]

        selected[i] = np.random.choice(mask)

    return selected

import torch
import torch.nn.functional as F

def top_p_sampling_torch(logits, p):
    # logits: (B, V)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)                 # (B, V)
    sorted_logits = logits.gather(dim=-1, index=sorted_indices)                    # (B, V)

    softmax_sorted = F.softmax(sorted_logits, dim=-1)                              # (B, V)
    probs_cumsum = torch.cumsum(softmax_sorted, dim=-1)                            # (B, V)

    keep = probs_cumsum <= p
    keep[:, 0] = True  # ensure at least one kept per row

    # Uniform choice among kept indices (vectorized):
    # argmax of i.i.d. U(0,1) over kept positions is uniform over those positions
    r = torch.rand_like(sorted_logits)
    r = r.masked_fill(~keep, -1.0)                                                 # invalid positions never win
    picked_pos = r.argmax(dim=-1)                                                  # (B,)

    selected_indices = sorted_indices.gather(-1, picked_pos.unsqueeze(-1)).squeeze(-1)  # (B,)
    return selected_indices



if __name__=="__main__":
    logits = np.random.randn(2,50)
    p = 0.8 
    print(top_p_sampling_numpy(logits,p))
    logits = torch.tensor(logits)
    print(top_p_sampling_torch(logits,p))