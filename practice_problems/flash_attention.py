import numpy as np

def flash_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                           block_size: int = 2) -> np.ndarray:
    """
    Compute attention output using Flash Attention v1 algorithm.
    
    Args:
        Q: Query matrix (seq_len, d_model)
        K: Key matrix (seq_len, d_model)
        V: Value matrix (seq_len, d_model)
        block_size: Size of blocks for tiled computation
    
    Returns:
        Output matrix (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    scale = 1.0 / np.sqrt(d_model)
    
    # Initialize output and statistics
    O = np.zeros((seq_len, d_model))
    l = np.zeros(seq_len)  # Softmax denominator
    m = np.full(seq_len, -np.inf)  # Max for stability
    
    num_blocks = (seq_len + block_size - 1) // block_size
    
    # Outer loop: Q blocks
    for i in range(num_blocks):
        q_start = i * block_size
        q_end = min((i + 1) * block_size, seq_len)
        Q_block = Q[q_start:q_end]
        
        # Inner loop: K, V blocks
        for j in range(num_blocks):
            k_start = j * block_size
            k_end = min((j + 1) * block_size, seq_len)
            K_block = K[k_start:k_end]
            V_block = V[k_start:k_end]
            
            # Compute attention scores
            S_block = scale * (Q_block @ K_block.T)
            
            # Update statistics
            m_old = m[q_start:q_end].copy()
            l_old = l[q_start:q_end].copy()
            m_new = np.maximum(m_old, np.max(S_block, axis=1))
            
            # Compute exponentials
            exp_scores = np.exp(S_block - m_new[:, None])
            l_new = np.exp(m_old - m_new) * l_old + np.sum(exp_scores, axis=1)
            
            # Update output
            P_block = exp_scores
            O_block = P_block @ V_block
            correction = np.exp(m_old - m_new)
            O[q_start:q_end] = O[q_start:q_end] * correction[:, None] + O_block
            
            m[q_start:q_end] = m_new
            l[q_start:q_end] = l_new
    
    # Normalize
    O = O / l[:, None]
    return O