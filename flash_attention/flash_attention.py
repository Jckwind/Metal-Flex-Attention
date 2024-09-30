import mlx.core as mx
import math

@mx.compile
def flash_attn_v2_multihead(q: mx.array, k: mx.array, v: mx.array, BLOCK_M: int = None):
    """
    Optimized flash attention implementation with scaling and reduced redundancy.
    """
    assert q.shape == k.shape, "Query, Key, and Value must have the same shape."
    assert q.shape == v.shape, "Query, Key, and Value must have the same shape."

    # Create output buffer in HBM.
    output_buffer = mx.zeros(v.shape)

    # Split Q, K, V into blocks along the sequence length axis.
    Q_BLOCKS = mx.split(q, BLOCK_M, axis=-2)
    K_BLOCKS = mx.split(k, BLOCK_M, axis=-2)
    V_BLOCKS = mx.split(v, BLOCK_M, axis=-2)

    bs, head, seqlen, headdim = q.shape
    print(q.shape)

    # Scaling factor as used in PyTorch's MultiheadAttention
    scale = 1.0 / math.sqrt(headdim)

    num_blocks = seqlen // BLOCK_M
    for j in range(num_blocks):
        qi = Q_BLOCKS[j] * scale  # Apply scaling to queries
        old_o = output_buffer[..., j * BLOCK_M : (j + 1) * BLOCK_M, :]
        
        # Initialize denominator and maximum buffers in HBM.
        old_d = mx.zeros((bs, head, BLOCK_M, 1))
        old_m = mx.full((bs, head, BLOCK_M, 1), -mx.inf)

        k_block_num = k.shape[-2] // BLOCK_M
        for i in range(k_block_num):
            kj = K_BLOCKS[i]
            vj = V_BLOCKS[i]

            # Compute QK^T and apply softmax.
            x_qkt = mx.softmax(qi @ kj.transpose(0, 1, 3, 2), axis=-1)
            
            # Compute the maximum for numerical stability.
            local_m = mx.max(x_qkt, axis=-1, keepdims=True)

            # Update the global maximum.
            new_m = mx.maximum(old_m, local_m)
            
            # Compute the exponentials safely.
            safe_e = mx.exp(x_qkt - new_m)
            
            # Update the denominator.
            curr_d = mx.sum(safe_e, axis=-1, keepdims=True)
            new_d = old_d * mx.exp(old_m - new_m) + curr_d
            
            # Accumulate the output.
            new_o = old_o * mx.exp(old_m - new_m) + safe_e @ vj

            # Update buffers for the next iteration.
            old_m = new_m
            old_d = new_d
            old_o = new_o

        # Normalize and store the output.
        output_buffer[..., j * BLOCK_M : (j + 1) * BLOCK_M, :] = old_o / old_d

    return output_buffer

def flash_attn(q: mx.array, k: mx.array, v: mx.array, BLOCK_M: int = 8) -> mx.array:
    """
    Memory-efficient flash attention implementation.
    """
    _, _, seqlen, _ = q.shape
    if seqlen % BLOCK_M != 0:
        raise ValueError(f"BLOCK_M ({BLOCK_M}) must evenly divide the sequence length ({seqlen})")
    return flash_attn_v2_multihead(q, k, v, BLOCK_M=BLOCK_M)
