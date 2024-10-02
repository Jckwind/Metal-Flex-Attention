import mlx.core as mx
import math

@mx.compile
def flash_attn_v2_multihead_optimized(q: mx.array, k: mx.array, v: mx.array, BLOCK_M: int):
    """
    Optimized flash attention implementation with scaling and reduced redundancy.
    Assumes that the sequence length is evenly divisible by BLOCK_M.
    """
    assert q.shape == k.shape == v.shape, "Query, Key, and Value must have the same shape."

    bs, head, seqlen, headdim = q.shape
    assert seqlen % BLOCK_M == 0, f"Sequence length ({seqlen}) must be divisible by BLOCK_M ({BLOCK_M})"

    scale = 1.0 / math.sqrt(headdim)
    q = q * scale  # Apply scaling to queries

    # Reshape to merge batch and head dimensions for vectorized operations
    q = q.reshape(-1, seqlen, headdim)         # Shape: (bs * head, seqlen, headdim)
    k = k.reshape(-1, seqlen, headdim)
    v = v.reshape(-1, seqlen, headdim)

    num_blocks = seqlen // BLOCK_M

    # Split q, k, v into blocks
    q_blocks = q.reshape(-1, num_blocks, BLOCK_M, headdim)
    k_blocks = k.reshape(-1, num_blocks, BLOCK_M, headdim)
    v_blocks = v.reshape(-1, num_blocks, BLOCK_M, headdim)

    # Initialize output buffer
    output_buffer = mx.zeros_like(q)

    for block_idx in range(num_blocks):  # Loop over query blocks
        qi = q_blocks[:, block_idx, :, :]  # Shape: (bs * head, BLOCK_M, headdim)

        # Initialize accumulators for numerator (numer), denominator (denom), and max score (max_score)
        numer = mx.zeros((qi.shape[0], BLOCK_M, headdim))
        denom = mx.zeros((qi.shape[0], BLOCK_M, 1))
        max_score = mx.full((qi.shape[0], BLOCK_M, 1), -mx.inf)

        for s in range(num_blocks):  # Loop over key/value blocks
            kj = k_blocks[:, s, :, :]  # Shape: (bs * head, BLOCK_M, headdim)
            vj = v_blocks[:, s, :, :]  # Shape: (bs * head, BLOCK_M, headdim)

            # Compute attention scores
            scores = mx.matmul(qi, kj.transpose(0, 2, 1), stream=mx.gpu)  # Shape: (bs * head, BLOCK_M, BLOCK_M)

            # Update max_score for numerical stability
            new_max_score = mx.maximum(mx.max(scores, axis=-1, keepdims=True, stream=mx.gpu), max_score, stream=mx.gpu)

            # Compute exp(scores - new_max_score)
            exp_scores = mx.exp(scores - new_max_score, stream=mx.gpu)

            # Update numerator and denominator with scaling for numerical stability
            numer = numer * mx.exp(max_score - new_max_score, stream=mx.gpu) + mx.matmul(exp_scores, vj, stream=mx.gpu)
            denom = denom * mx.exp(max_score - new_max_score, stream=mx.gpu) + mx.sum(exp_scores, axis=-1, keepdims=True, stream=mx.gpu)

            # Update max_score
            max_score = new_max_score

        # Compute output for the current query block
        output = numer / denom  # Shape: (bs * head, BLOCK_M, headdim)

        # Store output
        output_buffer[:, block_idx * BLOCK_M : (block_idx + 1) * BLOCK_M, :] = output

    # Reshape output buffer back to original dimensions
    output_buffer = output_buffer.reshape(bs, head, seqlen, headdim)

    return output_buffer

def flash_attn(q: mx.array, k: mx.array, v: mx.array, BLOCK_M: int = 128) -> mx.array:
    """
    Memory-efficient flash attention implementation.
    """
    _, _, seqlen, _ = q.shape
    assert seqlen % BLOCK_M == 0, f"Sequence length ({seqlen}) must be divisible by BLOCK_M ({BLOCK_M})"
    return flash_attn_v2_multihead_optimized(q, k, v, BLOCK_M=BLOCK_M)