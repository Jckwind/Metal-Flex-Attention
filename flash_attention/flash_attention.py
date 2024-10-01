import mlx.core as mx
import math

@mx.compile
def flash_attn_v2_multihead(q: mx.array, k: mx.array, v: mx.array, BLOCK_M: int):
    """
    Optimized flash attention implementation with scaling and reduced redundancy.
    """
    assert q.shape == k.shape, "Query, Key, and Value must have the same shape."
    assert q.shape == v.shape, "Query, Key, and Value must have the same shape."

    # Create output buffer in HBM.
    output_buffer = mx.zeros(v.shape)

    bs, head, seqlen, headdim = q.shape
    print(q.shape)

    # Scaling factor as used in PyTorch's MultiheadAttention
    scale = 1.0 / math.sqrt(headdim)

    # Calculate the number of full blocks and check for remainder
    num_full_blocks = seqlen // BLOCK_M
    remainder = seqlen % BLOCK_M
    print(f"Number of full blocks: {num_full_blocks}, Remainder: {remainder}")

    # Split Q, K, V into full blocks
    Q_FULL_BLOCKS = mx.split(q[:, :, :num_full_blocks * BLOCK_M, :], num_full_blocks, axis=-2)
    K_FULL_BLOCKS = mx.split(k[:, :, :num_full_blocks * BLOCK_M, :], num_full_blocks, axis=-2)
    V_FULL_BLOCKS = mx.split(v[:, :, :num_full_blocks * BLOCK_M, :], num_full_blocks, axis=-2)

    for j in range(num_full_blocks):
        qi = Q_FULL_BLOCKS[j] * scale  # Apply scaling to queries
        old_o = output_buffer[..., j * BLOCK_M : (j + 1) * BLOCK_M, :]

        # Initialize denominator and maximum buffers in HBM.
        old_d = mx.zeros((bs, head, BLOCK_M, 1))
        old_m = mx.full((bs, head, BLOCK_M, 1), -mx.inf)

        for i in range(num_full_blocks):
            kj = K_FULL_BLOCKS[i]
            vj = V_FULL_BLOCKS[i]

            # Compute QK^T and apply softmax.
            x_qkt = mx.softmax(mx.matmul(qi, kj.transpose(0, 1, 3, 2)), axis=-1)

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
            new_o = old_o * mx.exp(old_m - new_m) + mx.matmul(safe_e, vj)

            # Update buffers for the next iteration.
            old_m = new_m
            old_d = new_d
            old_o = new_o

        # Normalize and store the output.
        output_buffer[..., j * BLOCK_M : (j + 1) * BLOCK_M, :] = old_o / old_d

    # Handle the remainder block if exists
    if remainder > 0:
        qi = q[:, :, num_full_blocks * BLOCK_M :, :] * scale
        old_o = output_buffer[..., num_full_blocks * BLOCK_M :, :]

        # Initialize denominator and maximum buffers in HBM.
        old_d = mx.zeros((bs, head, remainder, 1))
        old_m = mx.full((bs, head, remainder, 1), -mx.inf)

        for i in range(num_full_blocks):
            kj = K_FULL_BLOCKS[i]
            vj = V_FULL_BLOCKS[i]

            # Compute QK^T and apply softmax.
            x_qkt = mx.softmax(mx.matmul(qi, kj.transpose(0, 1, 3, 2)), axis=-1)

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
            new_o = old_o * mx.exp(old_m - new_m) + mx.matmul(safe_e, vj)

            # Update buffers for the next iteration.
            old_m = new_m
            old_d = new_d
            old_o = new_o

        # Normalize and store the output.
        output_buffer[..., num_full_blocks * BLOCK_M :, :] = old_o / old_d

    return output_buffer

def flash_attn(q: mx.array, k: mx.array, v: mx.array, BLOCK_M: int = 128) -> mx.array:
    """
    Memory-efficient flash attention implementation.
    """
    _, _, seqlen, _ = q.shape
    if seqlen < BLOCK_M:
        raise ValueError(f"Sequence length ({seqlen}) must be at least BLOCK_M ({BLOCK_M})")
    return flash_attn_v2_multihead(q, k, v, BLOCK_M=BLOCK_M)
