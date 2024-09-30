import mlx.core as mx

@mx.compile
def flash_attn_v2_multihead(q: mx.array, k: mx.array, v: mx.array, BLOCK_M=4):
    """
    The tiny flash attention implement
    """
    assert q.shape == k.shape
    assert q.shape == v.shape

    # Create output buffer in HBM.
    output_buffer = mx.zeros(v.shape)

    Q_BLOCKS = mx.split(q, BLOCK_M, axis=-2)
    K_BLOCKS = mx.split(k, BLOCK_M, axis=-2)
    V_BLOCKS = mx.split(v, BLOCK_M, axis=-2)

    bs, head, seqlen, headdim = q.shape

    seqlen = q.shape[-2] // BLOCK_M
    for j in range(seqlen):
        qi = Q_BLOCKS[j]
        old_o = output_buffer[..., j * BLOCK_M : (j + 1) * BLOCK_M, :]
        # Create denominator buffer in HBM.
        old_d = mx.zeros((bs, head, BLOCK_M, 1))
        # Create max(x) buffer in HBM.
        old_m = mx.full((bs, head, BLOCK_M, 1), -mx.inf)

        k_block_num = k.shape[-2] // BLOCK_M
        for i in range(k_block_num):
            kj = K_BLOCKS[i]
            vj = V_BLOCKS[i]

            # Compute qk.
            x_qkt = mx.softmax(qi @ kj.transpose(2, 3), axis=-1)
            # Get local max of qk.
            # keepdim to avoid auto squeeze.
            # torch.max() return (max, max_index)
            local_m = mx.max(x_qkt, axis=-1, keepdims=True)

            # Compute new max.
            new_m = mx.maximum(old_m, local_m)
            # Compute numerator. i.e.: e^{x - max(x)}.
            safe_e = mx.exp(x_qkt - new_m)
            # Compute new part of denominator.
            curr_d = mx.sum(safe_e, axis=-1, keepdims=True)
            # Update denominator.
            new_d = old_d * mx.exp(old_m - new_m) + curr_d
            # Update old output and accumulate new output.
            new_o = old_o * mx.exp(old_m - new_m) + safe_e @ vj

            old_m = new_m
            old_d = new_d
            old_o = new_o

        output_buffer[..., j * BLOCK_M : (j + 1) * BLOCK_M, :] = old_o / old_d

    return output_buffer

def flash_attn(q: mx.array, k: mx.array, v: mx.array, BLOCK_M: int = 4) -> mx.array:
    """
    Memory effective flash attention implement
    """
    return flash_attn_v2_multihead(q, k, v, BLOCK_M=BLOCK_M)
