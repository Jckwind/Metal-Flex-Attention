import mlx.core as mx
from flash_attention import flash_attn
import torch
import time
import numpy as np

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Determine device (force to CPU)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Generate random inputs
    bs, num_heads, seqlen, head_dim = 2, 1, 64, 64
    q = torch.randn(bs, num_heads, seqlen, head_dim, device=device)
    k = torch.randn(bs, num_heads, seqlen, head_dim, device=device)
    v = torch.randn(bs, num_heads, seqlen, head_dim, device=device)

    # Convert to numpy
    q_np = q.cpu().numpy()
    k_np = k.cpu().numpy()
    v_np = v.cpu().numpy()

    # Convert to mlx arrays
    q_mx = mx.array(q_np)
    k_mx = mx.array(k_np)
    v_mx = mx.array(v_np)

    # Initialize PyTorch MultiheadAttention
    embed_dim = head_dim
    multihead_attn = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, bias=False)
    multihead_attn.to(device)

    # Use Xavier Initialization instead of setting all weights to 1.0
    torch.nn.init.xavier_uniform_(multihead_attn.in_proj_weight)
    torch.nn.init.xavier_uniform_(multihead_attn.out_proj.weight)
    # Removed the following line since bias=False
    # torch.nn.init.constant_(multihead_attn.out_proj.bias, 0.0)

    # Correctness Test
    print("Running correctness test...")

    # Run flash_attn_v2_multihead
    output_mx = flash_attn(q_mx, k_mx, v_mx)
    output_mx_np = np.array(output_mx)

    # Run PyTorch's MultiheadAttention
    # Transpose for PyTorch: (seqlen, batch, num_heads * head_dim)
    q_torch = q.transpose(0, 2).reshape(seqlen, bs, num_heads * head_dim)
    k_torch = k.transpose(0, 2).reshape(seqlen, bs, num_heads * head_dim)
    v_torch = v.transpose(0, 2).reshape(seqlen, bs, num_heads * head_dim)

    with torch.no_grad():
        attn_output, _ = multihead_attn(q_torch, k_torch, v_torch)

    # Reshape PyTorch output to match mlx output
    attn_output = attn_output.reshape(seqlen, bs, num_heads, head_dim).transpose(1, 2).transpose(0, 1)
    attn_output_np = attn_output.cpu().numpy()

    print("output mx np array", output_mx_np)
    
    print("attn output np array", attn_output_np)
    # Compare outputs
    print(f"output_mx_np shape: {output_mx_np.shape}")
    print(f"attn_output_np shape: {attn_output_np.shape}")

    # Transpose attn_output_np to match output_mx_np
    attn_output_np_transposed = attn_output_np.transpose(2, 0, 1, 3)
    print(f"attn_output_np_transposed shape: {attn_output_np_transposed.shape}")

    # Verify shapes are compatible
    if output_mx_np.shape == attn_output_np_transposed.shape:
        mse = np.mean((output_mx_np - attn_output_np_transposed) ** 2)
        print(f"MSE: {mse}")
    else:
        raise ValueError(f"Shape mismatch after transpose: {output_mx_np.shape} vs {attn_output_np_transposed.shape}")

    # Performance Test
    print("\nRunning performance test...")

    # Measure performance of flash_attn
    start_time = time.perf_counter_ns()
    output_mx = flash_attn(q_mx, k_mx, v_mx)
    # Ensure completion
    mx.eval(output_mx)
    end_time = time.perf_counter_ns()
    execution_time_ns = end_time - start_time
    execution_time_ms = execution_time_ns / 1e6
    print(f"mlx flash_attn Execution Time: {execution_time_ms:.3f} ms ({execution_time_ns:,} ns)")

    # Measure performance of PyTorch's MultiheadAttention
    start_time = time.perf_counter_ns()
    with torch.no_grad():
        attn_output, _ = multihead_attn(q_torch, k_torch, v_torch)
    # No synchronization needed for CPU
    end_time = time.perf_counter_ns()
    execution_time_ns = end_time - start_time
    execution_time_ms = execution_time_ns / 1e6
    print(f"PyTorch MultiheadAttention Execution Time: {execution_time_ms:.3f} ms ({execution_time_ns:,} ns)")

if __name__ == "__main__":
    main()
