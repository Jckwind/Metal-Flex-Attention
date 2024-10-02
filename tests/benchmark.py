import mlx.core as mx
from flash_attention import flash_attn
import torch
import time
import numpy as np

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Determine device (force to CPU for fair comparison)
    device = torch.device("mps")  # Changed from "mps" to "cpu"
    print(f"Using device: {device}")

    # Generate random inputs
    bs, num_heads, seqlen, head_dim = 1, 8, 4096, 64
    q = torch.randn(bs, num_heads, seqlen, head_dim, device=device)
    k = torch.randn(bs, num_heads, seqlen, head_dim, device=device)
    v = torch.randn(bs, num_heads, seqlen, head_dim, device=device)

    # Convert to mlx arrays without unnecessary transfers
    q_mx = mx.array(q.cpu().numpy())
    k_mx = mx.array(k.cpu().numpy())
    v_mx = mx.array(v.cpu().numpy())

    # Initialize PyTorch MultiheadAttention
    embed_dim = num_heads * head_dim
    multihead_attn = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, bias=False)
    multihead_attn.to(device)

    # Use Xavier Initialization
    torch.nn.init.xavier_uniform_(multihead_attn.in_proj_weight)
    torch.nn.init.xavier_uniform_(multihead_attn.out_proj.weight)

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

    print("output_mx_np shape:", output_mx_np.shape)
    print("attn_output_np shape:", attn_output_np.shape)

    # Reshape attn_output_np to match output_mx_np
    attn_output_np_reshaped = np.transpose(attn_output_np, (2, 0, 1, 3))

    # Ensure shapes match
    assert output_mx_np.shape == attn_output_np_reshaped.shape, "Shapes still don't match after reshaping"

    # Calculate MSE
    mse = np.mean((output_mx_np - attn_output_np_reshaped) ** 2)
    print("Mean Squared Error: ", mse)

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
    end_time = time.perf_counter_ns()
    execution_time_ns = end_time - start_time
    execution_time_ms = execution_time_ns / 1e6
    print(f"PyTorch MultiheadAttention Execution Time: {execution_time_ms:.3f} ms ({execution_time_ns:,} ns)")

if __name__ == "__main__":
    main()