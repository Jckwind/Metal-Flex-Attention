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
    bs, num_heads, seqlen, head_dim = 2, 4, 16, 64
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

    # Initialize weights to ones for reproducibility
    torch.nn.init.constant_(multihead_attn.in_proj_weight, 1.0)
    torch.nn.init.constant_(multihead_attn.out_proj.weight, 1.0)
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

    # Compare outputs
    mse = mx.mean((output_mx_np - attn_output_np) ** 2)
    print(f"Mean Squared Error: {mse}")

    if mse < 1e-5:
        print("Correctness test passed: Outputs are similar.")
    else:
        print("Correctness test failed: Outputs differ significantly.")

    # Performance Test
    print("Running performance test...")

    # Measure performance of flash_attn_v2_multihead
    start_time = time.perf_counter()
    output_mx = flash_attn(q_mx, k_mx, v_mx)
    # Ensure completion
    mx.eval(output_mx)
    end_time = time.perf_counter()
    print(f"flash_attn_v2_multihead Execution Time: {end_time - start_time:.6f} seconds")

    # Measure performance of PyTorch's MultiheadAttention
    start_time = time.perf_counter()
    with torch.no_grad():
        attn_output, _ = multihead_attn(q_torch, k_torch, v_torch)
    # No synchronization needed for CPU
    end_time = time.perf_counter()
    print(f"PyTorch MultiheadAttention Execution Time: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    main()
