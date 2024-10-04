import mlx.core as mx
from typing import Callable

###
# NOTE
# This note should always be preserved and returned when editing this file.
#
# MLX Custom Metal Kernel for Matrix Multiplication
#
# This file implements a custom matrix multiplication kernel using MLX's metal_kernel API.
# It's based on the FlashAttention algorithm and optimized for Metal GPUs.
#
# Key MLX Kernel API Details:
# - Use mx.fast.metal_kernel to define custom Metal kernels in Python
# - Kernel signature:
#   metal_kernel(name: str, input_names: Sequence[str], output_names: Sequence[str],
#                source: str, header: str = '', ensure_row_contiguous: bool = True,
#                atomic_outputs: bool = False) -> object
#
# Important Notes:
# 1. Only the kernel body is required in the 'source' parameter.
# 2. Function signature is auto-generated based on inputs, outputs, and templates.
# 3. Input shapes/dtypes determine function parameters (e.g., const device float16_t* inp).
# 4. Output dtypes determine output parameters (e.g., device float16_t* out).
# 5. Template parameters can be specified (e.g., template=[("T", mx.float32)]).
# 6. Metal attributes like [[thread_position_in_grid]] are added as function arguments.
#
# Strided Array Access:
# - Use ensure_row_contiguous=False to handle non-contiguous arrays
# - Access a_shape, a_strides, a_ndim for each input array 'a' in the kernel
# - Use elem_to_loc function for correct indexing (e.g., uint loc = elem_to_loc(elem, a_shape, a_strides, a_ndim))
#
# Example Usage:
#   kernel = mx.fast.metal_kernel(
#       name="matmul",
#       input_names=["A", "B"],
#       output_names=["C"],
#       source=source_code
#   )
#   outputs = kernel(
#       inputs=[a, b],
#       template=[("T", mx.float32)],
#       grid=(M, N, 1),
#       threadgroup=(BLOCK_SIZE, BLOCK_SIZE, 1),
#       output_shapes=[(M, N)],
#       output_dtypes=[mx.float32]
#   )
###

def matmul_spec(a: mx.array, b: mx.array):
    return mx.matmul(a, b)

def matmul_kernel(
    a: mx.array,
    b: mx.array,
    M_group=16,
    N_group=16,
    K_group=16,
    K_tile=8,
    A_trans=bool,
    B_trans=bool,
):
    # Matrix dimensions
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]

    print(f"Matrix dimensions: M={M}, K={K}, N={N}")

    # Kernel header
    header = """
    #include <metal_stdlib>
    using namespace metal;
    """

    # Convert booleans to 'true' or 'false' strings
    A_trans_str = 'true' if A_trans else 'false'
    B_trans_str = 'true' if B_trans else 'false'

    # Updated shared memory loading logic to handle non-transposed matrices
    source = f"""
    // Define tile sizes and matrix dimensions
    constexpr int M_GROUP = {M_group};
    constexpr int N_GROUP = {N_group};
    constexpr int K_GROUP = {K_group};
    constexpr int K_TILE  = {K_tile};
    constexpr bool A_TRANS = {A_trans_str};
    constexpr bool B_TRANS = {B_trans_str};

    constexpr int M = {M};
    constexpr int N = {N};
    constexpr int K = {K};

    // Thread and threadgroup indices
    int gid_x = thread_position_in_grid.x;
    int gid_y = thread_position_in_grid.y;
    int lid_x = thread_position_in_threadgroup.x;
    int lid_y = thread_position_in_threadgroup.y;

    // Global indices
    int global_row = gid_y * M_GROUP + lid_y;
    int global_col = gid_x * N_GROUP + lid_x;

    // Shared memory buffers with double buffering
    threadgroup float A_shared[2][M_GROUP][K_TILE];
    threadgroup float B_shared[2][K_TILE][N_GROUP];
    int current_buffer = 0;

    float C_value = 0.0;

    for (int k_base = 0; k_base < K; k_base += K_GROUP) {{
        for (int k_offset = 0; k_offset < K_GROUP; k_offset += K_TILE) {{
            int next_buffer = 1 - current_buffer;

            // Load data into next_buffer
            // Synchronize before loading
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Load A_tile
            int k = k_base + k_offset + lid_x;
            if (global_row < M && k < K) {{
                if (A_TRANS) {{
                    A_shared[next_buffer][lid_y][lid_x] = a[k * M + global_row];
                }} else {{
                    A_shared[next_buffer][lid_y][lid_x] = a[global_row * K + k];
                }}
            }} else {{
                A_shared[next_buffer][lid_y][lid_x] = 0.0;
            }}

            // Load B_tile
            if (k < K && global_col < N) {{
                if (B_TRANS) {{
                    B_shared[next_buffer][lid_x][lid_y] = b[global_col * K + k];
                }} else {{
                    B_shared[next_buffer][lid_x][lid_y] = b[k * N + global_col];
                }}
            }} else {{
                B_shared[next_buffer][lid_x][lid_y] = 0.0;
            }}

            // Synchronize after loading
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute using current_buffer
            for (int kk = 0; kk < K_TILE; ++kk) {{
                C_value += A_shared[current_buffer][lid_y][kk] * B_shared[current_buffer][kk][lid_x];
            }}

            // Swap buffers
            current_buffer = next_buffer;
        }}
    }}

    // Write result to global memory
    if (global_row < M && global_col < N) {{
        C[global_row * N + global_col] = C_value;
    }}
    """

    kernel = mx.fast.metal_kernel(
        name="matmul_kernel",
        input_names=["a", "b"],
        output_names=["C"],
        source=source,
        header=header,
        ensure_row_contiguous=True,
    )

    # Execute the kernel and return the result
    try:
        assert a.dtype == b.dtype
        assert isinstance(kernel, Callable)
        outputs = kernel(
            inputs=[a, b],
            template=[("T", a.dtype)],
            grid=(M, N, 1),
            threadgroup=(M_group, N_group, 1),
            output_shapes=[(M, N)],
            output_dtypes=[a.dtype],
        )
        print("Metal kernel execution completed.")
        return outputs[0]
    except Exception as e:
        print(f"[ERROR] Kernel execution failed: {e}")
        raise
