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
#       grid=(grid_x, grid_y, 1),
#       threadgroup=(threadgroup_x, threadgroup_y, 1),
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
    K_tile=16,
):
    # Matrix dimensions
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]

    print(f"Matrix dimensions: M={M}, K={K}, N={N}")

    # Kernel header
    header = """
    #include <metal_stdlib>
    using namespace metal;
    """

    # Optimized kernel source
    source = f"""
    constexpr int M_GROUP = {M_group};
    constexpr int N_GROUP = {N_group};
    constexpr int K_TILE  = {K_tile};

    constexpr int M = {M};
    constexpr int N = {N};
    constexpr int K = {K};

    // Thread and threadgroup indices
    int gid_x = threadgroup_position_in_grid.x;
    int gid_y = threadgroup_position_in_grid.y;
    int lid_x = thread_position_in_threadgroup.x;
    int lid_y = thread_position_in_threadgroup.y;

    // Global indices
    int global_row = gid_y * M_GROUP + lid_y;
    int global_col = gid_x * N_GROUP + lid_x;

    // Shared memory for tiles
    threadgroup float A_shared[M_GROUP][K_TILE];
    threadgroup float B_shared[K_TILE][N_GROUP];

    // Pre-calculate number of tiles
    int num_tiles = (K + K_TILE - 1) / K_TILE;

    float C_value = 0.0;

    for (int t = 0; t < num_tiles; ++t) {{
        int k_base = t * K_TILE;

        // Load A tile into shared memory
        int a_row = global_row;
        int a_col = k_base + lid_x;
        if (a_row < M && a_col < K) {{
            A_shared[lid_y][lid_x] = a[a_row * K + a_col];
        }} else {{
            A_shared[lid_y][lid_x] = 0.0f;
        }}

        // Load B tile into shared memory
        int b_row = k_base + lid_y;
        int b_col = global_col;
        if (b_row < K && b_col < N) {{
            B_shared[lid_y][lid_x] = b[b_row * N + b_col];
        }} else {{
            B_shared[lid_y][lid_x] = 0.0f;
        }}

        // Synchronize to make sure the tile is loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute the partial product
        for (int k = 0; k < K_TILE; ++k) {{
            C_value += A_shared[lid_y][k] * B_shared[k][lid_x];
        }}

        // Synchronize before loading the next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Write the result to global memory
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

    # Calculate grid dimensions
    grid_x = (N + N_group - 1) // N_group
    grid_y = (M + M_group - 1) // M_group

    # Execute the kernel and return the result
    try:
        assert a.dtype == b.dtype
        assert isinstance(kernel, Callable)
        outputs = kernel(
            inputs=[a, b],
            template=[("T", a.dtype)],
            grid=(grid_x, grid_y, 1),
            threadgroup=(N_group, M_group, 1),
            output_shapes=[(M, N)],
            output_dtypes=[a.dtype],
        )
        print("Metal kernel execution completed.")
        return outputs[0]
    except Exception as e:
        print(f"[ERROR] Kernel execution failed: {e}")
        raise
