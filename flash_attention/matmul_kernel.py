import mlx.core as mx


def matmul_spec(a: mx.array, b: mx.array):
    return mx.matmul(a, b)


def matmul_kernel(
    a: mx.array,
    b: mx.array,
    M_group=16,
    N_group=16,
    K_group=16,
    A_trans=False,
    B_trans=False,
):
    # Matrix dimensions
    M = a.shape[1] if A_trans else a.shape[0]
    K = a.shape[0] if A_trans else a.shape[1]
    N = b.shape[0] if B_trans else b.shape[1]

    assert K == (b.shape[1] if B_trans else b.shape[0]), "Inner dimensions of A and B must match."

    # Kernel header with utility functions
    header = """
    #include <metal_stdlib>
    using namespace metal;

    // Ceil division utility
    inline int ceil_div(int a, int b) {
        return (a + b - 1) / b;
    }
    """

    # Modified kernel source
    source = f"""
    // Define tile sizes
    constexpr int M_GROUP = {M_group};
    constexpr int N_GROUP = {N_group};
    constexpr int K_GROUP = {K_group};

    // Matrix dimensions
    constexpr int M = {M};
    constexpr int N = {N};
    constexpr int K = {K};

    // Transpose flags
    constexpr bool A_TRANS = {str(A_trans).lower()};
    constexpr bool B_TRANS = {str(B_trans).lower()};

    // Thread and threadgroup indices
    int gid_x = thread_position_in_grid.x;
    int gid_y = thread_position_in_grid.y;
    int lid_x = thread_position_in_threadgroup.x;
    int lid_y = thread_position_in_threadgroup.y;

    // Compute global row and column indices
    int row = gid_y * M_GROUP + lid_y;
    int col = gid_x * N_GROUP + lid_x;

    // Allocate shared memory for A and B tiles with double buffering
    threadgroup float A_shared[2][M_GROUP][K_GROUP];
    threadgroup float B_shared[2][K_GROUP][N_GROUP];

    // Buffers for double buffering
    int curr_buffer = 0;
    int next_buffer = 1;

    // Initialize the output value
    float C_value = 0.0;

    // Loop over tiles of K dimension
    for (int k_base = 0; k_base < K; k_base += K_GROUP) {{

        // Load A and B tiles into shared memory collaboratively
        // Switch buffers for double buffering
        curr_buffer = next_buffer;
        next_buffer = 1 - curr_buffer;

        // Synchronize threads before loading new tiles
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load tiles into shared memory
        int tiled_k = k_base + lid_x;
        if ((row < M) && (tiled_k < K)) {{
            A_shared[curr_buffer][lid_y][lid_x] = A_TRANS ? a[tiled_k * a_strides[0] + row * a_strides[1]] : a[row * a_strides[0] + tiled_k * a_strides[1]];
        }} else {{
            A_shared[curr_buffer][lid_y][lid_x] = 0.0;
        }}

        if ((tiled_k < K) && (col < N)) {{
            B_shared[curr_buffer][lid_x][lid_y] = B_TRANS ? b[col * b_strides[0] + tiled_k * b_strides[1]] : b[tiled_k * b_strides[0] + col * b_strides[1]];
        }} else {{
            B_shared[curr_buffer][lid_x][lid_y] = 0.0;
        }}

        // Synchronize to make sure the tiles are loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute the partial product for the current tile
        for (int k = 0; k < K_GROUP; ++k) {{
            C_value += A_shared[curr_buffer][lid_y][k] * B_shared[curr_buffer][k][lid_x];
        }}
    }}

    // Write the result back to global memory
    if ((row < M) && (col < N)) {{
        size_t idx = row * N + col;
        C[idx] = C_value;
    }}
    """

    kernel = mx.fast.metal_kernel(
        name="matmul_kernel",
        input_names=["a", "b"],
        output_names=["C"],
        source=source,
        header=header,
        ensure_row_contiguous=False,
    )

    return kernel
