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

    print(f"Matrix dimensions: M={M}, K={K}, N={N}")
    print(f"Transpose flags: A_trans={A_trans}, B_trans={B_trans}")

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

    // Allocate shared memory for A and B tiles
    threadgroup float A_shared[M_GROUP][K_GROUP];
    threadgroup float B_shared[K_GROUP][N_GROUP];

    // Initialize the output value
    float C_value = 0.0;

    // Loop over tiles of K dimension
    for (int k_base = 0; k_base < K; k_base += K_GROUP) {{

        // Synchronize threads before loading new tiles
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Initialize shared memory to 0.0 before loading
        for (int i = 0; i < M_GROUP; ++i) {{
            for (int j = 0; j < K_GROUP; ++j) {{
                A_shared[i][j] = 0.0;
            }}
        }}

        for (int i = 0; i < K_GROUP; ++i) {{
            for (int j = 0; j < N_GROUP; ++j) {{
                B_shared[i][j] = 0.0;
            }}
        }}


        // Load tiles into shared memory
        int tiled_k = k_base + lid_x;

        // Load A tile
        if ((row < M) && (tiled_k < K)) {{
            if (A_TRANS) {{
                A_shared[lid_y][lid_x] = a[tiled_k * M + row];
            }} else {{
                A_shared[lid_y][lid_x] = a[row * K + tiled_k];
            }}
        }}

        // Load B tile
        if ((tiled_k < K) && (col < N)) {{
            if (B_TRANS) {{
                B_shared[lid_x][lid_y] = b[col * K + tiled_k];
            }} else {{
                B_shared[lid_x][lid_y] = b[tiled_k * N + col];
            }}
        }}


        // Synchronize to make sure the tiles are loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute the number of valid k elements in this tile (KEEP THIS)
        int num_valid_k = min(K_GROUP, K - k_base);

        // Compute the partial product for the current tile
        for (int k = 0; k < num_valid_k; ++k) {{
            C_value += A_shared[lid_y][k] * B_shared[k][lid_x];
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

    print("Executing Metal kernel...")
    print("Metal kernel execution completed.")
    print(f"Output Kernel: {kernel}")
    return kernel