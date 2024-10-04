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
    K_a = a.shape[0] if A_trans else a.shape[1]
    K_b = b.shape[1] if B_trans else b.shape[0]
    N = b.shape[0] if B_trans else b.shape[1]

    assert K_a == K_b, "Inner dimensions of A and B must match."
    K = K_a  # Inner dimension

    # Define the kernel header with transpose flags
    header_macros = f"""
#define A_TRANSPOSED {1 if A_trans else 0}
#define B_TRANSPOSED {1 if B_trans else 0}
"""

    header = (
        header_macros
        + """
#include <metal_stdlib>
using namespace metal;

// Utility function for indexing
inline size_t get_index(
    int row, int col,
    constant size_t* strides
) {
    // Stride-based indexing
    return row * strides[0] + col * strides[1];
}
"""
    )

    # Kernel source (function body)
    source = f"""
// Define tile sizes
constexpr int M_GROUP = {M_group};
constexpr int N_GROUP = {N_group};
constexpr int K_GROUP = {K_group};

// Matrix dimensions
constexpr int M = {M};
constexpr int N = {N};
constexpr int K = {K};

// Thread and threadgroup indices
uint group_id_x = threadgroup_position_in_grid.x;
uint group_id_y = threadgroup_position_in_grid.y;
uint local_id_x = thread_position_in_threadgroup.x;
uint local_id_y = thread_position_in_threadgroup.y;

// Compute global row and column indices
int row = group_id_y * M_GROUP + local_id_y;
int col = group_id_x * N_GROUP + local_id_x;

// Initialize the output value
float C_value = 0.0;

// Allocate shared memory for A and B tiles
threadgroup float A_shared[M_GROUP][K_GROUP];
threadgroup float B_shared[K_GROUP][N_GROUP];

// Loop over tiles of K dimension
for (int k_base = 0; k_base < K; k_base += K_GROUP) {{
    // Handle edge cases for K
    int K_tile_size = min((int)K_GROUP, K - k_base);

    // Load A into shared memory
    int a_row = A_TRANSPOSED ? k_base + local_id_x : row;
    int a_col = A_TRANSPOSED ? row : k_base + local_id_x;

    float a_val = 0.0;
    if (a_row < A_shape[0] && a_col < A_shape[1]) {{
        size_t idx = get_index(a_row, a_col, A_strides);
        a_val = A[idx];
    }}
    A_shared[local_id_y][local_id_x] = a_val;

    // Load B into shared memory
    int b_row = B_TRANSPOSED ? col : k_base + local_id_y;
    int b_col = B_TRANSPOSED ? k_base + local_id_y : col;

    float b_val = 0.0;
    if (b_row < B_shape[0] && b_col < B_shape[1]) {{
        size_t idx = get_index(b_row, b_col, B_strides);
        b_val = B[idx];
    }}
    B_shared[local_id_y][local_id_x] = b_val;

    // Synchronize to make sure the tiles are loaded
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute the dot product for the current tile
    for (int k = 0; k < K_tile_size; ++k) {{
        C_value += A_shared[local_id_y][k] * B_shared[k][local_id_x];
    }}

    // Synchronize before loading the next tile
    threadgroup_barrier(mem_flags::mem_threadgroup);
}}

// Write the result back to global memory
if (row < M && col < N) {{
    size_t idx = row * N + col;
    C[idx] = C_value;
}}
"""

    kernel = mx.fast.metal_kernel(
        name="matmul_kernel",
        input_names=["A", "B"],
        output_names=["C"],
        source=source,
        header=header,
        ensure_row_contiguous=False,
    )

    return kernel
