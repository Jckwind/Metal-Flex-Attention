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

@mx.compile
def matmul_spec(a: mx.array, b: mx.array):
    return mx.matmul(a,b)

@mx.compile
def matmul_kernel(
    a: mx.array,
    b: mx.array,
    A_trans: bool,
    B_trans: bool,
):
    """
    Custom Metal kernel for matrix multiplication using MLX's metal_kernel API.
    """
    print("[DEBUG] Entering matmul_kernel function")
    
    # Validate input dimensions
    ashape0 = a.shape[0]
    ashape1 = a.shape[1]
    bshape0 = b.shape[0]
    bshape1 = b.shape[1]
    assert ashape1 == bshape0, f"Inner dimensions must match: ashape1={ashape1}, bshape0={bshape0}"
    
    print(f"[DEBUG] Transposition: A_trans={A_trans}, B_trans={B_trans}")
    print(f"[DEBUG] Input dtypes: a.dtype={a.dtype}, b.dtype={b.dtype}")

    # Define block size (tile size)
    THREADGROUP_MEM_SIZE = 3  # Ensure consistency with tests
    DEPTH_SIZE = 8  # New dimension for 3D tiling

    # Kernel header
    header = """
    #include <metal_stdlib>
    using namespace metal;
    """

    print("[DEBUG] Defining kernel source with 3D tiling")
    # Updated kernel source code with 3D tiling
    source = f"""
    uint gid_x = thread_position_in_grid.x;
    uint gid_y = thread_position_in_grid.y;
    uint gid_z = thread_position_in_grid.z; // Added z-axis for 3D tiling

    uint tid_x = thread_position_in_threadgroup.x;
    uint tid_y = thread_position_in_threadgroup.y;
    uint tid_z = thread_position_in_threadgroup.z; // Added z-axis for 3D tiling

    uint i = gid_x;
    uint j = gid_y;
    uint z = gid_z;

    const uint THREADGROUP_MEM_SIZE = {THREADGROUP_MEM_SIZE};
    const uint DEPTH_SIZE = {DEPTH_SIZE};

    // Shared memory allocations for 3D tiling
    threadgroup float Asub[DEPTH_SIZE][THREADGROUP_MEM_SIZE][THREADGROUP_MEM_SIZE];
    threadgroup float Bsub[DEPTH_SIZE][THREADGROUP_MEM_SIZE][THREADGROUP_MEM_SIZE];

    float sum = 0.0;

    const uint a_shape0 = {ashape0};
    const uint a_shape1 = {ashape1};
    const uint b_shape0 = {bshape0};
    const uint b_shape1 = {bshape1};
    const bool A_TRANS = {'true' if A_trans else 'false'};
    const bool B_TRANS = {'true' if B_trans else 'false'};

    // Calculate the number of tiles in each dimension
    uint tiles_x = (a_shape0 + DEPTH_SIZE * THREADGROUP_MEM_SIZE - 1) / (DEPTH_SIZE * THREADGROUP_MEM_SIZE);
    uint tiles_y = (b_shape1 + DEPTH_SIZE * THREADGROUP_MEM_SIZE - 1) / (DEPTH_SIZE * THREADGROUP_MEM_SIZE);
    uint tiles_z = (a_shape1 + DEPTH_SIZE * THREADGROUP_MEM_SIZE - 1) / (DEPTH_SIZE * THREADGROUP_MEM_SIZE);

    for (uint tile = 0; tile < tiles_z; ++tile) {{
        uint current_depth = tile * DEPTH_SIZE;

        for (uint d = 0; d < DEPTH_SIZE; ++d) {{
            if (current_depth + d < a_shape1) {{
                // Load a tile of A into shared memory
                uint a_col = current_depth + d;
                if (i < a_shape0 && a_col < a_shape1) {{
                    Asub[d][tid_y][tid_x] = A_TRANS ? a[a_col * a_shape0 + i] : a[i * a_shape1 + a_col];
                }} else {{
                    Asub[d][tid_y][tid_x] = 0.0;
                }}

                // Load a tile of B into shared memory
                uint b_row = current_depth + d;
                if (b_row < b_shape0 && j < b_shape1) {{
                    Bsub[d][tid_y][tid_x] = B_TRANS ? b[j * b_shape0 + b_row] : b[b_row * b_shape1 + j];
                }} else {{
                    Bsub[d][tid_y][tid_x] = 0.0;
                }}
            }} else {{
                Bsub[d][tid_x][tid_y] = 0.0;
            }}   

        }}

        // Synchronize to ensure all tiles are loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Perform multiplication for each depth slice
        for (uint d = 0; d < DEPTH_SIZE; ++d) {{
            sum += Asub[d][tid_y][tid_x] * Bsub[d][tid_y][tid_x];
        }}
    }}

    // Write the result to C
    if (i < a_shape0 && j < b_shape1) {{
        C[i * b_shape1 + j + z] = sum;
    }}
    """
    
    print("[DEBUG] Creating metal_kernel")
    kernel = mx.fast.metal_kernel(
        name="matmul_kernel",
        input_names=["a", "b"],
        output_names=["C"],
        source=source,
        header=header,
        ensure_row_contiguous=True,
    )

    # Set grid and threadgroup sizes
    grid_x = (bshape1 + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
    grid_y = (ashape0 + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
    grid = (grid_x, grid_y, 1)
    threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, 3)
    print(f"[DEBUG] Grid dimensions: grid_x={grid_x}, grid_y={grid_y}")
    
    # Execute the kernel and return the result
    try:
        print("[DEBUG] Asserting input dtypes")
        assert a.dtype == b.dtype, f"Input dtypes do not match: a.dtype={a.dtype}, b.dtype={b.dtype}"
        assert isinstance(kernel, Callable), "Kernel is not callable"
        
        print("[DEBUG] Executing kernel")
        outputs = kernel(
            inputs=[a, b],
            template=[("T", a.dtype)],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[(ashape0, bshape1)],
            output_dtypes=[a.dtype],
            verbose=False,  # Set to True to print the generated Metal code
        )
        print("[DEBUG] Metal kernel execution completed successfully")
        C = outputs[0]
        return C
    except Exception as e:
        print(f"[ERROR] Kernel execution failed: {e}")
        print(f"[DEBUG] Error details: {type(e).__name__}: {str(e)}")
        raise

print("[DEBUG] matmul_kernel function defined")
