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
def matmul_spec(a: mx.array, b: mx.array, A_trans: bool, B_trans: bool):
    if A_trans:
        a = mx.transpose(a)
    if B_trans:
        b = mx.transpose(b)
    return mx.matmul(a, b)

# @mx.compile
# def matmul_kernel(
#     a: mx.array,
#     b: mx.array,
#     A_trans: bool,
#     B_trans: bool,
# ):
#     """
#     Custom Metal kernel for matrix multiplication using MLX's metal_kernel API.
#     """
#     print("[DEBUG] Entering matmul_kernel function")
    
#     # Validate input dimensions
#     if A_trans:
#         M = a.shape[1]
#         K = a.shape[0]
#     else:
#         M = a.shape[0]
#         K = a.shape[1]

#     if B_trans:
#         N = b.shape[0]
#         Kb = b.shape[1]
#     else:
#         N = b.shape[1]
#         Kb = b.shape[0]
    
#     assert K == Kb, f"Inner dimensions must match: K={K}, Kb={Kb}"
    
#     print(f"[DEBUG] Transposition: A_trans={A_trans}, B_trans={B_trans}")
#     print(f"[DEBUG] Input dtypes: a.dtype={a.dtype}, b.dtype={b.dtype}")
    
#     # Ensure we use float32 for atomic operations
#     if a.dtype != mx.float32 or b.dtype != mx.float32:
#         raise ValueError("Atomic operations require float32 data type.")
    
#     # Define block size (tile size)
#     THREADGROUP_MEM_SIZE = 8
#     DEPTH_SIZE = 8
    
#     # Kernel header
#     header = """
#     #include <metal_stdlib>
#     using namespace metal;
#     """
    
#     print("[DEBUG] Defining kernel source with conditional indexing")
#     source = f"""
#     uint gid_x = thread_position_in_grid.x;
#     uint gid_y = thread_position_in_grid.y;

#     uint tid_x = thread_position_in_threadgroup.x;
#     uint tid_y = thread_position_in_threadgroup.y;

#     uint i = gid_x;  // Row index
#     uint j = gid_y;  // Column index

#     const uint THREADGROUP_MEM_SIZE = {THREADGROUP_MEM_SIZE};
#     const uint DEPTH_SIZE = {DEPTH_SIZE};

#     threadgroup float Asub[THREADGROUP_MEM_SIZE][DEPTH_SIZE];
#     threadgroup float Bsub[DEPTH_SIZE][THREADGROUP_MEM_SIZE];

#     float sum = 0.0;

#     const uint M = {M};
#     const uint N = {N};
#     const uint K = {K};

#     const bool A_TRANS = {'true' if A_trans else 'false'};
#     const bool B_TRANS = {'true' if B_trans else 'false'};

#     // Calculate the number of tiles
#     uint num_tiles = (K + DEPTH_SIZE - 1) / DEPTH_SIZE;

#     for (uint tile = 0; tile < num_tiles; ++tile) {{
#         uint current_k = tile * DEPTH_SIZE;

#         // Load A sub-matrix into shared memory
#         if (i < M && current_k + tid_y < K) {{
#             uint a_index;
#             if (A_TRANS) {{
#                 a_index = (current_k + tid_y) * M + i;
#             }} else {{
#                 a_index = i * K + (current_k + tid_y);
#             }}
#             Asub[tid_x][tid_y] = a[a_index];
#         }} else {{
#             Asub[tid_x][tid_y] = 0.0;
#         }}

#         // Load B sub-matrix into shared memory
#         if (j < N && current_k + tid_x < K) {{
#             uint b_index;
#             if (B_TRANS) {{
#                 b_index = j * K + (current_k + tid_x);
#             }} else {{
#                 b_index = (current_k + tid_x) * N + j;
#             }}
#             Bsub[tid_x][tid_y] = b[b_index];
#         }} else {{
#             Bsub[tid_x][tid_y] = 0.0;
#         }}

#         // Synchronize to make sure the sub-matrices are loaded
#         threadgroup_barrier(mem_flags::mem_threadgroup);

#         // Multiply the two matrices together
#         for (uint k = 0; k < DEPTH_SIZE; ++k) {{
#             sum += Asub[tid_x][k] * Bsub[k][tid_y];
#         }}

#         // Synchronize to make sure that computation is done before loading new sub-matrices
#         threadgroup_barrier(mem_flags::mem_threadgroup);
#     }}

#     // Write the result
#     if (i < M && j < N) {{
#         C[i * N + j] = sum;
#     }}
#     """
    
#     # Create a unique kernel name based on input shapes and transposition flags
#     kernel_name = f"matmul_kernel_{M}_{K}_{N}_{'A_T' if A_trans else 'A_N'}_{'B_T' if B_trans else 'B_N'}"
    
#     print("[DEBUG] Creating metal_kernel with unique name:", kernel_name)
#     kernel = mx.fast.metal_kernel(
#         name=kernel_name,
#         input_names=["a", "b"],
#         output_names=["C"],
#         source=source,
#         header=header,
#         ensure_row_contiguous=True,
#     )
    
#     # Set grid and threadgroup sizes
#     grid_x = (M + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
#     grid_y = (N + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
#     grid = (grid_x, grid_y, 3)
#     threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, 3)
    
#     print(f"[DEBUG] Grid dimensions: grid_x={grid_x}, grid_y={grid_y}")
#     print(f"[DEBUG] Threadgroup dimensions: {threadgroup}")
    
#     # Execute the kernel and return the result
#     try:
#         print("[DEBUG] Asserting input dtypes")
#         assert a.dtype == b.dtype, f"Input dtypes do not match: a.dtype={a.dtype}, b.dtype={b.dtype}"
#         assert a.dtype == mx.float32, "Only float32 dtype is supported with atomic operations."
#         assert isinstance(kernel, Callable), "Kernel is not callable"
        
#         print("[DEBUG] Executing kernel")
#         outputs = kernel(
#             inputs=[a, b],
#             template=[("T", a.dtype)],
#             grid=grid,
#             threadgroup=threadgroup,
#             output_shapes=[(M, N)],
#             output_dtypes=[a.dtype],
#             verbose=False,  # Set to True to print the generated Metal code
#         )
#         print("[DEBUG] Metal kernel execution completed successfully")
#         C = outputs[0]
#         return C
#     except AssertionError as ae:
#         print(f"[ASSERTION ERROR] {ae}")
#         raise
#     except Exception as e:
#         print(f"[ERROR] Kernel execution failed: {e}")
#         print(f"[DEBUG] Error details: {type(e).__name__}: {str(e)}")
#         raise

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
    if A_trans:
        M = a.shape[1]
        K = a.shape[0]
    else:
        M = a.shape[0]
        K = a.shape[1]

    if B_trans:
        N = b.shape[0]
        Kb = b.shape[1]
    else:
        N = b.shape[1]
        Kb = b.shape[0]
    
    assert K == Kb, f"Inner dimensions must match: K={K}, Kb={Kb}"
    
    print(f"[DEBUG] Transposition: A_trans={A_trans}, B_trans={B_trans}")
    print(f"[DEBUG] Input dtypes: a.dtype={a.dtype}, b.dtype={b.dtype}")
    
    # Ensure we use float32 for atomic operations
    if a.dtype != mx.float32 or b.dtype != mx.float32:
        raise ValueError("Atomic operations require float32 data type.")
    
    # Define block size (tile size)
    THREADGROUP_MEM_SIZE = 8
    DEPTH_SIZE = 8
    
    # Kernel header
    header = """
    #include <metal_stdlib>
    using namespace metal;
    """
    
    print("[DEBUG] Defining kernel source with conditional indexing")
    source = f"""
    uint gid_x = thread_position_in_grid.x;
    uint gid_y = thread_position_in_grid.y;

    uint tid_x = thread_position_in_threadgroup.x;
    uint tid_y = thread_position_in_threadgroup.y;

    uint i = gid_x;  // Row index
    uint j = gid_y;  // Column index

    const uint THREADGROUP_MEM_SIZE = {THREADGROUP_MEM_SIZE};
    const uint DEPTH_SIZE = {DEPTH_SIZE};

    threadgroup float Asub[THREADGROUP_MEM_SIZE][DEPTH_SIZE];
    threadgroup float Bsub[DEPTH_SIZE][THREADGROUP_MEM_SIZE];

    float sum = 0.0;

    const uint M = {M};
    const uint N = {N};
    const uint K = {K};

    const bool A_TRANS = {'true' if A_trans else 'false'};
    const bool B_TRANS = {'true' if B_trans else 'false'};

    // Calculate the number of tiles
    uint num_tiles = (K + DEPTH_SIZE - 1) / DEPTH_SIZE;

    for (uint tile = 0; tile < num_tiles; ++tile) {{
        uint current_k = tile * DEPTH_SIZE;

        // Load tiles into shared memory with coalesced accesses
        Asub[tid_x][tid_y] = A_TRANS ? a[(current_k + tid_y) * M + i] : a[i * K + (current_k + tid_y)];
        Bsub[tid_x][tid_y] = B_TRANS ? b[j * K + (current_k + tid_x)] : b[(current_k + tid_x) * N + j];
        
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Utilize vectorized operations if possible
        for (uint k = 0; k < DEPTH_SIZE; ++k) {{
            sum += Asub[tid_x][k] * Bsub[k][tid_y];
        }}
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Write the result with coalesced access
    if (i < M && j < N) {{
        C[i * N + j] = sum;
    }}
    """

    # Create a unique kernel name based on input shapes and transposition flags
    kernel_name = f"matmul_kernel_{M}_{K}_{N}_{'A_T' if A_trans else 'A_N'}_{'B_T' if B_trans else 'B_N'}"
    
    print("[DEBUG] Creating metal_kernel with unique name:", kernel_name)
    kernel = mx.fast.metal_kernel(
        name=kernel_name,
        input_names=["a", "b"],
        output_names=["C"],
        source=source,
        header=header,
        ensure_row_contiguous=True,
    )
    
    # Set grid and threadgroup sizes
    grid_x = (M + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
    grid_y = (N + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
    grid = (grid_x, grid_y, 3)
    threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, 3)
    
    print(f"[DEBUG] Grid dimensions: grid_x={grid_x}, grid_y={grid_y}")
    print(f"[DEBUG] Threadgroup dimensions: {threadgroup}")
    
    # Execute the kernel and return the result
    try:
        print("[DEBUG] Asserting input dtypes")
        assert a.dtype == b.dtype, f"Input dtypes do not match: a.dtype={a.dtype}, b.dtype={b.dtype}"
        assert a.dtype == mx.float32, "Only float32 dtype is supported with atomic operations."
        assert isinstance(kernel, Callable), "Kernel is not callable"
        
        print("[DEBUG] Executing kernel")
        outputs = kernel(
            inputs=[a, b],
            template=[("T", a.dtype)],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[(M, N)],
            output_dtypes=[a.dtype],
            verbose=False,  # Set to True to print the generated Metal code
        )
        print("[DEBUG] Metal kernel execution completed successfully")
        C = outputs[0]
        return C
    except AssertionError as ae:
        print(f"[ASSERTION ERROR] {ae}")
        raise
    except Exception as e:
        print(f"[ERROR] Kernel execution failed: {e}")
        print(f"[DEBUG] Error details: {type(e).__name__}: {str(e)}")
        raise