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
#       grid=(grid_x, grid_y, depth),
#       threadgroup=(threadgroup_x, threadgroup_y, threadgroup_z),
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

@mx.compile
def matmul_kernel(a: mx.array, b: mx.array, A_trans: bool, B_trans: bool):
    """
    Optimized Custom Metal kernel for matrix multiplication using MLX's metal_kernel API with 3D tiling.
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
    
    # Ensure we use float32 or float16 for operations
    if a.dtype not in [mx.float32, mx.float16] or b.dtype not in [mx.float32, mx.float16]:
        raise ValueError("Only float32 and float16 dtypes are supported.")
    
    # Adjust block size (tile size) and depth size
    THREADGROUP_MEM_SIZE = 16  # Increased tile size
    DEPTH_SIZE = 3  # Reduced DEPTH_SIZE to align with threadgroup_z=4
    
    # Kernel header
    header = """
    #include <metal_stdlib>
    using namespace metal;
    """
    
    print("[DEBUG] Defining optimized kernel source with 3D tiling")
    source = f"""
    uint gid_x = thread_position_in_grid.x;
    uint gid_y = thread_position_in_grid.y;
    uint gid_z = thread_position_in_grid.z;  // Added: z-dimension for depth
    
    uint tid_x = thread_position_in_threadgroup.x;
    uint tid_y = thread_position_in_threadgroup.y;
    uint tid_z = thread_position_in_threadgroup.z;  // Added: z-dimension for depth
    
    uint i = gid_x;  // Row index
    uint j = gid_y;  // Column index
    uint d = gid_z;  // Depth index
    
    const uint THREADGROUP_MEM_SIZE_X = {THREADGROUP_MEM_SIZE};
    const uint THREADGROUP_MEM_SIZE_Y = {THREADGROUP_MEM_SIZE};
    const uint THREADGROUP_MEM_SIZE_Z = {DEPTH_SIZE};  // Separate threadgroup memory size for depth
    
    threadgroup float Asub[THREADGROUP_MEM_SIZE_X][THREADGROUP_MEM_SIZE_Y][THREADGROUP_MEM_SIZE_Z];  // 3D shared memory for A
    threadgroup float Bsub[THREADGROUP_MEM_SIZE_Y][THREADGROUP_MEM_SIZE_Z][THREADGROUP_MEM_SIZE_X];  // 3D shared memory for B
    
    float sum = 0.0;
    
    const uint M = {M};
    const uint N = {N};
    const uint K = {K};
    
    const bool A_TRANS = {'true' if A_trans else 'false'};
    const bool B_TRANS = {'true' if B_trans else 'false'};
    
    // Calculate the number of tiles
    uint num_tiles = (K + THREADGROUP_MEM_SIZE_Z - 1) / THREADGROUP_MEM_SIZE_Z;  // Use THREADGROUP_MEM_SIZE_Z for depth tiling
    
    for (uint tile = 0; tile < num_tiles; ++tile) {{
        uint current_k = tile * THREADGROUP_MEM_SIZE_Z;
    
        // Load A sub-matrix into shared memory
        if (i < M && (current_k + tid_z) < K) {{
            uint a_index;
            if (A_TRANS) {{
                a_index = (current_k + tid_z) * M + i;
            }} else {{
                a_index = i * K + (current_k + tid_z);
            }}
            Asub[tid_x][tid_y][tid_z] = a[a_index];
        }} else {{
            Asub[tid_x][tid_y][tid_z] = 0.0;
        }}
    
        // Load B sub-matrix into shared memory
        if (j < N && (current_k + tid_z) < K) {{
            uint b_index;
            if (B_TRANS) {{
                b_index = j * K + (current_k + tid_z);
            }} else {{
                b_index = (current_k + tid_z) * N + j;
            }}
            Bsub[tid_y][tid_z][tid_x] = b[b_index];
        }} else {{
            Bsub[tid_y][tid_z][tid_x] = 0.0;
        }}
    
        // Synchronize to make sure the sub-matrices are loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);
    
        // Multiply the two matrices together
        for (uint k = 0; k < THREADGROUP_MEM_SIZE_Z; ++k) {{
            sum += Asub[tid_x][tid_y][k] * Bsub[tid_y][k][tid_x];
        }}
    
        // Synchronize to make sure that computation is done before loading new sub-matrices
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    
    // Write the result
    if (i < M && j < N) {{
        C[i * N + j] = sum;
    }}
    """
    
    # Create a unique kernel name based on input shapes and transposition flags
    kernel_name = f"matmul_kernel_{M}_{K}_{N}_{'A_T' if A_trans else 'A_N'}_{'B_T' if B_trans else 'B_N'}"
    
    print("[DEBUG] Creating optimized metal_kernel with unique name:", kernel_name)
    kernel = mx.fast.metal_kernel(
        name=kernel_name,
        input_names=["a", "b"],
        output_names=["C"],
        source=source,
        header=header,
        ensure_row_contiguous=True,
    )
    
    # Set grid and threadgroup sizes
    grid_x = ((M + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
    grid_y = ((N + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
    grid_z = DEPTH_SIZE  # Correct grid_z based on DEPTH_SIZE
    threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, DEPTH_SIZE)  # threadgroup_z aligns with DEPTH_SIZE
    
    print(f"[DEBUG] Grid dimensions: grid_x={grid_x}, grid_y={grid_y}, grid_z={grid_z}")
    print(f"[DEBUG] Threadgroup dimensions: {threadgroup}")
    
    # Execute the kernel and return the result
    try:
        print("[DEBUG] Asserting input dtypes")
        assert a.dtype == b.dtype, f"Input dtypes do not match: a.dtype={a.dtype}, b.dtype={b.dtype}"
        assert a.dtype in [mx.float32, mx.float16], "Only float32 and float16 dtypes are supported."
        assert isinstance(kernel, Callable), "Kernel is not callable"
        
        print("[DEBUG] Executing optimized kernel")
        outputs = kernel(
            inputs=[a, b],
            template=[("T", a.dtype)],
            grid=(grid_x, grid_y, grid_z),
            threadgroup=threadgroup,
            output_shapes=[(M, N)],
            output_dtypes=[a.dtype],
            verbose=False,  # Set to True to print the generated Metal code
        )
        print("[DEBUG] Optimized metal kernel execution completed successfully")
        C = outputs[0]
        return C
    except AssertionError as ae:
        print(f"[ASSERTION ERROR] {ae}")
        raise
    except Exception as e:
        print(f"[ERROR] Kernel execution failed: {e}")
        print(f"[DEBUG] Error details: {type(e).__name__}: {str(e)}")
        raise
