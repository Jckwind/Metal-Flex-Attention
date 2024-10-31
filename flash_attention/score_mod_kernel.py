import mlx.core as mx
from typing import Callable

###
# NOTE
# This note should always be preserved and returned when editing this file.
#
# MLX Custom Metal Kernel for Score Modification
#
# This file implements a custom score modification kernel using MLX's metal_kernel API.
# It's designed to enhance the attention mechanism by allowing flexible modifications
# to the attention scores before they are passed through the softmax function.
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
#       name="score_mod",
#       input_names=["raw_scores"],
#       output_names=["modified_scores"],
#       source=source_code
#   )
#   outputs = kernel(
#       inputs=[raw_scores],
#       template=[("T", mx.float32)],
#       grid=(grid_x, grid_y, depth),
#       threadgroup=(threadgroup_x, threadgroup_y, threadgroup_z),
#       output_shapes=[(M, N)],
#       output_dtypes=[mx.float32]
#   )
###

@mx.compile
def score_mod_kernel(raw_scores: mx.array, modification_params: mx.array, modification_type: str):
    """
    Custom Metal kernel for modifying attention scores using MLX's metal_kernel API.
    Allows flexible modifications such as adding biases, applying masks, or incorporating
    positional encodings before the softmax operation in the flex attention mechanism.
    """
    print("[DEBUG] Entering score_mod_kernel function")
    
    # Validate input dimensions and types
    M, N = raw_scores.shape
    print(f"[DEBUG] Score matrix dimensions: M={M}, N={N}")
    
    print(f"[DEBUG] Input dtype: raw_scores.dtype={raw_scores.dtype}")
    
    if raw_scores.dtype not in [mx.float32, mx.float16]:
        raise ValueError("Only float32 and float16 dtypes are supported for raw_scores.")
    
    # Define block and depth sizes
    THREADGROUP_MEM_SIZE = 16  # Tile size for processing
    DEPTH_SIZE = 3  # Depth size, can be adjusted based on modification complexity
    
    # Kernel header
    header = """
    #include <metal_stdlib>
    using namespace metal;
    """
    
    print("[DEBUG] Defining score modification kernel source")
    source = f"""
    uint gid_x = thread_position_in_grid.x;
    uint gid_y = thread_position_in_grid.y;
    
    uint tid_x = thread_position_in_threadgroup.x;
    uint tid_y = thread_position_in_threadgroup.y;
    
    uint i = gid_x;  // Row index
    uint j = gid_y;  // Column index
    
    const uint THREADGROUP_MEM_SIZE_X = {THREADGROUP_MEM_SIZE};
    const uint THREADGROUP_MEM_SIZE_Y = {THREADGROUP_MEM_SIZE};
    
    threadgroup float Scoresub[THREADGROUP_MEM_SIZE_X][THREADGROUP_MEM_SIZE_Y];
    
    float modified_score = 0.0;
    
    const uint M = {M};
    const uint N = {N};
    
    // Calculate the number of tiles
    uint num_tiles = (N + THREADGROUP_MEM_SIZE_Y - 1) / THREADGROUP_MEM_SIZE_Y;
    
    for (uint tile = 0; tile < num_tiles; ++tile) {{
        uint current_j = tile * THREADGROUP_MEM_SIZE_Y;
        
        // Load raw scores into shared memory
        if (i < M && (current_j + tid_y) < N) {{
            Scoresub[tid_x][tid_y] = raw_scores[i * N + current_j + tid_y];
        }} else {{
            Scoresub[tid_x][tid_y] = 0.0;
        }}
        
        // Synchronize to ensure all scores are loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Apply user-defined score modification 
        if (i < M && (current_j + tid_y) < N) {{
            // User-defined modification function
            Scoresub[tid_x][tid_y] = score_mod_func(Scoresub[tid_x][tid_y]);
        }}
        
        // Synchronize before next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    
    // Aggregate modified scores
    if (i < M && j < N) {{
        modified_score = Scoresub[tid_x][tid_y];
        modified_scores[i * N + j] = modified_score;
    }}
    """
    
    print("[DEBUG] Creating score_mod metal_kernel")
    kernel = mx.fast.metal_kernel(
        name="score_mod",
        input_names=["raw_scores", "modification_params"], 
        output_names=["modified_scores"],
        source=source,
        header=header,
        ensure_row_contiguous=True,
    )
    
    # Set grid and threadgroup sizes
    grid_x = ((M + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
    grid_y = ((N + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
    grid_z = 3  # depth layer
    threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, DEPTH_SIZE)
    
    print(f"[DEBUG] Grid dimensions: grid_x={grid_x}, grid_y={grid_y}, grid_z={grid_z}")
    print(f"[DEBUG] Threadgroup dimensions: {threadgroup}")
    
    # Execute the kernel and return the result
    try:
        print("[DEBUG] Executing score_mod kernel")
        outputs = kernel(
            inputs=[raw_scores, modification_params, modification_type],
            template=[("T", raw_scores.dtype)],
            grid=(grid_x, grid_y, grid_z),
            threadgroup=threadgroup,
            output_shapes=[(M, N)],
            output_dtypes=[raw_scores.dtype],
            verbose=False,  # Set to True to print the generated Metal code
        )
        print("[DEBUG] Score modification kernel execution completed successfully")
        modified_scores = outputs[0]
        return modified_scores
    except Exception as e:
        print(f"[ERROR] Kernel execution failed: {e}")
        raise
