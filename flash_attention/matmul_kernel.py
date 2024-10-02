import mlx.core as mx
from utils.kernel_utils import MetalKernel  

def matmul_spec(a: mx.array, b: mx.array):
    return mx.matmul(a,b)

def matmul_kernel(a: mx.array, b: mx.array):
    header = """
        constant uint THREADGROUP_MEM_SIZE = 3;
    """

    source = """
        threadgroup float a_shared[THREADGROUP_MEM_SIZE][THREADGROUP_MEM_SIZE];
        threadgroup float b_shared[THREADGROUP_MEM_SIZE][THREADGROUP_MEM_SIZE];

        uint i = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
        uint j = threadgroup_position_in_grid.y * threads_per_threadgroup.y + thread_position_in_threadgroup.y;

        uint local_i = thread_position_in_threadgroup.x;
        uint local_j = thread_position_in_threadgroup.y;

        float sum = 0.0;
        
        for (uint tile = 0; tile < (a_shape[1] + THREADGROUP_MEM_SIZE - 1) / THREADGROUP_MEM_SIZE; ++tile) {
            // Load a tile of a and b into shared memory
            if (i < a_shape[0] && tile * THREADGROUP_MEM_SIZE + local_j < a_shape[1]) {
                a_shared[local_i][local_j] = a[i * a_shape[1] + tile * THREADGROUP_MEM_SIZE + local_j];
            } else {
                a_shared[local_i][local_j] = 0.0;
            }
            
            if (j < b_shape[1] && tile * THREADGROUP_MEM_SIZE + local_i < b_shape[0]) {
                b_shared[local_i][local_j] = b[(tile * THREADGROUP_MEM_SIZE + local_i) * b_shape[1] + j];
            } else {
                b_shared[local_i][local_j] = 0.0;
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Compute partial dot product for this tile
            for (uint k = 0; k < THREADGROUP_MEM_SIZE; ++k) {
                sum += a_shared[local_i][k] * b_shared[k][local_j];
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // Write result to output
        if (i < a_shape[0] && j < b_shape[1]) {
            out[i * b_shape[1] + j] = sum;
        }
    """

    kernel = MetalKernel(
        name="matmul",
        input_names=["a", "b"],
        output_names=["out"],
        header=header,
        source=source,
    )

    return kernel