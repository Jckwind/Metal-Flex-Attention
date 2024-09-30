import mlx.core as mx
import numpy as np
import time

# Define constants
SEQUENCE_LENGTH = 128  # Length of the input sequence
HEAD_SIZE = 64         # Size of each attention head
BATCH_COUNT = 2
ITERATIONS = 1000      # Number of iterations for benchmarking

# Create random data for our input tensors (Q, K, V)
query_data = np.random.randn(BATCH_COUNT, SEQUENCE_LENGTH, HEAD_SIZE).astype(np.float32)
key_data = np.random.randn(BATCH_COUNT, SEQUENCE_LENGTH, HEAD_SIZE).astype(np.float32)
value_data = np.random.randn(BATCH_COUNT, SEQUENCE_LENGTH, HEAD_SIZE).astype(np.float32)

# Flatten the tensors to 1D arrays for MLX processing
flat_query = query_data.flatten()
flat_key = key_data.flatten()
flat_value = value_data.flatten()

# Convert flattened numpy arrays to MLX arrays
query_mlx = mx.array(flat_query)
key_mlx = mx.array(flat_key)
value_mlx = mx.array(flat_value)

# Simple Metal kernel for basic attention
source = """
using namespace metal;

constexpr uint SEQ_LEN = 128;
constexpr uint HEAD_SIZE = 64;

[[kernel]] void attention(
    device const float* Q [[ buffer(0) ]],
    device const float* K [[ buffer(1) ]],
    device const float* V [[ buffer(2) ]],
    device float* output [[ buffer(3) ]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]]);
    
    // Calculate total sequence length
    uint total_seq = SEQ_LEN * BATCH_COUNT;

    // Calculate the global thread ID
    uint gid = (threadgroup_position_in_grid.x * threads_per_threadgroup.x) + thread_position_in_threadgroup.x;

    if (gid >= total_seq) return;

    // Compute Q_i * K_j^T for all j
    float scores[SEQ_LEN];
    for (uint j = 0; j < SEQ_LEN; ++j) {
        float score = 0.0f;
        for (uint d = 0; d < HEAD_SIZE; ++d) {
            float q = Q[i * HEAD_SIZE + d];
            float k = K[j * HEAD_SIZE + d];
            score += q * k;
        }
        scores[j] = score / sqrt((float)HEAD_SIZE);
    }

    // Compute softmax over scores
    float max_score = scores[0];
    for (uint j = 1; j < SEQ_LEN; ++j) {
        if (scores[j] > max_score) max_score = scores[j];
    }

    float sum_exp = 0.0f;
    for (uint j = 0; j < SEQ_LEN; ++j) {
        scores[j] = exp(scores[j] - max_score);
        sum_exp += scores[j];
    }

    for (uint j = 0; j < SEQ_LEN; ++j) {
        scores[j] /= sum_exp;
    }

    // Compute output_i = sum_j (scores[j] * V_j)
    for (uint d = 0; d < HEAD_SIZE; ++d) {
        float o = 0.0f;
        for (uint j = 0; j < SEQ_LEN; ++j) {
            float v = V[j * HEAD_SIZE + d];
            o += scores[j] * v;
        }
        output[i * HEAD_SIZE + d] = o;
    }

"""

# Create the kernel
kernel = mx.fast.metal_kernel(
    name="attention",
    input_names=["Q", "K", "V"],
    output_names=["output"],
    source=source,
)

def mlx_attention():
    """Performs attention using the MLX (Metal) implementation."""
    result = kernel(
        inputs=[query_mlx, key_mlx, value_mlx],
        output_shapes=[(SEQUENCE_LENGTH * HEAD_SIZE,)],
        output_dtypes=[mx.float32],
        grid=(1, 1, 1),
        threadgroup=(128, 1, 1),
        verbose=False,
    )
    output_array = result[0].reshape(SEQUENCE_LENGTH, HEAD_SIZE)
    return output_array

def numpy_attention():
    """Performs attention using the NumPy implementation."""
    scores = np.matmul(query_data, key_data.T) / np.sqrt(HEAD_SIZE)
    max_scores = np.max(scores, axis=1, keepdims=True)
    scores = np.exp(scores - max_scores)
    weights = scores / np.sum(scores, axis=1, keepdims=True)
    expected_output = np.matmul(weights, value_data)
    return expected_output

def benchmark():
    """Benchmarks both MLX and NumPy attention implementations."""
    # Warm up runs to mitigate any initial overhead
    mlx_attention()
    numpy_attention()

    mlx_times = []
    numpy_times = []

    for _ in range(ITERATIONS):
        # Benchmark MLX Attention
        start_time = time.perf_counter()
        mlx_attention()
        end_time = time.perf_counter()
        mlx_times.append(end_time - start_time)

        # Benchmark NumPy Attention
        start_time = time.perf_counter()
        numpy_attention()
        end_time = time.perf_counter()
        numpy_times.append(end_time - start_time)

    # Calculate average times in milliseconds
    avg_mlx_time = (sum(mlx_times) / ITERATIONS) * 1e3
    avg_numpy_time = (sum(numpy_times) / ITERATIONS) * 1e3

    print(f"Average MLX Attention Time over {ITERATIONS} runs: {avg_mlx_time:.6f} ms")
    print(f"Average NumPy Attention Time over {ITERATIONS} runs: {avg_numpy_time:.6f} ms")

def main():
    """Main function to execute attention and benchmark."""
    # Execute attention once to display results
    output_mlx = mlx_attention()
    print("Metal Kernel Attention Array:", output_mlx)
    print("Metal Kernel Attention Shape:", output_mlx.shape)

    expected_output = numpy_attention()
    print("Numpy Attention Array:", expected_output)
    print("Numpy Attention Shape:", expected_output.shape)

    # Perform benchmarking
    benchmark()

if __name__ == "__main__":
    main()