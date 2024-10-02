import unittest
import mlx.core as mx
import time
from flash_attention.matmul_kernel import matmul_kernel, matmul_spec
from utils.kernel_utils import MetalProblem

class TestMatmulKernel(unittest.TestCase):
    def test_matmul_simple(self):
        SIZE = 2
        a = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE))
        b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T
        output_shape = (SIZE, SIZE)

        problem = MetalProblem(
            "Matrix Multiplication (Simple)",
            matmul_kernel,
            [a, b], 
            output_shape,
            grid=(3,3,1), 
            threadgroup=(3,3,1),
            spec=matmul_spec
        )

        problem.show()
        self.assertFalse(problem.check(), "Simple matmul test passed")

    def test_matmul_full(self):
        SIZE = 8
        a = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE))
        b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T
        output_shape = (SIZE, SIZE)

        problem = MetalProblem(
            "Matrix Multiplication (Full)",
            matmul_kernel,
            [a, b], 
            output_shape,
            grid=(9,9,1), 
            threadgroup=(3,3,1),
            spec=matmul_spec
        )
        
        problem.show()
        self.assertFalse(problem.check(), "Full matmul test passed")

    def test_matmul_zero_matrix(self):
        SIZE = 4
        a = mx.zeros((SIZE, SIZE), dtype=mx.float32)
        b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T
        output_shape = (SIZE, SIZE)

        problem = MetalProblem(
            "Matrix Multiplication (Zero Matrix)",
            matmul_kernel,
            [a, b],
            output_shape,
            grid=(9, 9, 1),
            threadgroup=(3, 3, 1),
            spec=matmul_spec
        )

        problem.show()
        self.assertFalse(problem.check(), "Zero matrix matmul test passed")

    def test_matmul_identity_matrix(self):
        SIZE = 5
        a = mx.eye(SIZE, dtype=mx.float32)
        b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE))
        output_shape = (SIZE, SIZE)

        problem = MetalProblem(
            "Matrix Multiplication (Identity Matrix)",
            matmul_kernel,
            [a, b],
            output_shape,
            grid=(9, 9, 1),
            threadgroup=(3, 3, 1),
            spec=matmul_spec
        )

        problem.show()
        self.assertFalse(problem.check(), "Identity matrix matmul test passed")

    def test_benchmark_matmul(self):
        SIZE = 512  # Increased size for more realistic benchmarking
        NUM_RUNS = 1000
        WARM_UP_RUNS = 10  # Reduced warm-up runs for larger size

        a = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE))
        b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T

        # benchmark for custom matmul python source kernel
        def benchmark_custom():
            custom_result = matmul_kernel(a, b)
            mx.eval(custom_result)

        # benchmark for mlx built in matmul() function
        def benchmark_mlx():
            mlx_result = mx.matmul(a, b, stream=mx.gpu)
            mx.eval(mlx_result)

        # Warm-up runs
        for _ in range(WARM_UP_RUNS):
            benchmark_custom()
            benchmark_mlx()
            mx.synchronize()  # Ensure all operations are complete

        # Benchmark custom implementation
        start_time = time.perf_counter()
        for _ in range(NUM_RUNS):
            benchmark_custom()
        mx.synchronize()  # Ensure all operations are complete
        custom_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        # Benchmark MLX built-in matmul
        start_time = time.perf_counter()
        for _ in range(NUM_RUNS):
            benchmark_mlx()
        mx.synchronize()  # Ensure all operations are complete
        mlx_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        print(f"Custom matmul time: {custom_time:.3f} ms")
        print(f"MLX matmul time: {mlx_time:.3f} ms")
        print(f"Speedup: {mlx_time / custom_time:.2f}x")

if __name__ == '__main__':
    unittest.main()
