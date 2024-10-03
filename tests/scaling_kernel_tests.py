import unittest
import mlx.core as mx
from math import sqrt
import time
from flash_attention.scaling_kernel import scaling_kernel, scaling_spec
from utils.kernel_utils import MetalProblem

class TestScalingKernel(unittest.TestCase):
    def test_scaling_simple(self):
        """Test scaling with a simple small matrix and a known scaling factor."""
        SIZE = 256
        scores = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T
        d_k = 2
        scaling_factor = 1.0 / sqrt(d_k)
        
        
        problem = MetalProblem(
            "Scaling Kernel (Simple)",
            scaling_kernel,
            [scores, scaling_factor],  # Pass scores_flat directly
            scores.shape,
            grid=(3, 3, 1),   # Adjust grid to match number of elements
            threadgroup=(9, 9, 1),         # Use a reasonable threadgroup size
            spec=scaling_spec
        )
        
        problem.show()
        
        self.assertFalse(problem.check(), "Simple scaling test passed")

    def test_scaling_zero_matrix(self):
        """Test scaling of a zero matrix."""
        scores = mx.zeros((4, 4), dtype=mx.float32)
        d_k = 4
        scaling_factor = 1.0 / sqrt(d_k)
        
        problem = MetalProblem(
            "Scaling Kernel (Zero Matrix)",
            scaling_kernel,
            [scores, scaling_factor],  # Pass scores_flat directly
            scores.shape,
            grid=(9, 9, 1),
            threadgroup=(32, 32, 1),
            spec=scaling_spec
        )
        
        problem.show()
        self.assertFalse(problem.check(), "Scaling zero matrix test passed")


    def test_scaling_identity_matrix(self):
        """Test scaling of an identity matrix."""
        SIZE = 5
        scores = mx.eye(SIZE, dtype=mx.float32)
        d_k = SIZE
        scaling_factor = 1.0 / sqrt(d_k)
        
        problem = MetalProblem(
            "Scaling Kernel (Identity Matrix)",
            scaling_kernel,
            [scores, scaling_factor],
            scores.shape,
            grid=(9, 9, 1),  # Use 2D grid
            threadgroup=(32, 32, 1),  # Use 2D threadgroup
            spec=scaling_spec
        )
        
        problem.show()
        self.assertFalse(problem.check(), "Scaling identity matrix test passed")


    def test_scaling_large_matrix(self):
        """Test scaling with a large matrix."""
        SIZE = 512
        scores = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T
        d_k = SIZE
        scaling_factor = 1.0 / sqrt(d_k)
        
        
        problem = MetalProblem(
            "Scaling Kernel (Large Matrix)",
            scaling_kernel,
            [scores, scaling_factor],  
            scores.shape,
            grid=(3, 3, 1),
            threadgroup=(9, 9, 1),
            spec=scaling_spec
        )
        
        problem.show()
        self.assertFalse(problem.check(), "Scaling large matrix test passed")

    def test_benchmark_scaling(self):
        """Benchmark the scaling kernel against MLX's built-in scaling function."""
        SIZE = 512
        NUM_RUNS = 1000
        WARM_UP_RUNS = 100
        d_k = SIZE
        scaling_factor = 1.0 / sqrt(d_k)
        
        scores = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE))
        
        # Benchmark for custom scaling python source kernel
        def benchmark_custom():
            custom_result = scaling_kernel(scores, scaling_factor)
            mx.eval(custom_result)

        # Benchmark for MLX built-in scaling functionality
        @mx.compile
        def benchmark_mlx(s, factor):
            mlx_result = mx.multiply(s, factor)
            return mlx_result
        
        # Warm-up runs
        for _ in range(WARM_UP_RUNS):
            benchmark_custom()
            benchmark_mlx(scores, scaling_factor)
            mx.synchronize()  # Ensure all operations are complete

        # Benchmark custom implementation
        start_time = time.perf_counter()
        for _ in range(NUM_RUNS):
            benchmark_custom()
        mx.synchronize()  # Ensure all operations are complete
        custom_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        # Benchmark MLX built-in scaling
        start_time = time.perf_counter()
        for _ in range(NUM_RUNS):
            benchmark_mlx(scores, scaling_factor)
        mx.synchronize()  # Ensure all operations are complete
        mlx_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        print(f"\nCustom scaling (Medium) time: {custom_time:.3f} ms for {NUM_RUNS} runs")
        print(f"MLX scaling (Medium) time: {mlx_time:.3f} ms for {NUM_RUNS} runs")
        print(f"Speedup: {mlx_time / custom_time:.2f}x\n")

if __name__ == '__main__':
    unittest.main()