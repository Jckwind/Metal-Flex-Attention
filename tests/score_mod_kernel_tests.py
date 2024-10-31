import unittest
import mlx.core as mx
from flash_attention.score_mod_kernel import score_mod_kernel
import time
import contextlib
from typing import Callable

class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix="", on_exit=None, enabled=True):
        self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled

    def __enter__(self):
        self.st = time.perf_counter_ns()

    def __exit__(self, *exc):
        self.et = time.perf_counter_ns() - self.st
        if self.enabled:
            print(
                f"{self.prefix}{self.et*1e-6:6.2f} ms"
                + (self.on_exit(self.et) if self.on_exit else "")
            )

class ScoreModKernelTest:
    def __init__(
        self,
        name,
        kernel,
        inputs,
        output_shape,
        grid,
        threadgroup,
        modification_type: str,
        spec=None,
        **kernel_kwargs,
    ):
        print(f"[INIT] {name}")
        print(f"       Function: {kernel.__name__ if hasattr(kernel, '__name__') else kernel}")
        print(f"       Inputs shapes: {[inp.shape for inp in inputs]}")
        print(f"       Output shape: {output_shape}")
        print(f"       Grid: {grid}, Threadgroup: {threadgroup}")
        if spec:
            print(f"       Spec: {spec.__name__ if hasattr(spec, '__name__') else spec}")
        if kernel_kwargs:
            print(f"       Kernel kwargs: {kernel_kwargs}")

        self.name = name
        self.kernel = kernel
        self.inputs = inputs
        self.output_shape = output_shape
        self.grid = grid
        self.threadgroup = threadgroup
        self.modification_type = modification_type
        self.spec = spec
        self.kernel_kwargs = kernel_kwargs

    def run_kernel(self):
        print(f"[RUN] {self.name}")
        try:
            modified_scores = self.kernel(
                *self.inputs,
                modification_type=self.modification_type
            )
            return modified_scores
        except Exception as e:
            print(f"       [ERROR] Kernel execution failed: {e}")
            raise

    def check(self):
        print(f"[CHECK] {self.name}")
        if self.spec is None:
            print("       Spec not provided. Skipping verification.")
            return True

        try:
            kernel_output = self.run_kernel()
            spec_output = self.spec(*self.inputs)
            if mx.allclose(kernel_output, spec_output, atol=1e-2):
                print("       Test Passed.")
                return True
            else:
                print("       Test Failed.")
                print(f"         Expected: {spec_output}")
                print(f"         Received: {kernel_output}")
                return False
        except Exception as e:
            print(f"       [ERROR] Check failed: {e}")
            return False

    def show(self):
        print(f"[SHOW] {self.name}")
        self.check()

class TestScoreModKernel(unittest.TestCase):
    def run_kernel_score_mod(self, raw_scores, modification_params, modification_type, grid, threadgroup):
        output_shape = raw_scores.shape
        problem = ScoreModKernelTest(
            "Score Modification Kernel Test",
            score_mod_kernel,
            [raw_scores, modification_params],
            output_shape,
            grid=grid,
            threadgroup=threadgroup,
            modification_type=modification_type,
            spec=self.reference_score_mod
        )
        result = problem.run_kernel()
        return result

    def reference_score_mod(self, raw_scores, modification_params):
        """
        Reference implementation for score_mod_kernel using pure Python.
        Applies the modification based on modification_type.
        """
        modification_type = self.current_modification_type
        if modification_type == "add_bias":
            return raw_scores + modification_params
        elif modification_type == "apply_mask":
            return mx.where(modification_params > 0, raw_scores, 0.0)
        elif modification_type == "scale":
            return raw_scores * modification_params
        elif modification_type == "zero_modification":
            return mx.zeros_like(raw_scores)
        else:
            raise ValueError(f"Unknown modification type: {modification_type}")

    def test_score_mod_add_bias(self):
        print("\n=== test_score_mod_add_bias ===")
        SIZE = 8
        dtype = mx.float32

        a = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE))
        bias = mx.ones((SIZE, SIZE), dtype=dtype) * 0.5  # Bias of 0.5

        modification_type = "add_bias"
        THREADGROUP_MEM_SIZE = 16
        grid_x = ((SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
        grid_y = ((SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
        grid_z = 1
        threadgroup_z = 1

        grid = (grid_x, grid_y, grid_z)
        threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, threadgroup_z)

        print(f"       raw_scores.shape: {a.shape}, modification_params.shape: {bias.shape}, dtype={dtype}")
        print(f"       Grid: {grid}, Threadgroup: {threadgroup}")

        self.current_modification_type = modification_type
        result = self.run_kernel_score_mod(a, bias, modification_type, grid, threadgroup)
        expected = self.reference_score_mod(a, bias)

        self.assertTrue(mx.allclose(result, expected, atol=1e-2), "Add Bias Test Failed.")

    def test_score_mod_apply_mask(self):
        print("\n=== test_score_mod_apply_mask ===")
        SIZE = 8
        dtype = mx.float32

        a = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE))
        mask = mx.ones((SIZE, SIZE), dtype=dtype)
        mask[SIZE//2:, :] = 0.0  # Mask lower half

        modification_type = "apply_mask"
        THREADGROUP_MEM_SIZE = 16
        grid_x = ((SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
        grid_y = ((SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
        grid_z = 1
        threadgroup_z = 1

        grid = (grid_x, grid_y, grid_z)
        threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, threadgroup_z)

        print(f"       raw_scores.shape: {a.shape}, modification_params.shape: {mask.shape}, dtype={dtype}")
        print(f"       Grid: {grid}, Threadgroup: {threadgroup}")

        self.current_modification_type = modification_type
        result = self.run_kernel_score_mod(a, mask, modification_type, grid, threadgroup)
        expected = self.reference_score_mod(a, mask)

        self.assertTrue(mx.allclose(result, expected, atol=1e-2), "Apply Mask Test Failed.")

    def test_score_mod_scale(self):
        print("\n=== test_score_mod_scale ===")
        SIZE = 8
        dtype = mx.float32

        a = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE))
        scale = mx.full((SIZE, SIZE), 2.0, dtype=dtype)  # Scale by 2.0

        modification_type = "scale"
        THREADGROUP_MEM_SIZE = 16
        grid_x = ((SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
        grid_y = ((SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
        grid_z = 1
        threadgroup_z = 1

        grid = (grid_x, grid_y, grid_z)
        threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, threadgroup_z)

        print(f"       raw_scores.shape: {a.shape}, modification_params.shape: {scale.shape}, dtype={dtype}")
        print(f"       Grid: {grid}, Threadgroup: {threadgroup}")

        self.current_modification_type = modification_type
        result = self.run_kernel_score_mod(a, scale, modification_type, grid, threadgroup)
        expected = self.reference_score_mod(a, scale)

        self.assertTrue(mx.allclose(result, expected, atol=1e-2), "Scale Scores Test Failed.")

    def test_score_mod_zero_modification(self):
        print("\n=== test_score_mod_zero_modification ===")
        SIZE = 8
        dtype = mx.float32

        a = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE))
        zero_mod = mx.zeros((SIZE, SIZE), dtype=dtype)

        modification_type = "zero_modification"
        THREADGROUP_MEM_SIZE = 16
        grid_x = ((SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
        grid_y = ((SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
        grid_z = 1
        threadgroup_z = 1

        grid = (grid_x, grid_y, grid_z)
        threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, threadgroup_z)

        print(f"       raw_scores.shape: {a.shape}, modification_params.shape: {zero_mod.shape}, dtype={dtype}")
        print(f"       Grid: {grid}, Threadgroup: {threadgroup}")

        self.current_modification_type = modification_type
        result = self.run_kernel_score_mod(a, zero_mod, modification_type, grid, threadgroup)
        expected = self.reference_score_mod(a, zero_mod)

        self.assertTrue(mx.allclose(result, expected, atol=1e-2), "Zero Modification Test Failed.")

    def test_score_mod_large_matrix(self):
        print("\n=== test_score_mod_large_matrix ===")
        SIZE = 256
        dtype = mx.float32

        a = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE))
        bias = mx.ones((SIZE, SIZE), dtype=dtype) * 1.0  # Bias of 1.0

        modification_type = "add_bias"
        THREADGROUP_MEM_SIZE = 16
        grid_x = ((SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
        grid_y = ((SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
        grid_z = 1
        threadgroup_z = 1

        grid = (grid_x, grid_y, grid_z)
        threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, threadgroup_z)

        print(f"       raw_scores.shape: {a.shape}, modification_params.shape: {bias.shape}, dtype={dtype}")
        print(f"       Grid: {grid}, Threadgroup: {threadgroup}")

        self.current_modification_type = modification_type
        result = self.run_kernel_score_mod(a, bias, modification_type, grid, threadgroup)
        expected = self.reference_score_mod(a, bias)

        self.assertTrue(mx.allclose(result, expected, atol=1e-2), "Large Matrix Add Bias Test Failed.")

    def test_score_mod_benchmark(self):
        print("\n=== test_score_mod_benchmark ===")
        sizes = [64, 128, 256, 512, 1024]
        warm_up_runs = 50
        benchmark_runs = 100  # Adjust as needed for practical execution time

        for size in sizes:
            a = mx.arange(size * size, dtype=mx.float32).reshape((size, size))
            bias = mx.ones((size, size), dtype=mx.float32) * 0.1  # Small bias

            modification_type = "add_bias"
            THREADGROUP_MEM_SIZE = 16
            grid_x = ((size + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
            grid_y = ((size + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
            grid_z = 1
            threadgroup_z = 1

            grid = (grid_x, grid_y, grid_z)
            threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, threadgroup_z)

            print(f"\n--- Benchmarking size: {size}x{size} ---")

            # Warm-up runs
            for _ in range(warm_up_runs):
                modified = score_mod_kernel(a, bias, modification_type, grid, threadgroup)
                expected = self.reference_score_mod(a, bias)
                mx.synchronize()  # Ensure GPU operations are complete

            # Benchmark custom kernel
            start = time.perf_counter_ns()
            for _ in range(benchmark_runs):
                modified = score_mod_kernel(a, bias, modification_type)
            mx.synchronize()  # Ensure GPU operations are complete
            custom_time = (time.perf_counter_ns() - start) / benchmark_runs / 1e6  # ms

            # Benchmark reference implementation
            start = time.perf_counter_ns()
            for _ in range(benchmark_runs):
                expected = self.reference_score_mod(a, bias)
            reference_time = (time.perf_counter_ns() - start) / benchmark_runs / 1e6  # ms

            print(f"Size: {size}x{size} | Score Mod Kernel: {custom_time:.4f} ms | Reference: {reference_time:.4f} ms")
            self.assertTrue(mx.allclose(modified, expected, atol=1e-2), "Benchmark Test Failed.")

    def test_score_mod_non_square_matrices(self):
        print("\n=== test_score_mod_non_square_matrices ===")
        test_cases = [
            ((64, 128), (64, 128)),
            ((128, 64), (128, 64)),
            ((256, 512), (256, 512)),
            ((1024, 2048), (1024, 2048)),
        ]
        dtype = mx.float32

        for (raw_shape, mod_shape) in test_cases:
            print(f"\n--- Testing non-square matrices: raw_scores{raw_shape} x modification_params{mod_shape} ---")
            raw_scores = mx.arange(raw_shape[0] * raw_shape[1], dtype=dtype).reshape(raw_shape)
            modification_params = mx.full(mod_shape, 2.0, dtype=dtype)  # Scale by 2.0

            modification_type = "scale"
            THREADGROUP_MEM_SIZE = 16
            grid_x = ((raw_shape[0] + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
            grid_y = ((raw_shape[1] + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE) * THREADGROUP_MEM_SIZE
            grid_z = 1
            threadgroup_z = 1

            grid = (grid_x, grid_y, grid_z)
            threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, threadgroup_z)

            print(f"       raw_scores.shape: {raw_scores.shape}, modification_params.shape: {modification_params.shape}, dtype={dtype}")
            print(f"       Grid: {grid}, Threadgroup: {threadgroup}")

            self.current_modification_type = modification_type
            result = self.run_kernel_score_mod(raw_scores, modification_params, modification_type, grid, threadgroup)
            expected = self.reference_score_mod(raw_scores, modification_params)

            self.assertTrue(mx.allclose(result, expected, atol=1e-2), "Non-Square Matrices Test Failed.")

if __name__ == "__main__":
    print("=== Starting Score Mod Kernel Tests ===")
    unittest.main()
