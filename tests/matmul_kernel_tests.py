import unittest
import mlx.core as mx
import time
import os
import glob
import shutil
from flash_attention.matmul_kernel import matmul_kernel, matmul_spec


# Utility classes and functions integrated from kernel_utils.py
class MetalKernelTest:

    def __init__(
        self,
        name,
        fn,
        inputs,
        output_shape,
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        spec=None,
    ):
        self.name = name
        self.fn = fn
        self.inputs = inputs
        self.output_shape = output_shape
        self.grid = grid
        self.threadgroup = threadgroup
        self.spec = spec

    def run_metal(self):
        outputs = self.fn(*self.inputs)(
            inputs=self.inputs,
            grid=self.grid,
            threadgroup=self.threadgroup,
            output_shapes=[self.output_shape],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
            verbose=os.getenv("VERBOSE") == "1",
            init_value=0,
        )
        return outputs[0]

    def check(self):
        if self.spec is None:
            print(f"{self.name}: No spec provided, skipping check.")
            return True

        x = self.run_metal()
        y = self.spec(*self.inputs)
        if mx.allclose(x, y):
            print(f"{self.name}: Passed Tests!")
            return True
        else:
            print(f"{self.name}: Failed Tests.")
            print("Yours:", x)
            print("Spec :", y)
            return False

    def show(self):
        self.check()


def remove_gputrace_file():
    gputrace_files = glob.glob("custom_kernel_*.gputrace")
    if not gputrace_files:
        return
    for gputrace_file in gputrace_files:
        try:
            if os.path.isdir(gputrace_file):
                shutil.rmtree(gputrace_file)
            else:
                os.remove(gputrace_file)
        except Exception as e:
            print(f"Error removing {gputrace_file}: {e}")


class TestMatmulKernel(unittest.TestCase):
    def test_matmul_simple(self):
        SIZE = 2
        a = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE))
        b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T
        output_shape = (SIZE, SIZE)

        problem = MetalKernelTest(
            "Matrix Multiplication (Simple)",
            matmul_kernel,
            [a, b],
            output_shape,
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
        )

        self.assertTrue(problem.check())

    def test_matmul_full(self):
        SIZE = 8
        a = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE))
        b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T
        output_shape = (SIZE, SIZE)

        problem = MetalKernelTest(
            "Matrix Multiplication (Full)",
            matmul_kernel,
            [a, b],
            output_shape,
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            spec=matmul_spec,
        )

        self.assertTrue(problem.check())

    def test_matmul_zero_matrix(self):
        SIZE = 4
        a = mx.zeros((SIZE, SIZE), dtype=mx.float32)
        b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T
        output_shape = (SIZE, SIZE)

        problem = MetalKernelTest(
            "Matrix Multiplication (Zero Matrix)",
            matmul_kernel,
            [a, b],
            output_shape,
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            spec=matmul_spec,
        )

        self.assertTrue(problem.check())

    def test_matmul_identity_matrix(self):
        SIZE = 5
        a = mx.eye(SIZE, dtype=mx.float32)
        b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE))
        output_shape = (SIZE, SIZE)

        problem = MetalKernelTest(
            "Matrix Multiplication (Identity Matrix)",
            matmul_kernel,
            [a, b],
            output_shape,
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            spec=matmul_spec,
        )

        self.assertTrue(problem.check())

    def test_benchmark_matmul(self):
        SIZE = 512
        NUM_RUNS = 100
        WARM_UP_RUNS = 10

        a = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE))
        b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T

        # Warm-up runs
        for _ in range(WARM_UP_RUNS):
            self.run_metal_matmul(a, b)
            self.run_mlx_matmul(a, b)
            mx.synchronize()

        # Benchmark custom implementation
        start_time = time.perf_counter()
        for _ in range(NUM_RUNS):
            self.run_metal_matmul(a, b)
        mx.synchronize()
        custom_time = (time.perf_counter() - start_time) * 1000  # ms

        # Benchmark MLX built-in matmul
        start_time = time.perf_counter()
        for _ in range(NUM_RUNS):
            self.run_mlx_matmul(a, b)
        mx.synchronize()
        mlx_time = (time.perf_counter() - start_time) * 1000  # ms

        print(f"\nCustom matmul time: {custom_time:.3f} ms")
        print(f"MLX matmul time: {mlx_time:.3f} ms")
        print(f"Speedup: {mlx_time / custom_time:.2f}x\n")

    def run_metal_matmul(self, a, b):
        output_shape = (a.shape[0], b.shape[1])
        problem = MetalKernelTest(
            "Benchmark Metal Matmul",
            matmul_kernel,
            [a, b],
            output_shape,
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            spec=matmul_spec,
        )
        return problem.run_metal()

    def run_mlx_matmul(self, a, b):
        return mx.matmul(a, b, stream=mx.gpu)


if __name__ == "__main__":
    unittest.main()
