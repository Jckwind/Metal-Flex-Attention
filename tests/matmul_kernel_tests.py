import unittest
import mlx.core as mx
import time
import os
from flash_attention.matmul_kernel import matmul_kernel, matmul_spec


class MetalKernelTest:
    def __init__(
        self,
        name,
        fn,
        inputs,
        output_shape,
        grid,
        threadgroup,
        spec=None,
        **kernel_kwargs,
    ):
        self.name = name
        self.fn = fn
        self.inputs = inputs
        self.output_shape = output_shape
        self.grid = grid
        self.threadgroup = threadgroup
        self.spec = spec
        self.kernel_kwargs = kernel_kwargs

    def run_metal(self):
        kernel = self.fn(*self.inputs, **self.kernel_kwargs)
        outputs = kernel(
            inputs=self.inputs,
            grid=self.grid,
            threadgroup=self.threadgroup,
            output_shapes=[self.output_shape],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
            verbose=os.getenv("VERBOSE") == "1",
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


class TestMatmulKernel(unittest.TestCase):
    def test_matmul_simple(self):
        SIZE = 8
        M_group = 8
        N_group = 8
        K_group = 8

        a = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE))
        b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T
        output_shape = (SIZE, SIZE)

        grid = (SIZE // N_group, SIZE // M_group, 1)
        threadgroup = (N_group, M_group, 1)

        problem = MetalKernelTest(
            "Matrix Multiplication (Simple)",
            matmul_kernel,
            [a, b],
            output_shape,
            grid=grid,
            threadgroup=threadgroup,
            spec=matmul_spec,
            M_group=M_group,
            N_group=N_group,
            K_group=K_group,
        )

        self.assertTrue(problem.check())

    def test_matmul_transposed(self):
        SIZE = 8
        M_group = 8
        N_group = 8
        K_group = 8

        a = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T
        b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE))
        output_shape = (SIZE, SIZE)

        grid = (SIZE // N_group, SIZE // M_group, 1)
        threadgroup = (N_group, M_group, 1)

        problem = MetalKernelTest(
            "Matrix Multiplication (Transposed A)",
            matmul_kernel,
            [a, b],
            output_shape,
            grid=grid,
            threadgroup=threadgroup,
            spec=matmul_spec,
            M_group=M_group,
            N_group=N_group,
            K_group=K_group,
            A_trans=True,
            B_trans=False,
        )

        self.assertTrue(problem.check())

    def test_matmul_incompatible_sizes(self):
        M = 17
        K = 13
        N = 11
        M_group = 8
        N_group = 8
        K_group = 8

        a = mx.random.normal(shape=(M, K), dtype=mx.float32)
        b = mx.random.normal(shape=(K, N), dtype=mx.float32)
        output_shape = (M, N)

        grid = ((N + N_group - 1) // N_group, (M + M_group - 1) // M_group, 1)
        threadgroup = (N_group, M_group, 1)

        problem = MetalKernelTest(
            "Matrix Multiplication (Incompatible Sizes)",
            matmul_kernel,
            [a, b],
            output_shape,
            grid=grid,
            threadgroup=threadgroup,
            spec=matmul_spec,
            M_group=M_group,
            N_group=N_group,
            K_group=K_group,
        )

        self.assertTrue(problem.check())

    def test_benchmark_matmul(self):
        SIZE = 512
        M_group = 16
        N_group = 16
        K_group = 16
        NUM_RUNS = 10
        WARM_UP_RUNS = 2

        a = mx.random.normal(shape=(SIZE, SIZE), dtype=mx.float32)
        b = mx.random.normal(shape=(SIZE, SIZE), dtype=mx.float32)

        grid = (SIZE // N_group, SIZE // M_group, 1)
        threadgroup = (N_group, M_group, 1)

        # Warm-up runs
        for _ in range(WARM_UP_RUNS):
            self.run_metal_matmul(a, b, grid, threadgroup, M_group, N_group, K_group)
            self.run_mlx_matmul(a, b)
            mx.synchronize()

        # Benchmark custom implementation
        start_time = time.perf_counter()
        for _ in range(NUM_RUNS):
            self.run_metal_matmul(a, b, grid, threadgroup, M_group, N_group, K_group)
        mx.synchronize()
        custom_time = (time.perf_counter() - start_time) * 1000  # ms

        # Benchmark MLX built-in matmul
        start_time = time.perf_counter()
        for _ in range(NUM_RUNS):
            self.run_mlx_matmul(a, b)
        mx.synchronize()
        mlx_time = (time.perf_counter() - start_time) * 1000  # ms

        print(f"\nCustom matmul time: {custom_time / NUM_RUNS:.3f} ms per run")
        print(f"MLX matmul time: {mlx_time / NUM_RUNS:.3f} ms per run")
        print(f"Speedup: {mlx_time / custom_time:.2f}x\n")

    def run_metal_matmul(self, a, b, grid, threadgroup, M_group, N_group, K_group):
        output_shape = (a.shape[0], b.shape[1])
        problem = MetalKernelTest(
            "Benchmark Metal Matmul",
            matmul_kernel,
            [a, b],
            output_shape,
            grid=grid,
            threadgroup=threadgroup,
            M_group=M_group,
            N_group=N_group,
            K_group=K_group,
        )
        return problem.run_metal()

    def run_mlx_matmul(self, a, b):
        return mx.matmul(a, b, stream=mx.gpu)


if __name__ == "__main__":
    unittest.main()
