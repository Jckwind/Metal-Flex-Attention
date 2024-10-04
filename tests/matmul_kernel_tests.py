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
        print(f"[INIT] {name}")
        print(f"       Function: {fn.__name__ if hasattr(fn, '__name__') else fn}")
        print(f"       Inputs shapes: {[inp.shape for inp in inputs]}")
        print(f"       Output shape: {output_shape}")
        print(f"       Grid: {grid}, Threadgroup: {threadgroup}")
        if spec:
            print(f"       Spec: {spec.__name__ if hasattr(spec, '__name__') else spec}")
        if kernel_kwargs:
            print(f"       Kernel kwargs: {kernel_kwargs}")
        
        self.name = name
        self.fn = fn
        self.inputs = inputs
        self.output_shape = output_shape
        self.grid = grid
        self.threadgroup = threadgroup
        self.spec = spec
        self.kernel_kwargs = kernel_kwargs

    def run_metal(self):
        print(f"[RUN] {self.name}")
        kernel = self.fn(*self.inputs, **self.kernel_kwargs)
        print(f"       Kernel object: {kernel}")
        try:
            outputs = kernel(
                inputs=self.inputs,
                grid=self.grid,
                threadgroup=self.threadgroup,
                output_shapes=[self.output_shape],
                output_dtypes=[mx.float32],
                stream=mx.gpu,
                verbose=os.getenv("VERBOSE") == "1",
            )
            print(f"       Execution completed. Output shape: {outputs[0].shape}")
            return outputs[0]
        except Exception as e:
            print(f"       [ERROR] Kernel execution failed: {e}")
            raise

    def check(self):
        print(f"[CHECK] {self.name}")
        if self.spec is None:
            print("       Spec not provided. Skipping verification.")
            return True

        try:
            metal_output = self.run_metal()
            spec_output = self.spec(*self.inputs)
            if mx.allclose(metal_output, spec_output):
                print("       Test Passed.")
                return True
            else:
                print("       Test Failed.")
                print(f"         Expected: {spec_output}")
                print(f"         Received: {metal_output}")
                return False
        except Exception as e:
            print(f"       [ERROR] Check failed: {e}")
            return False

    def show(self):
        print(f"[SHOW] {self.name}")
        self.check()


class TestMatmulKernel(unittest.TestCase):
    def test_matmul_simple(self):
        print("\n=== test_matmul_simple ===")
        SIZE = 8
        M_group = N_group = K_group = 8

        a = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE))
        b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T
        output_shape = (SIZE, SIZE)

        threadgroup = (min(N_group, SIZE), min(M_group, SIZE), 1)
        grid = ((SIZE + N_group - 1) // N_group, (SIZE + M_group - 1) // M_group, 1)
        
        print(f"       a.shape: {a.shape}, b.shape: {b.shape}")
        print(f"       Grid: {grid}, Threadgroup: {threadgroup}")

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
        print("\n=== test_matmul_transposed ===")
        SIZE = 8
        M_group = N_group = K_group = 8

        a = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T
        b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE))
        output_shape = (SIZE, SIZE)

        threadgroup = (min(N_group, SIZE), min(M_group, SIZE), 1)
        grid = ((SIZE + N_group - 1) // N_group, (SIZE + M_group - 1) // M_group, 1)
        print(f"       a.shape (transposed): {a.shape}, b.shape: {b.shape}")
        print(f"       Grid: {grid}, Threadgroup: {threadgroup}")

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
        print("\n=== test_matmul_incompatible_sizes ===")
        M, K, N = 17, 13, 11
        M_group = N_group = K_group = 8

        a = mx.random.normal(shape=(M, K), dtype=mx.float32)
        b = mx.random.normal(shape=(K, N), dtype=mx.float32)
        output_shape = (M, N)

        grid = ((N + N_group - 1) // N_group, (M + M_group - 1) // M_group, 1)
        threadgroup = (min(N_group, N), min(M_group, M), 1)
        
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
        print("\n=== test_benchmark_matmul ===")
        SIZE = 512
        M_group = N_group = K_group = 16
        NUM_RUNS = 10
        WARM_UP_RUNS = 2

        a = mx.random.normal(shape=(SIZE, SIZE), dtype=mx.float32)
        b = mx.random.normal(shape=(SIZE, SIZE), dtype=mx.float32)

        threadgroup = (min(N_group, SIZE), min(M_group, SIZE), 1)
        grid = ((SIZE + N_group - 1) // N_group, (SIZE + M_group - 1) // M_group, 1)

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

        avg_custom = custom_time / NUM_RUNS
        avg_mlx = mlx_time / NUM_RUNS
        speedup = mlx_time / custom_time if custom_time > 0 else float('inf')

        print(f"       Custom Matmul Avg Time: {avg_custom:.3f} ms/run")
        print(f"       MLX Matmul Avg Time: {avg_mlx:.3f} ms/run")
        print(f"       Speedup: {speedup:.2f}x\n")

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
        result = problem.run_metal()
        return result

    def run_mlx_matmul(self, a, b):
        result = mx.matmul(a, b, stream=mx.gpu)
        return result


if __name__ == "__main__":
    print("=== Starting Matmul Kernel Tests ===")
    unittest.main()