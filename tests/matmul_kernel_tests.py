import unittest
import mlx.core as mx
from flash_attention.matmul_kernel import matmul_kernel, matmul_spec
import time
import contextlib

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

class MetalKernelTest:
    def __init__(
        self,
        name,
        kernel,
        inputs,
        output_shape,
        grid,
        threadgroup,
        spec=None,
        A_trans: bool = True,
        B_trans: bool = True,
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
        self.spec = spec
        self.A_trans = A_trans
        self.B_trans = B_trans
        self.kernel_kwargs = kernel_kwargs

    def run_metal(self):
        print(f"[RUN] {self.name}")
        try:
            # Pass A_trans and B_trans as keyword arguments
            kernel = self.kernel(*self.inputs, A_trans=self.A_trans, B_trans=self.B_trans)
        except Exception as e:
            print(f"       [ERROR] Kernel creation failed: {e}")
            raise
        return kernel

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
        M_group = N_group = 8

        for dtype in [mx.float16, mx.float32]:
            print(f"\n--- Testing matmul_simple with dtype={dtype} ---")
            a = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE))
            b = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE))
            output_shape = (SIZE, SIZE)

            threadgroup = (min(N_group, SIZE), min(M_group, SIZE), 3)
            grid = ((SIZE + N_group - 1) // N_group * SIZE, (SIZE + M_group - 1) // M_group * SIZE, 3)

            print(f"       a.shape: {a.shape}, b.shape: {b.shape}")
            print(f"       Grid: {grid}, Threadgroup: {threadgroup}, dtype={dtype}")

            problem = MetalKernelTest(
                "Matrix Multiplication (Simple)",
                matmul_kernel,
                [a, b],
                output_shape,
                grid=grid,
                threadgroup=threadgroup,
                spec=matmul_spec,
                A_trans=False,
                B_trans=False,
            )

            self.assertTrue(problem.check())

    def test_matmul_transposed_a(self):
        print("\n=== test_matmul_transposed_a ===")
        SIZE = 8
        M_group = N_group = 8

        for dtype in [mx.float16, mx.float32]:
            print(f"\n--- Testing matmul_transposed_a with dtype={dtype} ---")
            a = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE)).T
            b = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE))
            output_shape = (SIZE, SIZE)

            threadgroup = (min(N_group, SIZE), min(M_group, SIZE), 3)
            grid = ((SIZE + N_group - 1) // N_group * SIZE, (SIZE + M_group - 1) // M_group * SIZE, 3)
            print(f"       a.shape (transposed): {a.shape}, b.shape: {b.shape}, dtype={dtype}")
            print(f"       Grid: {grid}, Threadgroup: {threadgroup}")

            problem = MetalKernelTest(
                "Matrix Multiplication (Transposed A)",
                matmul_kernel,
                [a, b],
                output_shape,
                grid=grid,
                threadgroup=threadgroup,
                spec=matmul_spec,
                A_trans=True,
                B_trans=False,
            )

            self.assertTrue(problem.check())

    def test_matmul_incompatible_sizes(self):
        print("\n=== test_matmul_incompatible_sizes ===")

        # Initialize matrices with incompatible sizes
        a = mx.array([[1.0] * 13] * 17)  # Shape: (17, 13)
        b = mx.array([[1.0] * 11] * 17)  # Shape: (17, 11)  # Incompatible

        # Expect an AssertionError due to incompatible inner dimensions
        with self.assertRaises(AssertionError):
            matmul_kernel(a, b, A_trans=False, B_trans=False)

    def test_benchmark_matmul(self):
        print("\n=== test_benchmark_matmul ===")
        SIZES = [512, 1024, 2048]  # Test different matrix sizes
        # Remove M_group and N_group as they're no longer needed
        NUM_RUNS = 10
        WARM_UP_RUNS = 2

        for SIZE in SIZES:
            print(f"Benchmarking for size {SIZE}x{SIZE}:")
            a = mx.random.normal(shape=(SIZE, SIZE), dtype=mx.float32)
            b = mx.random.normal(shape=(SIZE, SIZE), dtype=mx.float32)

            # {{ Edit Start: Update grid and threadgroup calculation }}
            BLOCK_SIZE = 8  # Ensure this matches the kernel's BLOCK_SIZE
            grid_x = (SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
            grid_y = (SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
            grid = (grid_x, grid_y, 3)
            threadgroup = (BLOCK_SIZE, BLOCK_SIZE, 3)
            # {{ Edit End }}

            # Warm-up runs
            for _ in range(WARM_UP_RUNS):
                self.run_metal_matmul(a, b, grid, threadgroup, A_trans=False, B_trans=False)
                self.run_metal_matmul(a, b, grid, threadgroup, A_trans=True, B_trans=False)
                self.run_metal_matmul(a, b, grid, threadgroup, A_trans=False, B_trans=True)
                self.run_metal_matmul(a, b, grid, threadgroup, A_trans=True, B_trans=True)
                self.run_mlx_matmul(a, b)
                mx.synchronize()

            # Benchmark custom implementation using Timing
            with Timing("Custom Matmul", lambda x: f" {x:.3f} ms"):
                for _ in range(NUM_RUNS):
                    self.run_metal_matmul(a, b, grid, threadgroup, A_trans=False, B_trans=False)
                    self.run_metal_matmul(a, b, grid, threadgroup, A_trans=True, B_trans=False)
                    self.run_metal_matmul(a, b, grid, threadgroup, A_trans=False, B_trans=True)
                    self.run_metal_matmul(a, b, grid, threadgroup, A_trans=True, B_trans=True)
                mx.synchronize()

            # Benchmark MLX built-in matmul using Timing
            with Timing("MLX Matmul", lambda x: f" {x:.3f} ms"):
                for _ in range(NUM_RUNS):
                    self.run_mlx_matmul(a, b)
                mx.synchronize()

    def run_metal_matmul(self, a, b, grid, threadgroup, A_trans=False, B_trans=False):
        output_shape = (a.shape[0], b.shape[1])
        problem = MetalKernelTest(
            "Benchmark Metal Matmul",
            matmul_kernel,
            [a, b],
            output_shape,
            grid=grid,
            threadgroup=threadgroup,
            spec=matmul_spec,
            A_trans=A_trans,
            B_trans=B_trans,
        )
        result = problem.run_metal()
        return result

    def run_mlx_matmul(self, a, b):
        result = mx.matmul(a, b) # removed stream=mx.gpu as it's default
        return result

    def test_matmul_transposed_b(self):
        print("\n=== test_matmul_transposed_b ===")
        SIZE = 8
        M_group = N_group = 8

        for dtype in [mx.float16, mx.float32]:
            print(f"\n--- Testing matmul_transposed_b with dtype={dtype} ---")
            a = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE))
            b = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE)).T
            output_shape = (SIZE, SIZE)

            threadgroup = (min(N_group, SIZE), min(M_group, SIZE), 3)
            grid = ((SIZE + N_group - 1) // N_group * SIZE, (SIZE + M_group - 1) // M_group * SIZE, 3)
            print(f"       a.shape: {a.shape}, b.shape (transposed): {b.shape}, dtype={dtype}")
            print(f"       Grid: {grid}, Threadgroup: {threadgroup}")

            problem = MetalKernelTest(
                "Matrix Multiplication (Transposed B)",
                matmul_kernel,
                [a, b],
                output_shape,
                grid=grid,
                threadgroup=threadgroup,
                spec=matmul_spec,
                A_trans=False,
                B_trans=True,
            )

            self.assertTrue(problem.check())

    def test_matmul_zero(self):
        print("\n=== test_matmul_zero ===")
        M, K, N = 8, 8, 8

        for dtype in [mx.float16, mx.float32]:
            print(f"\n--- Testing matmul_zero with dtype={dtype} ---")
            a = mx.zeros((M, K), dtype=dtype)
            b = mx.zeros((K, N), dtype=dtype)
            output_shape = (M, N)

            threadgroup = (min(N, 8), min(M, 8), 3)
            grid = ((8 + N - 1) // N * 8, (8 + M - 1) // M * 8, 3)
            print(f"       a.shape: {a.shape}, b.shape: {b.shape}, dtype={dtype}")
            print(f"       Grid: {grid}, Threadgroup: {threadgroup}")

            problem = MetalKernelTest(
                "Matrix Multiplication (Zero Matrices)",
                matmul_kernel,
                [a, b],
                output_shape,
                grid=grid,
                threadgroup=threadgroup,
                spec=matmul_spec,
                A_trans=False,
                B_trans=False,
            )

            self.assertTrue(problem.check())

    def test_matmul_negative(self):
        print("\n=== test_matmul_negative ===")
        M, K, N = 8, 8, 8

        for dtype in [mx.float16, mx.float32]:
            print(f"\n--- Testing matmul_negative with dtype={dtype} ---")
            a = mx.array([(-1) ** (i % 2) * i for i in range(M * K)], dtype=dtype).reshape((M, K))
            b = mx.array([(-1) ** (i % 2) * (i + 1) for i in range(K * N)], dtype=dtype).reshape((K, N))
            output_shape = (M, N)

            threadgroup = (min(N, 8), min(M, 8), 3)
            grid = ((8 + N - 1) // N * 8, (8 + M - 1) // M * 8, 3)
            print(f"       a.shape: {a.shape}, b.shape: {b.shape}, dtype={dtype}")
            print(f"       Grid: {grid}, Threadgroup: {threadgroup}")

            problem = MetalKernelTest(
                "Matrix Multiplication (Negative Values)",
                matmul_kernel,
                [a, b],
                output_shape,
                grid=grid,
                threadgroup=threadgroup,
                spec=matmul_spec,
                A_trans=False,
                B_trans=False,
            )

            self.assertTrue(problem.check())

    def test_matmul_transpose_both(self):
        print("\n=== test_matmul_transpose_both ===")
        M, K, N = 8, 8, 8

        for dtype in [mx.float16, mx.float32]:
            print(f"\n--- Testing matmul_transpose_both with dtype={dtype} ---")
            a = mx.arange(M * K, dtype=dtype).reshape((M, K)).T
            b = mx.arange(K * N, dtype=dtype).reshape((K, N)).T
            output_shape = (K, M)  # Since both A and B are transposed

            threadgroup = (min(N, 8), min(M, 8), 3)
            grid = ((8 + N - 1) // N * 8, (8 + M - 1) // M * 8, 3)
            print(f"       a.shape (transposed): {a.shape}, b.shape (transposed): {b.shape}, dtype={dtype}")
            print(f"       Grid: {grid}, Threadgroup: {threadgroup}")

            problem = MetalKernelTest(
                "Matrix Multiplication (Transposed Both)",
                matmul_kernel,
                [a, b],
                output_shape,
                grid=grid,
                threadgroup=threadgroup,
                spec=matmul_spec,
                A_trans=True,
                B_trans=True,
            )

            self.assertTrue(problem.check())


if __name__ == "__main__":
    print("=== Starting Matmul Kernel Tests ===")
    unittest.main()
