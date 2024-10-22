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
            spec_output = self.spec(*self.inputs, A_trans=self.A_trans, B_trans=self.B_trans)
            if mx.allclose(metal_output, spec_output, atol=1e-2):
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
        result = mx.matmul(a, b) 
        return result

    def test_matmul_transpose_both(self):
        print("\n=== test_matmul_transpose_both ===")
        M, K, N = 8, 8, 8

        for dtype in [mx.float32]:  # Only float32 is supported now
            print(f"\n--- Testing matmul_transpose_both with dtype={dtype} ---")
            a = mx.arange(M * K, dtype=dtype).reshape((M, K)).transpose()
            b = mx.arange(K * N, dtype=dtype).reshape((K, N)).transpose()
            output_shape = (K, M)  # Since both A and B are transposed

            # {{ Edit Start: Correct grid and threadgroup calculation for 3D tiling }}
            THREADGROUP_MEM_SIZE = 8  # Ensure consistency with matmul_kernel.py

            grid_x = (M + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
            grid_y = (N + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
            grid_z = 3
            threadgroup_z = 3  # Align with DEPTH_SIZE to prevent multiple writes

            grid = (grid_x, grid_y, grid_z)
            threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, threadgroup_z)
            # {{ Edit End }}

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

    def test_matmul_simple(self):
        print("\n=== test_matmul_simple ===")
        SIZE = 8

        for dtype in [mx.float32]:  # Only float32 is supported now
            print(f"\n--- Testing matmul_simple with dtype={dtype} ---")
            a = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE))
            b = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE))
            output_shape = (SIZE, SIZE)

            # {{ Edit Start: Correct grid and threadgroup calculation for 3D tiling }}
            THREADGROUP_MEM_SIZE = 8  # Ensure consistency with matmul_kernel.py

            grid_x = (SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
            grid_y = (SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
            grid_z = 3
            threadgroup_z = 3  # Align with DEPTH_SIZE to prevent multiple writes

            grid = (grid_x, grid_y, grid_z)
            threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, threadgroup_z)
            # {{ Edit End }}

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

        for dtype in [mx.float32]:  # Only float32 is supported now
            print(f"\n--- Testing matmul_transposed_a with dtype={dtype} ---")
            a = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE)).transpose()
            b = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE))
            output_shape = (SIZE, SIZE)

            # {{ Edit Start: Correct grid and threadgroup calculation for 3D tiling }}
            THREADGROUP_MEM_SIZE = 8  # Ensure consistency with matmul_kernel.py

            grid_x = (SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
            grid_y = (SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
            grid_z = 3
            threadgroup_z = 3  # Align with DEPTH_SIZE to prevent multiple writes

            grid = (grid_x, grid_y, grid_z)
            threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, threadgroup_z)
            # {{ Edit End }}

            print(f"       a.shape: {a.shape}, b.shape: {b.shape}, dtype={dtype}")
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

    def test_matmul_transposed_b(self):
        print("\n=== test_matmul_transposed_b ===")
        SIZE = 8

        for dtype in [mx.float32]:  # Only float32 is supported now
            print(f"\n--- Testing matmul_transposed_b with dtype={dtype} ---")
            a = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE))
            b = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE)).transpose()
            output_shape = (SIZE, SIZE)

            # {{ Edit Start: Correct grid and threadgroup calculation for 3D tiling }}
            THREADGROUP_MEM_SIZE = 8  # Ensure consistency with matmul_kernel.py

            grid_x = (SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
            grid_y = (SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
            grid_z = 3
            threadgroup_z = 3  # Align with DEPTH_SIZE to prevent multiple writes

            grid = (grid_x, grid_y, grid_z)
            threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, threadgroup_z)

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

        for dtype in [mx.float32]:  # Only float32 is supported now
            print(f"\n--- Testing matmul_zero with dtype={dtype} ---")
            a = mx.zeros((M, K), dtype=dtype)
            b = mx.zeros((K, N), dtype=dtype)
            output_shape = (M, N)

            # {{ Edit Start: Correct grid and threadgroup calculation for 3D tiling }}
            THREADGROUP_MEM_SIZE = 8  # Ensure consistency with matmul_kernel.py

            grid_x = (M + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
            grid_y = (N + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
            grid_z = 3
            threadgroup_z = 3  # Align with DEPTH_SIZE to prevent multiple writes

            grid = (grid_x, grid_y, grid_z)
            threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, threadgroup_z)

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

        for dtype in [mx.float32]:  # Only float32 is supported now
            print(f"\n--- Testing matmul_negative with dtype={dtype} ---")
            a = mx.array([(-1) ** (i % 2) * i for i in range(M * K)], dtype=dtype).reshape((M, K))
            b = mx.array([(-1) ** (i % 2) * (i + 1) for i in range(K * N)], dtype=dtype).reshape((K, N))
            output_shape = (M, N)

            # {{ edit_1 }}
            THREADGROUP_MEM_SIZE = 8  # Ensure consistency with matmul_kernel.py

            grid_x = (M + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
            grid_y = (N + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
            grid_z = 3
            threadgroup_z = 3  # Align with DEPTH_SIZE to prevent multiple writes

            grid = (grid_x, grid_y, grid_z)
            threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, threadgroup_z)
            # {{ edit_1 }}

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

    def test_matmul_large(self):
        print("\n=== test_matmul_large ===")
        SIZE = 256

        for dtype in [mx.float32]:  # Only float32 is supported now due to atomic operations
            print(f"\n--- Testing matmul_large with dtype={dtype} ---")
            a = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE))
            b = mx.arange(SIZE * SIZE, dtype=dtype).reshape((SIZE, SIZE))
            output_shape = (SIZE, SIZE)

            THREADGROUP_MEM_SIZE = 8  # Ensure consistency with matmul_kernel.py

            grid_x = (SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
            grid_y = (SIZE + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE * THREADGROUP_MEM_SIZE
            grid_z = 3  # Retain 3 for large matrices
            threadgroup_z = 3  # Retain 3 for large matrices

            grid = (grid_x, grid_y, grid_z)
            threadgroup = (THREADGROUP_MEM_SIZE, THREADGROUP_MEM_SIZE, threadgroup_z)

            print(f"       a.shape: {a.shape}, b.shape: {b.shape}")
            print(f"       Grid: {grid}, Threadgroup: {threadgroup}, dtype={dtype}")

            problem = MetalKernelTest(
                "Matrix Multiplication (Large)",
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

    def test_benchmark_matmul(self):
        print("\n=== test_benchmark_matmul ===")
        sizes = [64, 128, 256, 512, 1024, 2048, 4096]
        warm_up_runs = 100
        benchmark_runs = 1000  # Increased number of runs for better accuracy

        for size in sizes:
            a = mx.arange(size * size, dtype=mx.float32).reshape((size, size))
            b = mx.arange(size * size, dtype=mx.float32).reshape((size, size))

            print(f"\n--- Benchmarking size: {size}x{size} ---")

            # Warm-up runs
            for _ in range(warm_up_runs):
                custom_result = matmul_kernel(a, b, False, False)
                mlx_result = mx.matmul(a, b)
                mx.synchronize()  # Ensure GPU operations are complete

            # Benchmark custom kernel
            start = time.perf_counter_ns()
            for _ in range(benchmark_runs):
                custom_result = matmul_kernel(a, b, False, False)
                mx.synchronize()  # Ensure GPU operations are complete
            end = time.perf_counter_ns()
            custom_time = (end - start) / benchmark_runs / 1e6  # Convert to milliseconds

            # Benchmark MLX matmul
            start = time.perf_counter_ns()
            for _ in range(benchmark_runs):
                mlx_result = mx.matmul(a, b)
                mx.synchronize()  # Ensure GPU operations are complete
            end = time.perf_counter_ns()
            mlx_time = (end - start) / benchmark_runs / 1e6  # Convert to milliseconds

            print(f"Size: {size}x{size} | Custom Matmul Kernel: {custom_time:.4f} ms | MLX Matmul: {mlx_time:.4f} ms")
            self.assertTrue(mx.allclose(custom_result, mlx_result))

if __name__ == "__main__":
    print("=== Starting Matmul Kernel Tests ===")
    unittest.main()
