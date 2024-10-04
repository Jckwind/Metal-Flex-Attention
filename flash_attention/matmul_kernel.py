import mlx.core as mx


def matmul_spec(a: mx.array, b: mx.array):
    return mx.matmul(a, b)


def matmul_kernel(a: mx.array, b: mx.array):
    # Matrix dimensions
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]

    # Leading dimensions
    A_leading_dimension = a.strides[1] // a.itemsize
    B_leading_dimension = b.strides[1] // b.itemsize
    C_leading_dimension = N  # Assuming row-major order

    # Configuration parameters
    load_previous_C = False
    A_trans = False
    B_trans = False
    M_group = 64
    N_group = 64
    K_group = 64

    # Define macros directly in the header
    header_macros = f"""
    #define M {M}
    #define N {N}
    #define K {K}

    #define A_ld {A_leading_dimension}
    #define B_ld {B_leading_dimension}
    #define C_ld {C_leading_dimension}

    #define load_previous_C {int(load_previous_C)}
    #define A_trans {int(A_trans)}
    #define B_trans {int(B_trans)}
    #define M_group {M_group}
    #define N_group {N_group}
    #define K_group {K_group}
    """

    header = (
        header_macros
        + """
    #include <metal_stdlib>
    using namespace metal;

    template <typename T>
    """
    )

    # Kernel source (function body)
    source = """
    [[kernel]] void matmul_kernel(
        const device T* A [[buffer(0)]],
        const device T* B [[buffer(1)]],
        device T* C [[buffer(2)]],
        uint3 gid [[threadgroup_position_in_grid]],
        ushort threads_per_threadgroup [[threads_per_threadgroup]],
        ushort thread_index_in_simdgroup [[thread_index_in_simdgroup]],
        ushort simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
        uint thread_position_in_threadgroup [[thread_position_in_threadgroup]]
    ) {
        // Compute edges and remainders
        uint M_edge = M - (M % M_group);
        uint N_edge = N - (N % N_group);

        ushort M_remainder = (M % 8 == 0) ? 8 : ushort(M % 8);
        ushort N_remainder = (N % 8 == 0) ? 8 : ushort(N % 8);
        ushort K_remainder = (K % K_group == 0) ? ushort(K_group) : ushort(K % K_group);
        ushort K_remainder_padded = ((K_remainder + 7) / 8) * 8;

        ushort M_shift = (M < M_group) ? 0 : ushort(8 - M_remainder);
        ushort N_shift = (N < N_group) ? 0 : ushort(8 - N_remainder);

        // Thread and threadgroup indices
        uint2 gid_xy = uint2(gid.x, gid.y);
        ushort sidx = simdgroup_index_in_threadgroup;
        ushort lane_id = thread_index_in_simdgroup;
        uint threads_per_threadgroup = threads_per_threadgroup;
        uint thread_position_in_threadgroup = thread_position_in_threadgroup;

        ushort2 sid = ushort2(sidx % (N_group / 8), sidx / (N_group / 8));
        ushort2 morton_offset = ushort2(lane_id % 8, lane_id / 8);

        // Return early if SIMD is out of bounds
        uint M_offset = gid_xy.y * M_group;
        uint N_offset = gid_xy.x * N_group;
        if (M_offset + sid.y * 8 >= M || N_offset + sid.x * 8 >= N) {
            return;
        }
        ushort2 offset_in_group = ushort2(sid.x * 8 + morton_offset.x,
                                          sid.y * 8 + morton_offset.y);

        // Shift the matrix block within bounds, if possible
        uint M_shift_local = M_shift;
        uint N_shift_local = N_shift;
        uint M_edge_local = M_edge;
        uint N_edge_local = N_edge;

        if ((M_shift_local != 0) && (gid_xy.y * M_group >= M_edge_local)) {
            M_offset -= M_shift_local;
        }
        if ((N_shift_local != 0) && (gid_xy.x * N_group >= N_edge_local)) {
            N_offset -= N_shift_local;
        }

        // Initialize accumulator
        thread simdgroup_matrix<T, 8, 8> C_sram;

        if (load_previous_C) {
            // Load previous C values
            uint2 C_offset = uint2(N_offset + offset_in_group.x,
                                   M_offset + offset_in_group.y);
            device T* C_dst = C + C_offset.y * C_ld + C_offset.x;
            // Initialize C_sram
            C_sram = simdgroup_matrix<T, 8, 8>((T)0);
            if (C_offset.x < N && C_offset.y < M) {
                C_sram.load(C_dst, C_ld);
            }
        } else {
            // Initialize C_sram to zero
            C_sram = simdgroup_matrix<T, 8, 8>((T)0);
        }

        // Main loop over K dimension with asynchronous copy
        for (uint k_base = 0; k_base < K; k_base += K_group) {
            // Declare threadgroup memory for A and B blocks
            threadgroup T* A_block = (threadgroup T*) &A_block_raw[0];
            threadgroup T* B_block = (threadgroup T*) &B_block_raw[0];

            // Allocate raw storage
            threadgroup char A_block_raw[M_group * K_group * sizeof(T)];
            threadgroup char B_block_raw[K_group * N_group * sizeof(T)];

            // Asynchronous copy of A and B into shared memory
            uint thread_id = thread_position_in_threadgroup;
            uint total_threads = threads_per_threadgroup;

            // Copy A into A_block
            for (uint idx = thread_id; idx < M_group * K_group; idx += total_threads) {
                uint m = idx / K_group;
                uint k = idx % K_group;
                uint global_m = M_offset + m;
                uint global_k = k_base + k;
                if (global_m < M && global_k < K) {
                    if (A_trans) {
                        A_block[idx] = A[global_k * A_ld + global_m];
                    } else {
                        A_block[idx] = A[global_m * A_ld + global_k];
                    }
                } else {
                    A_block[idx] = (T)0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Copy B into B_block
            for (uint idx = thread_id; idx < K_group * N_group; idx += total_threads) {
                uint k = idx / N_group;
                uint n = idx % N_group;
                uint global_k = k_base + k;
                uint global_n = N_offset + n;
                if (global_k < K && global_n < N) {
                    if (B_trans) {
                        B_block[idx] = B[global_n * B_ld + global_k];
                    } else {
                        B_block[idx] = B[global_k * B_ld + global_n];
                    }
                } else {
                    B_block[idx] = (T)0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Perform multiplication and accumulation
            for (ushort kk = 0; kk < K_group; kk += 8) {
                // Load A and B into simdgroup_matrix
                thread simdgroup_matrix<T, 8, 8> A_sram;
                thread simdgroup_matrix<T, 8, 8> B_sram;

                uint A_index = (offset_in_group.y) * K_group + kk;
                uint B_index = kk * N_group + offset_in_group.x;

                A_sram.load(&A_block[A_index], K_group);
                B_sram.load(&B_block[B_index], N_group);

                // Multiply and accumulate
                C_sram.multiply_add(A_sram, B_sram);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Store the result back to global memory
        uint2 C_offset = uint2(N_offset + offset_in_group.x,
                               M_offset + offset_in_group.y);
        device T* C_dst = C + C_offset.y * C_ld + C_offset.x;

        if (C_offset.x < N && C_offset.y < M) {
            C_sram.store(C_dst, C_ld);
        }
    }
    """

    # Now define the kernel
    kernel = mx.fast.metal_kernel(
        name="matmul_kernel",
        input_names=["A", "B"],
        output_names=["C"],
        source=source,
        header=header,
        ensure_row_contiguous=True,
    )

    return kernel
