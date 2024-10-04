import mlx.core as mx


def matmul_spec(a: mx.array, b: mx.array):
    return mx.matmul(a, b)


def matmul_kernel(a: mx.array, b: mx.array):
    # Directly use the values from the input arrays
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    A_leading_dimension = a.shape[1]
    B_leading_dimension = b.shape[1]
    C_leading_dimension = b.shape[1]
    load_previous_C = False
    A_trans = False
    B_trans = False
    M_group = 64
    N_group = 64
    K_group = 64

    header = f"""
    #include <metal_stdlib>
    using namespace metal;

    typedef unsigned int uint;
    typedef unsigned short ushort;

    // Function constants for dynamic matrix dimensions and settings
    constant uint M = {M};
    constant uint N = {N};
    constant uint K = {K};

    constant uint A_leading_dimension = {A_leading_dimension};
    constant uint B_leading_dimension = {B_leading_dimension};
    constant uint C_leading_dimension = {C_leading_dimension};

    constant bool load_previous_C = {str(load_previous_C).lower()};
    constant bool A_trans = {str(A_trans).lower()};
    constant bool B_trans = {str(B_trans).lower()};

    constant ushort M_group = {M_group};
    constant ushort N_group = {N_group};
    constant ushort K_group = {K_group};
    """

    source = """
    #include <metal_stdlib>
    using namespace metal;

    typedef unsigned int uint;
    typedef unsigned short ushort;

    // Calculate edge thresholds and shifts
    constant uint M_edge = M - (M % M_group);
    constant uint N_edge = N - (N % N_group);

    constant ushort M_remainder = (M % 8 == 0) ? 8 : M % 8;
    constant ushort N_remainder = (N % 8 == 0) ? 8 : N % 8;
    constant ushort K_remainder = (K % K_group == 0) ? K_group : K % K_group;
    constant ushort K_remainder_padded = ((K_remainder + 7) / 8) * 8;

    constant ushort M_shift = (M < M_group) ? 0 : 8 - M_remainder;
    constant ushort N_shift = (N < N_group) ? 0 : 8 - N_remainder;

    // Function to index into an array of registers
    template <typename T>
    inline thread simdgroup_matrix<T, 8, 8>* get_sram(
        thread simdgroup_matrix<T, 8, 8> *sram,
        ushort sram_leading_dim,
        ushort2 matrix_origin
    ) {
        return sram + (matrix_origin.y / 8) * (sram_leading_dim / 8) + (matrix_origin.x / 8);
    }

    // Multiply-accumulate function
    void multiply_accumulate(
        threadgroup const float *A_block,
        threadgroup const float *B_block,
        thread simdgroup_matrix<float, 8, 8> *A_sram,
        thread simdgroup_matrix<float, 8, 8> *B_sram,
        thread simdgroup_matrix<float, 8, 8> *C_sram,
        ushort k,
        ushort2 offset_in_group
    ) {
        // Load A_sram
        #pragma unroll
        for (ushort m = 0; m < 8; m += 8) {
            ushort2 origin = ushort2(0, m);
            auto A_ptr = get_sram(A_sram, 8, origin);
            ushort A_block_index = ((offset_in_group.y + m) * K_group) + k;
            A_ptr->load(&A_block[A_block_index], K_group, ushort2(0, 0), A_trans);
        }

        // Load B_sram
        #pragma unroll
        for (ushort n = 0; n < 8; n += 8) {
            ushort2 origin = ushort2(n, 0);
            auto B_ptr = get_sram(B_sram, 8, origin);
            ushort B_block_index = (k * N_group) + (offset_in_group.x + n);
            B_ptr->load(&B_block[B_block_index], N_group, ushort2(0, 0), B_trans);
        }

        // Multiply and accumulate
        #pragma unroll
        for (ushort m = 0; m < 8; m += 8) {
            #pragma unroll
            for (ushort n = 0; n < 8; n += 8) {
                auto A_ptr = get_sram(A_sram, 8, ushort2(0, m));
                auto B_ptr = get_sram(B_sram, 8, ushort2(n, 0));
                auto C_ptr = get_sram(C_sram, 8, ushort2(n, m));
                C_ptr->multiply_add(*A_ptr, *B_ptr);
            }
        }
    }

    // Main kernel function
    kernel void gemm(
        device const float *A [[buffer(0)]],
        device const float *B [[buffer(1)]],
        device float *C [[buffer(2)]],
        threadgroup uchar *threadgroup_block [[threadgroup(0)]],
        uint3 gid [[threadgroup_position_in_grid]],
        ushort sidx [[simdgroup_index_in_threadgroup]],
        ushort lane_id [[thread_index_in_simdgroup]],
        uint threads_per_threadgroup [[threads_per_threadgroup]],
        uint thread_position_in_threadgroup [[thread_position_in_threadgroup]]
    ) {
        ushort2 sid = ushort2(sidx % (N_group / 8), sidx / (N_group / 8));
        ushort2 morton_offset = ushort2(lane_id % 8, lane_id / 8);

        // Return early if SIMD is out of bounds
        uint M_offset = gid.y * M_group;
        uint N_offset = gid.x * N_group;
        if (M_offset + sid.y * 8 >= M || N_offset + sid.x * 8 >= N) {
            return;
        }
        ushort2 offset_in_group = ushort2(sid.x * 8 + morton_offset.x,
                                          sid.y * 8 + morton_offset.y);

        // Shift the matrix block within bounds, if possible
        if ((M_shift != 0) && (gid.y * M_group >= M_edge)) {
            M_offset -= M_shift;
        }
        if ((N_shift != 0) && (gid.x * N_group >= N_edge)) {
            N_offset -= N_shift;
        }

        // Initialize accumulator
        thread simdgroup_matrix<float, 8, 8> C_sram[(8 / 8) * (8 / 8)];
        if (load_previous_C) {
            // Load previous C values
            uint2 C_offset = uint2(N_offset + offset_in_group.x,
                                   M_offset + offset_in_group.y);
            device const float *C_src = C + C_offset.y * C_leading_dimension + C_offset.x;
            #pragma unroll
            for (ushort m = 0; m < 8; m += 8) {
                #pragma unroll
                for (ushort n = 0; n < 8; n += 8) {
                    ushort2 origin = ushort2(n, m);
                    auto C_ptr = get_sram(C_sram, 8, origin);
                    if (C_offset.x + n < N && C_offset.y + m < M) {
                        C_ptr->load(C_src, C_leading_dimension, origin);
                    } else {
                        *C_ptr = simdgroup_matrix<float, 8, 8>(0.0);
                    }
                }
            }
        } else {
            // Initialize C_sram to zero
            #pragma unroll
            for (ushort m = 0; m < 8; m += 8) {
                #pragma unroll
                for (ushort n = 0; n < 8; n += 8) {
                    ushort2 origin = ushort2(n, m);
                    auto C_ptr = get_sram(C_sram, 8, origin);
                    *C_ptr = simdgroup_matrix<float, 8, 8>(0.0);
                }
            }
        }

        // Main loop over K dimension with asynchronous copy
        for (uint k_base = 0; k_base < K; k_base += K_group) {
            threadgroup float *A_block = (threadgroup float *)threadgroup_block;
            threadgroup float *B_block = A_block + M_group * K_group;

            // Asynchronous copy of A and B into shared memory
            uint thread_id = thread_position_in_threadgroup.x;
            uint total_threads = threads_per_threadgroup;

            // Copy A into A_block
            for (uint idx = thread_id; idx < M_group * K_group; idx += total_threads) {
                uint m = idx / K_group;
                uint k = idx % K_group;
                uint global_m = M_offset + m;
                uint global_k = k_base + k;
                if (global_m < M && global_k < K) {
                    if (A_trans) {
                        A_block[idx] = A[global_k * A_leading_dimension + global_m];
                    } else {
                        A_block[idx] = A[global_m * A_leading_dimension + global_k];
                    }
                } else {
                    A_block[idx] = 0.0;
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
                        B_block[idx] = B[global_n * B_leading_dimension + global_k];
                    } else {
                        B_block[idx] = B[global_k * B_leading_dimension + global_n];
                    }
                } else {
                    B_block[idx] = 0.0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Perform multiplication and accumulation
            for (ushort k = 0; k < K_group; k += 8) {
                thread simdgroup_matrix<float, 8, 8> A_sram[(8 / 8) * (8 / 8)];
                thread simdgroup_matrix<float, 8, 8> B_sram[(8 / 8) * (8 / 8)];

                multiply_accumulate(
                    A_block,
                    B_block,
                    A_sram,
                    B_sram,
                    C_sram,
                    k,
                    offset_in_group
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Store the result back to global memory
        uint2 C_offset = uint2(N_offset + offset_in_group.x,
                               M_offset + offset_in_group.y);
        device float *C_dst = C + C_offset.y * C_leading_dimension + C_offset.x;

        #pragma unroll
        for (ushort m = 0; m < 8; m += 8) {
            #pragma unroll
            for (ushort n = 0; n < 8; n += 8) {
                ushort2 origin = ushort2(n, m);
                auto C_ptr = get_sram(C_sram, 8, origin);
                if (C_offset.x + n < N && C_offset.y + m < M) {
                    C_ptr->store(C_dst, C_leading_dimension, origin);
                }
            }
        }
    }
    """

    kernel = mx.fast.metal_kernel(
        name="gemm",
        input_names=["A", "B"],
        output_names=["C"],
        header=header,
        source=source,
    )

    return kernel
