#include "ggml-dsp.h"

// 128 byte vectors
#define VSIZE_BYTES 128
#define VSIZE_WORDS VSIZE_BYTES/4

union ui32f { int32_t i; float f; };

// create a vector of floats from a float
static __attribute__((always_inline)) HVX_Vector create_sfv_from_sf(float value) {
    union ui32f cvt;
    cvt.f = value;
    HVX_Vector tmp = Q6_V_vsplat_R(cvt.i);
    return tmp;
}

// create a vector of qf32's from a float
static __attribute__((always_inline)) HVX_Vector create_qf32v_from_sf(float value) {
    HVX_Vector tmp = Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_vsplat_R(0), create_sfv_from_sf(value));
    return tmp;
}

// convert qf32 vector to float vector
static __attribute__((always_inline)) HVX_Vector convert_qf32v_to_fltv(HVX_Vector vect) {
    HVX_Vector tmp = Q6_Vsf_equals_Vqf32(vect);
    return tmp;
}

// get lowest float from a vector of floats
static __attribute__((always_inline)) float get_flt0_from_fltv(HVX_Vector vect) {
    union ui32f cvt;
    cvt.i = vect[0];
    return cvt.f;
}

// get lowest float from a vector of qf32's
static __attribute__((always_inline)) float get_flt0_from_qf32v(HVX_Vector vect) {
    union ui32f cvt;
    HVX_Vector tmp = convert_qf32v_to_fltv(vect);
    cvt.i = tmp[0];
    return cvt.f;
}

static void vec_dot_f32(int n, float *GGML_RESTRICT s, size_t bs, const float *GGML_RESTRICT x,
                    size_t bx, const float *GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    // scalar
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float) (x[i] * y[i]);
    }
    *s = sumf;
}

static void ggml_compute_forward_mul_mat_one_chunk(const ggml_tensor *src0, const ggml_tensor *src1,
                                                   struct ggml_tensor *dst,
                                                   const enum ggml_type type,
                                                   const int32_t num_rows_per_vec_dot,
                                                   const int32_t ir0_start, const int32_t ir0_end,
                                                   const int32_t ir1_start, const int32_t ir1_end) {
    ggmlhexagon_dump_tensor(src0, 0);
    ggmlhexagon_dump_tensor(src1, 0);
    ggmlhexagon_dump_tensor(dst, 0);

    dst->ne[0] = src0->ne[1];
    dst->ne[1] = src1->ne[1];
    dst->ne[2] = src1->ne[2];
    dst->ne[3] = src1->ne[3];

    dst->nb[0] = 4;
    dst->nb[1] = dst->nb[0] * dst->ne[0];
    dst->nb[2] = dst->nb[1] * dst->ne[1];
    dst->nb[3] = dst->nb[2] * dst->ne[2];
    ggmlhexagon_dump_tensor(dst, 0);

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    // broadcast factors
    const int32_t r2 = ne12 / ne02;
    const int32_t r3 = ne13 / ne03;

    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = src1->data;
    const size_t row_size = 4* ne10;

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int32_t blck_0 = 16;
    const int32_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int32_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int32_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int32_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int32_t i13 = (ir1 / (ne12 * ne1));
                const int32_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int32_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int32_t i03 = i13 / r3;
                const int32_t i02 = i12 / r2;

                const int32_t i1 = i11;
                const int32_t i2 = i12;
                const int32_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                const char * src1_col = (const char*)wdata +
                                        (src1_cont
                                         ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                                         : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                for (int32_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot_f32(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0),
                                (float*)(src0_row + ir0 * nb01), (num_rows_per_vec_dot > 1 ? nb01 : 0),
                                (float*)src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

//TODO: only support fp32 mulmat on cDSP
static int ggmlop_dsp_mulmat_singlethread(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    ggmlhexagon_dump_tensor(src0, 0);
    ggmlhexagon_dump_tensor(src1, 0);
    ggmlhexagon_dump_tensor(dst, 0);

    dst->ne[0] = src0->ne[1];
    dst->ne[1] = src1->ne[1];
    dst->ne[2] = src1->ne[2];
    dst->ne[3] = src1->ne[3];

    dst->nb[0] = 4;
    dst->nb[1] = dst->nb[0] * dst->ne[0];
    dst->nb[2] = dst->nb[1] * dst->ne[1];
    dst->nb[3] = dst->nb[2] * dst->ne[2];
    ggmlhexagon_dump_tensor(dst, 0);

    GGML_TENSOR_BINARY_OP_LOCALS

    int32_t  const vec_dot_num_rows     = 1;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == 4);
    GGML_ASSERT(nb10 == 4);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

#if 0 //naive algorithm for fp32, can pass various case in UT
    {
        //ggml_dump_tensor(src0);
        //ggml_dump_tensor(src1);

        float * a = (float*)src0->data;
        float * b = (float*)src1->data;
        float * c = (float*)dst->data;
        int M = src0->ne[1];
        int K = src0->ne[0];
        int N = src1->ne[1];
        float sum = 0;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                sum = 0;
                for (int h = 0; h < K; h++) {
                    sum += a[i * K + h] * b[h * N + j];
                }
                c[i * N + j] = sum;
            }
        }
        return 0;
    }
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int32_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int32_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int32_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int32_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 <  4) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 =  1; // parallelize by src0 rows
        nchunk1 =  1; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int32_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int32_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = 0;

    while (current_chunk < nchunk0 * nchunk1) {
        const int32_t ith0 = current_chunk % nchunk0;
        const int32_t ith1 = current_chunk / nchunk0;

        const int32_t ir0_start = dr0 * ith0;
        const int32_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int32_t ir1_start = dr1 * ith1;
        const int32_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int32_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(src0, src1, dst, src0->type, num_rows_per_vec_dot,
                                               ir0_start, ir0_end, ir1_start, ir1_end);

        if (1 >= nchunk0 * nchunk1) {
            break;
        }
        current_chunk++;
    }

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

//TODO:multithreading mulmat
static int ggmlop_dsp_mulmat_multithread(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

int ggmlop_dsp_mulmat(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst) {
    if (ggmlop_get_thread_counts() > 1) {
        return ggmlop_dsp_mulmat_multithread(h, src0, src1, dst);
    } else {
        return ggmlop_dsp_mulmat_singlethread(h, src0, src1, dst);
    }
}
