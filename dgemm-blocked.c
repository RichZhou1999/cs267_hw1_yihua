#include "x86intrin.h"

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef BLOCK_SIZE_INNER
#define BLOCK_SIZE_INNER 32
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */

static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each row i of A
    __m256d va,vb,vc;
    for (int k = 0; k < (K/4) *4; ++k) {
        // For each column j of B
        for (int j = 0; j < (N/4)*4; ++j) {
            // Compute C(i,j)
            for (int i = 0; i < (M/4) * 4; i+=4) {
//                double cij = C[i + j * lda];
//                cij += A[i + k * lda] * B[k*lda + j];
//                C[i + j * lda] = cij;
                vb = _mm256_loadu_pd(&B[k * lda + j]);
                va = _mm256_loadu_pd(&A[i + k*lda]);
                vc = _mm256_loadu_pd(&C[i + j*lda]);
                vc = _mm256_fmadd_pd(vc, va, vb);
                _mm256_storeu_pd( &C[i + j*lda], vc );
            }
            for (int i =(M/4) * 4; i < M;++i ){
                double cij = C[i + j * lda];
                cij += A[i + k * lda] * B[k*lda + j];
                C[i + j * lda] = cij;
            }
        }
        for (int j = (N/4)*4; j < N; ++j) {
            for (int i = 0; i < M; ++i) {
                double cij = C[i + j * lda];
                cij += A[i + k * lda] * B[k * lda + j];
                C[i + j * lda] = cij;
            }
        }

    }
    for (int k = (K / 4) * 4 ; k < K; ++k) {
        // For each column j of B
        for (int j = 0 ; j < N; ++j) {
            for (int i = 0; i < M; ++i) {
                double cij = C[i + j * lda];
                cij += A[i + k * lda] * B[k * lda + j];
                C[i + j * lda] = cij;
            }
        }
    }

}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {

    // get the column-wised B 1D array
    double B_column_list[lda*lda];
    for (int i =0; i <lda; i += 1){
        for(int j = 0; j < lda; j+=1){
            B_column_list[i*lda + j] = *(B + (j*lda) + i);
        }
    }
    //use a pointer point to the head of the column-wised array B
    double* B_column = B_column_list;

    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
//                for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE_INNER){
//                    for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE_INNER){
//                        for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE_INNER) {
//                            int M0 = min(BLOCK_SIZE_INNER, M - i0);
//                            int N0 = min(BLOCK_SIZE_INNER, N - j0);
//                            int K0 = min(BLOCK_SIZE_INNER, K - k0);
//                            do_block(lda, M0, N0, K0, A + (i+i0) + (k+k0) * lda, B_column + (k+k0)*lda + (j+j0), C + i+i0 + (j+j0) * lda);
//                        }
//                    }
//                }
                do_block(lda, M, N, K, A + i + k * lda, B_column + k*lda + j , C + i + j * lda);
                // Perform individual block dgemm
            }
        }
    }
}
