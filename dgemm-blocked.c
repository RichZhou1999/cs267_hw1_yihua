#include "x86intrin.h"

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef BLOCK_SIZE_INNER
#define BLOCK_SIZE_INNER 256
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */

//static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
//    // For each row i of A
//    __m256d va,vb,vc;
//    for (int k = 0; k < K; ++k) {
//        // For each column j of B
//        for (int j = 0; j < N; ++j) {
//            // Compute C(i,j)
//            vb = _mm256_broadcast_sd(&B[k + j*lda]);
//            for (int i = 0; i < (M/4) * 4; i+=4) {
//                va = _mm256_loadu_pd(&A[i * lda + k]);
//                vc = _mm256_loadu_pd(&C[i + j * lda]);
//                vc = _mm256_fmadd_pd(va, vb, vc);
//                _mm256_storeu_pd( &C[i + j*lda], vc );
//            }
//            for (int i =(M/4) * 4; i < M;++i ){
//                double cij = C[i + j * lda];
//                cij += A[i + k * lda] * B[k*lda + j];
//                C[i + j * lda] = cij;
//            }
//        }
//    }
//
//}


//static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
//    // For each row i of A
//    __m256d va1,vb1,vc1;
//    for (int k = 0; k < K; ++k) {
//        // For each column j of B
//        for (int j = 0; j < N; ++j) {
////            __m256d result_list[(M/4)*4];
//            for (int i = 0; i < (M/4)*4; i+=4 ){
//                va1 = _mm256_loadu_pd(&A[i + k * lda]);
//                vb1 = _mm256_broadcast_sd(&B[k + j*lda]);
//                vc1 = _mm256_loadu_pd(&C[i + j * lda]);
//                vc1 = _mm256_fmadd_pd(va1, vb1, vc1);
////                result_list[(i/4)] = vc1;
//                _mm256_storeu_pd( &C[i + j*lda], vc1 );
//            }
////            for (int i = 0; i < (M/4)*4; i+=4 ){
////                _mm256_storeu_pd( &C[i + j*lda], result_list[(i/4)] );
////            }
//            for (int i = (M/4)*4; i < M;++i ){
//                double cij = C[i + j * lda];
//                cij += A[i + k*lda] * B[k + j *lda];
//                C[i + j * lda] = cij;
//            }
//        }
//    }
//
//}


//static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
//    // For each row i of A
//    for (int i = 0; i < (M/4)*4; i += 4){
//        for (int j = 0; j < (N/4)*4; j += 4){
//            __m256d vc0 = _mm256_loadu_pd(&C[i + j * lda]);
//            __m256d vc1 = _mm256_loadu_pd(&C[i + (j+1) * lda]);
//            __m256d vc2 = _mm256_loadu_pd(&C[i + (j+2) * lda]);
//            __m256d vc3 = _mm256_loadu_pd(&C[i + (j+3) * lda]);
//            for (int k = 0; k < K; k++){
//                __m256d va = _mm256_loadu_pd(&A[i + k * lda]);
//                __m256d vb0 = _mm256_broadcast_sd(&B[k * lda + j]);
//                __m256d vb1 = _mm256_broadcast_sd(&B[k * lda + j + 1]);
//                __m256d vb2 = _mm256_broadcast_sd(&B[k * lda + j + 2]);
//                __m256d vb3 = _mm256_broadcast_sd(&B[k * lda + j + 3]);
//                vc0 = _mm256_fmadd_pd(va, vb0, vc0);
//                vc1 = _mm256_fmadd_pd(va, vb1, vc1);
//                vc2 = _mm256_fmadd_pd(va, vb2, vc2);
//                vc3 = _mm256_fmadd_pd(va, vb3, vc3);
//            }
//            _mm256_storeu_pd( &C[i + j * lda], vc0 );
//            _mm256_storeu_pd( &C[i + (j+1)*lda], vc1 );
//            _mm256_storeu_pd( &C[i + (j+2)*lda], vc2 );
//            _mm256_storeu_pd( &C[i + (j+3)*lda], vc3 );
//        }
//        for (int j = (N/4)*4; j < N; j++){
//            for (int k = 0; k < K; k++){
//                double cij = C[i + j * lda];
//                cij += A[i + k*lda] * B[k*lda + j];
//                C[i + j * lda] = cij;
//            }
//        }
//
//
//    }
//    for (int i = (M/4)*4; i < M; i++){
//        for (int j = 0; j < N; j++) {
//            for (int k = 0; k < K; k++) {
//                double cij = C[i + j * lda];
//                cij += A[i + k * lda] * B[k * lda + j];
//                C[i + j * lda] = cij;
//            }
//        }
//    }
//
//}
//

static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each row i of A
    for (int i = 0; i < (M/4)*4; i += 4) {
        for (int j = 0; j < N; j++) {
            double cij0 = C[i + j * lda];
            double cij1 = C[i+1 + j * lda];
            double cij2 = C[i+2 + j * lda];
            double cij3 = C[i+3 + j * lda];
            for (int k = 0; k < K; k++) {
                cij0 += A[i + k * lda] * B[k * lda + j];
                cij1 += A[i + 1 + k * lda] * B[k * lda + j];
                cij2 += A[i + 2 + k * lda] * B[k * lda + j];
                cij3 += A[i + 3 + k * lda] * B[k * lda + j];
            }
            C[i + j * lda] = cij0;
            C[i + 1 + j * lda] = cij1;
            C[i + 2 + j * lda] = cij2;
            C[i + 3 + j * lda] = cij3;
        }
    }
    for (int i = (M/4)*4; i < M; i++){
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
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

//    double A_column_list[lda*lda];
//    for (int i =0; i <lda; i += 1){
//        for(int j = 0; j < lda; j+=1){
//            A_column_list[i*lda + j] = *(A + (j*lda) + i);
//        }
//    }
    //use a pointer point to the head of the column-wised array B
    double* B_column = B_column_list;
//    double* A_column = A_column_list;

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
//                            do_block(lda, M0, N0, K0, A + (i+i0) + (k+k0) * lda, B + (k+k0) + (j+j0)*lda, C + i+i0 + (j+j0) * lda);
//                        }
//                    }
//                }
                do_block(lda, M, N, K, A + i + k*lda, B_column + k*lda + j , C + i + j * lda);
                // Perform individual block dgemm
            }
        }
    }
}
