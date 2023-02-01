#include "x86intrin.h"

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
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


static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each row i of A
    for (int i = 0; i < (M/8)*8; i += 8){
        for (int j = 0; j < (N/6)*6; j += 6){
            __m256d vc0 = _mm256_loadu_pd(&C[i + j * lda]);
            __m256d vc1 = _mm256_loadu_pd(&C[i + (j+1) * lda]);
            __m256d vc2 = _mm256_loadu_pd(&C[i + (j+2) * lda]);
            __m256d vc3 = _mm256_loadu_pd(&C[i + (j+3) * lda]);
            __m256d vc4 = _mm256_loadu_pd(&C[i + (j+4) * lda]);
            __m256d vc5 = _mm256_loadu_pd(&C[i + (j+5) * lda]);
//            __m256d vc6 = _mm256_loadu_pd(&C[i + (j+6) * lda]);
//            __m256d vc7 = _mm256_loadu_pd(&C[i + (j+7) * lda]);

            __m256d vc10 = _mm256_loadu_pd(&C[i + 4 + j * lda]);
            __m256d vc11 = _mm256_loadu_pd(&C[i + 4+(j+1) * lda]);
            __m256d vc12 = _mm256_loadu_pd(&C[i + 4+(j+2) * lda]);
            __m256d vc13 = _mm256_loadu_pd(&C[i + 4+(j+3) * lda]);
            __m256d vc14 = _mm256_loadu_pd(&C[i + 4+(j+4) * lda]);
            __m256d vc15 = _mm256_loadu_pd(&C[i + 4+(j+5) * lda]);
//            __m256d vc16 = _mm256_loadu_pd(&C[i + 4+(j+6) * lda]);
//            __m256d vc17 = _mm256_loadu_pd(&C[i + 4+(j+7) * lda]);
            for (int k = 0; k < K; k++){
                __m256d va = _mm256_loadu_pd(&A[i + k * lda]);
                __m256d va2 = _mm256_loadu_pd(&A[i + 4 + k * lda]);
                __m256d vb0 = _mm256_broadcast_sd(&B[k * lda + j]);
                __m256d vb1 = _mm256_broadcast_sd(&B[k * lda + j + 1]);
                __m256d vb2 = _mm256_broadcast_sd(&B[k * lda + j + 2]);
                __m256d vb3 = _mm256_broadcast_sd(&B[k * lda + j + 3]);
                __m256d vb4 = _mm256_broadcast_sd(&B[k * lda + j +4]);
                __m256d vb5 = _mm256_broadcast_sd(&B[k * lda + j + 5]);
//                __m256d vb6 = _mm256_broadcast_sd(&B[k * lda + j + 6]);
//                __m256d vb7 = _mm256_broadcast_sd(&B[k * lda + j + 7]);

                vc0 = _mm256_fmadd_pd(va, vb0, vc0);
                vc1 = _mm256_fmadd_pd(va, vb1, vc1);
                vc2 = _mm256_fmadd_pd(va, vb2, vc2);
                vc3 = _mm256_fmadd_pd(va, vb3, vc3);
                vc4 = _mm256_fmadd_pd(va, vb4, vc4);
                vc5 = _mm256_fmadd_pd(va, vb5, vc5);
//                vc6 = _mm256_fmadd_pd(va, vb6, vc6);
//                vc7 = _mm256_fmadd_pd(va, vb7, vc7);

                vc10 = _mm256_fmadd_pd(va2, vb0, vc10);
                vc11 = _mm256_fmadd_pd(va2, vb1, vc11);
                vc12 = _mm256_fmadd_pd(va2, vb2, vc12);
                vc13 = _mm256_fmadd_pd(va2, vb3, vc13);
                vc14 = _mm256_fmadd_pd(va2, vb4, vc14);
                vc15 = _mm256_fmadd_pd(va2, vb5, vc15);
//                vc16 = _mm256_fmadd_pd(va2, vb6, vc16);
//                vc17 = _mm256_fmadd_pd(va2, vb7, vc17);

            }
            _mm256_storeu_pd( &C[i + j * lda], vc0 );
            _mm256_storeu_pd( &C[i + (j+1)*lda], vc1 );
            _mm256_storeu_pd( &C[i + (j+2)*lda], vc2 );
            _mm256_storeu_pd( &C[i + (j+3)*lda], vc3 );
            _mm256_storeu_pd( &C[i + (j+4) * lda], vc4 );
            _mm256_storeu_pd( &C[i + (j+5)*lda], vc5 );
//            _mm256_storeu_pd( &C[i + (j+6)*lda], vc6 );
//            _mm256_storeu_pd( &C[i + (j+7)*lda], vc7 );

            _mm256_storeu_pd( &C[i + 4 + j * lda], vc10 );
            _mm256_storeu_pd( &C[i + 4+ (j+1)*lda], vc11 );
            _mm256_storeu_pd( &C[i + 4+(j+2)*lda], vc12 );
            _mm256_storeu_pd( &C[i + 4+(j+3)*lda], vc13 );
            _mm256_storeu_pd( &C[i + 4+(j+4) * lda], vc14 );
            _mm256_storeu_pd( &C[i + 4+(j+5)*lda], vc15 );
//            _mm256_storeu_pd( &C[i + 4+(j+6)*lda], vc16 );
//            _mm256_storeu_pd( &C[i + 4+(j+7)*lda], vc17 );
        }

        for (int j = (N/6)*6; j < N; j++) {
            double cij0 = C[i + j * lda];
            double cij1 = C[i + 1 + j * lda];
            double cij2 = C[i + 2 + j * lda];
            double cij3 = C[i + 3 + j * lda];
            double cij4 = C[i + 4 + j * lda];
            double cij5 = C[i + 5 + j * lda];
            double cij6 = C[i + 6 + j * lda];
            double cij7 = C[i + 7 + j * lda];
            for (int k = 0; k < K; k++) {
                cij0 += A[i + k * lda] * B[k * lda + j];
                cij1 += A[i + 1 + k * lda] * B[k * lda + j];
                cij2 += A[i + 2 + k * lda] * B[k * lda + j];
                cij3 += A[i + 3 +k * lda] * B[k * lda + j];
                cij4 += A[i + 4 + k * lda] * B[k * lda + j];
                cij5 += A[i + 5 + k * lda] * B[k * lda + j];
                cij6 += A[i + 6 + k * lda] * B[k * lda + j];
                cij7 += A[i + 7 +k * lda] * B[k * lda + j];
            }
            C[i + j * lda] = cij0;
            C[i + 1 + j * lda] = cij1;
            C[i + 2 + j * lda] = cij2;
            C[i + 3 + j * lda] = cij3;
            C[i + 4 + j * lda] = cij4;
            C[i + 5 + j * lda] = cij5;
            C[i + 6 + j * lda] = cij6;
            C[i + 7 + j * lda] = cij7;
        }

    }
    for (int i = (M/8)*8; i < M; i++){
        for (int j = 0; j < N; j++) {
            double cij = C[i + j * lda];
            for (int k = 0; k < K; k++) {
                cij += A[i + k * lda] * B[k * lda + j];
            }
            C[i + j * lda] = cij;
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

    for (int i = 0; i < lda; i+= BLOCK_SIZE) {
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
