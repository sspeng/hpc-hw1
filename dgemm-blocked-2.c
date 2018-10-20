const char *dgemm_desc = "Simple blocked dgemm.";
#define _POSIX_C_SOURCE 200809L
#include "arch.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#define BLOCK_SIZE 4

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define CEIL(u) ((u / 4) * 4)


void multiply_1x1(double *A, double *B, double *C, int lda){
    double tmp = C[0];
    for(int i = 0; i < lda; i++){
        tmp += A[i * lda] * B[i];
    }
    C[0] = tmp;
}

void multiply_1x4(double *A, double *B, double *C, int lda){
    for(int i = 0; i < lda; i++){
        double tmp = A[i*lda];
        C[0 * lda] += tmp * B[i + 0 * lda];
        C[1 * lda] += tmp * B[i + 1 * lda];
        C[2 * lda] += tmp * B[i + 2 * lda];
        C[3 * lda] += tmp * B[i + 3 * lda];
    }
}

void multiply_4x1(double *A, double *B, double *C, int lda){
    encode_t CC = _load(C);
    for(int i = 0; i < lda; i++){
        encode_t AA = _load(A + i * lda);
        encode_t BB = _mm256_set1_pd(B[i]);
        CC = _mm256_fmadd_pd(AA, BB, CC);
    }
    _store(C, CC);
}

void multiply_4x4(double *A, double *B, double *C, int lda){
    encode_t CC1 = _load(C + 0 * lda);
    encode_t CC2 = _load(C + 1 * lda);
    encode_t CC3 = _load(C + 2 * lda);
    encode_t CC4 = _load(C + 3 * lda);
    for(int i = 0; i < lda; i++){
        encode_t AA = _load(A + i * lda);
        encode_t BB1 = _mm256_set1_pd(B[i + 0 * lda]);
        encode_t BB2 = _mm256_set1_pd(B[i + 1 * lda]);
        encode_t BB3 = _mm256_set1_pd(B[i + 2 * lda]);
        encode_t BB4 = _mm256_set1_pd(B[i + 3 * lda]);

        CC1 = _mm256_fmadd_pd(AA, BB1, CC1);
        CC2 = _mm256_fmadd_pd(AA, BB2, CC2);
        CC3 = _mm256_fmadd_pd(AA, BB3, CC3);
        CC4 = _mm256_fmadd_pd(AA, BB4, CC4);
    }
    _store(C + 0 * lda, CC1);
    _store(C + 1 * lda, CC2);
    _store(C + 2 * lda, CC3);
    _store(C + 3 * lda, CC4);
}

//divied into 4 * 4 * 4, 4 * 4 * 1, 1 * 4 * 4, 1 * 1 * 1
void square_dgemm(int lda, double *A, double *B, double *C)
{
    /* For each block-row of A */
    for(int i = 0; i < CEIL(lda); i += 4){
        for(int j = 0; j < CEIL(lda); j += 4){
            multiply_4x4(A + i, B + j * lda, C + i + j * lda, lda);
        }
        for(int j = CEIL(lda); j < lda; j++){
            multiply_4x1(A + i, B + j * lda, C + i + j * lda, lda);
        }
    }
    for(int i = CEIL(lda); i < lda; i++){
        for(int j = 0; j < CEIL(lda); j += 4){
            multiply_1x4(A + i, B + j * lda, C + i + j * lda, lda);
        }
        for(int j = CEIL(lda); j < lda; j++){
            multiply_1x1(A + i, B + j * lda, C + i + j * lda, lda);
        }
    }
}
