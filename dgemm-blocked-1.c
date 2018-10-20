const char *dgemm_desc = "Simple blocked dgemm.";
#define _POSIX_C_SOURCE 200809L
#include "arch.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define floor(u) ((u * sizeof(double) + sizeof(encode_t) - 1) / sizeof(encode_t) * (sizeof(encode_t) / sizeof(double)))
#define fl(u) (u * sizeof(double) + sizeof(encode_t) - 1) / sizeof(encode_t)
/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, const int M, const int N, const int K, double **A, double **B, double **C)
{
    encode_t **input = (encode_t **)A;
    encode_t **output = (encode_t **)C;
    encode_t res[N][fl(M)];
    // for(int i = 0; i < N; i++){
    //     for(int j = 0; j < fl(M); j++){
    //         output[i][j] = zero();
    //     }
    // }
    for(int k = 0; k < K; k++){
        for(int j = 0; j < N; j++){
            double tmp = B[j][k]; 
            for(int i = 0; i < fl(M); i++){
                output[j][i] = add_256(output[j][i], multiply_256(input[k][i], tmp));
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C)
{
    /* For each block-row of A */
    double *AA[BLOCK_SIZE], *BB[BLOCK_SIZE], *CC[BLOCK_SIZE];
    for(int i = 0; i < BLOCK_SIZE; i++){
        posix_memalign((void **)&(AA[i]), 64, sizeof(double) * BLOCK_SIZE);
        posix_memalign((void **)&(BB[i]), 64, sizeof(double) * BLOCK_SIZE);
        posix_memalign((void **)&(CC[i]), 64, sizeof(double) * BLOCK_SIZE);
    }
    for (int i = 0; i < lda; i += BLOCK_SIZE)
        /* For each block-column of B */
        for (int j = 0; j < lda; j += BLOCK_SIZE)
            /* Accumulate block dgemms into block of C */
            for (int k = 0; k < lda; k += BLOCK_SIZE)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
        
                /* Perform individual block dgemm */
                for(int h = 0; h < K; h++){
                    memmove(AA[h], A + i + (k + h) * lda, sizeof(double) * M);
                    memset(AA[h] + M, 0, sizeof(double) * (min(floor(M), BLOCK_SIZE) - M));
                }
                for(int h = 0; h < N; h++){
                    memmove(BB[h], B + k + (j + h) * lda, sizeof(double) * K);
                    memmove(CC[h], C + i + (j + h) * lda, sizeof(double) * M);
                    memset(CC[h] + M, 0, sizeof(double) * (floor(M) - M));
                }
                do_block(lda, M, N, K, AA, BB, CC);
                for(int h = 0; h < N; h++){
                    memmove(C + i + (j + h) * lda, CC[h], sizeof(double) * M);
                }
            }
    for(int i = 0; i < BLOCK_SIZE; i++){
        if(AA[i] != NULL) free(AA[i]);
        if(BB[i] != NULL) free(BB[i]);
        if(CC[i] != NULL) free(CC[i]);
    }
}
