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
    printf("in do block %d %d %d %d\n", M, N, K, fl(M));
    encode_t res[N][fl(M)];
    for(int i = 0; i < N; i++){
        for(int j = 0; j < fl(M); j++){
            res[i][j] = zero();
        }
    }
    for(int k = 0; k < K; k++){
        for(int j = 0; j < N; j++){
            for(int i = 0; i < fl(M); i++){
                res[j][i] = add_256(res[j][i], multiply_256(input[k][i], B[j][k]));
            }
        }
    }
    for(int i = 0; i < N; i++){
        for(int j = 0; j < fl(M); j++){
            store(&C[i][j * sizeof(encode_t)], res[i][j]);
            //printf("%.1f\n", C[i][j * sizeof(encode_t)]);
        }
    }
    printf("do_block over\n");
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
                // double a[8];
                // double b = 3.0;
                // double c[8];
                // posix_memalign((void **)&a, 64, sizeof(double) * 8);
                // posix_memalign((void **)&c, 64, sizeof(double) * 8);
                // a[0] = a[1] = a[2] = a[3] = 4.0;
                // c[0] = c[1] = c[2] = c[3] = 0.0;
                // encode_t* d = (encode_t *)a;
                // printf("%f %f %f %f %d %d\n", c[0],c[1],c[2],c[3], (int)sizeof(double), (int)sizeof(encode_t));
                // printf("1\n");
                // encode_t f = multiply_256(d[0], b);
                // _mm256_stream_pd(c, f);   
                // printf("%f %f %f %f\n", c[0],c[1],c[2],c[3]);

                /* Perform individual block dgemm */
                for(int h = 0; h < K; h++){
                    memmove(AA[h], A + i + (k + h) * lda, sizeof(double) * M);
                    memset(AA[h] + M, 0, sizeof(double) * (floor(M) - M));
                }
                for(int h = 0; h < N; h++){
                    memmove(BB[h], B + k + (j + h) * lda, sizeof(double) * K);
                    memset(CC[h], 0, sizeof(double) * floor(M));
                }
                printf("floor M %d\n", floor(M));
                // printf("1");
                // for(int ii = 0; ii < M; ii++){
                //     printf("\n");
                //     for(int jj = 0; jj < K; jj++){
                //         printf("%.1f ",AA[jj][ii]);
                //     }
                // }
                //printf("do_block\n");
                do_block(lda, M, N, K, AA, BB, CC);
                printf("2\n");
                for(int h = 0; h < N; h++){
                    for(int hh = 0; hh < M; hh++){
                        C[i + hh + (j + h) * lda] += CC[h][hh];
                    }
                }
                break;
            }
    printf("123123\n");
    for(int i = 0; i < BLOCK_SIZE; i++){
        free(AA[i]);
        printf("123123\n");
        free(BB[i]);
        printf("123123\n");
        //free(CC[i]);
        printf("123123\n");
    }
}
