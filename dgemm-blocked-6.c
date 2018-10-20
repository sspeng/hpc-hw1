const char *dgemm_desc = "Simple blocked dgemm.";
#define _POSIX_C_SOURCE 200809L
#include "arch.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#define BLOCK_SIZE 4

#define min(a, b) (((a) < (b)) ? (a) : (b))

double _A[8 * 1025 + 10];
double _B[1025 * 1025 + 10];

inline void multiply_1x1(double *A, double *B, double *C, int lda, int aa, int bb){
    double tmp = C[0];
    for(int i = 0; i < lda; i++){
        tmp += A[i * aa] * B[i * bb];
    }
    C[0] = tmp;
}

inline void multiply_1x4(double *A, double *B, double *C, int lda, int aa, int bb){
    for(int i = 0; i < lda; i++){
        double tmp = A[i * aa];
        C[0 * lda] += tmp * B[i * bb + 0];
        C[1 * lda] += tmp * B[i * bb + 1];
        C[2 * lda] += tmp * B[i * bb + 2];
        C[3 * lda] += tmp * B[i * bb + 3];
    }
}

inline void multiply_2x1(double *A, double *B, double *C, int lda, int aa, int bb){
    __m128d AA, BB, CC;
    CC = _load128(C);
    for(int i = 0; i < lda; i++){
        AA = _load128(A + i * aa);
        BB = _mm_set1_pd(B[i * bb]);
        CC = _mm_fmadd_pd(AA, BB, CC);
    }
    _store128(C, CC);
}

inline void multiply_2x4(double *A, double *B, double *C, int lda, int aa, int bb){
    __m128d BB1, BB2, BB3, BB4;
    __m128d CC1, CC2, CC3, CC4;
    CC1 = _load128(C + 0 * lda);
    CC2 = _load128(C + 1 * lda);
    CC3 = _load128(C + 2 * lda);
    CC4 = _load128(C + 3 * lda);
    for(int i = 0; i < lda; i++){
        __m128d AA = _load128(A + i * aa);
        BB1 = _mm_set1_pd(B[i * bb + 0]);
        BB2 = _mm_set1_pd(B[i * bb + 1]);
        BB3 = _mm_set1_pd(B[i * bb + 2]);
        BB4 = _mm_set1_pd(B[i * bb + 3]);

        CC1 = _mm_fmadd_pd(AA, BB1, CC1);
        CC2 = _mm_fmadd_pd(AA, BB2, CC2);
        CC3 = _mm_fmadd_pd(AA, BB3, CC3);
        CC4 = _mm_fmadd_pd(AA, BB4, CC4);
    }
    _store128(C + 0 * lda, CC1);
    _store128(C + 1 * lda, CC2);
    _store128(C + 2 * lda, CC3);
    _store128(C + 3 * lda, CC4);
}

inline void multiply_4x1(double *A, double *B, double *C, int lda, int aa, int bb){
    encode_t AA, BB, CC;
    CC = _load(C);
    for(int i = 0; i < lda; i++){
        AA = _load(A + i * aa);
        BB = _mm256_broadcast_sd(B + i * bb);
        CC = _mm256_fmadd_pd(AA, BB, CC);
    }
    _store(C, CC);
}

inline void multiply_4x4(double *A, double *B, double *C, int lda, int aa, int bb){
    encode_t BB1, BB2, BB3, BB4;
    encode_t CC1, CC2, CC3, CC4;
    CC1 = _load(C + 0 * lda);
    CC2 = _load(C + 1 * lda);
    CC3 = _load(C + 2 * lda);
    CC4 = _load(C + 3 * lda);
    for(int i = 0; i < lda; i++){
        encode_t AA = _load(A + i * aa);
        BB1 = _mm256_broadcast_sd(B + i * bb + 0);
        BB2 = _mm256_broadcast_sd(B + i * bb + 1);
        BB3 = _mm256_broadcast_sd(B + i * bb + 2);
        BB4 = _mm256_broadcast_sd(B + i * bb + 3);

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

inline void multiply_8x4(double *A, double *B, double *C, int lda, int aa, int bb){
    encode_t AA1, AA2; 
    encode_t BB1, BB2, BB3, BB4;
    encode_t CC1, CC2, CC3, CC4, CC5, CC6, CC7, CC8;

    CC1 = _load(C + 0 * lda);
    CC2 = _load(C + 1 * lda);
    CC3 = _load(C + 2 * lda);
    CC4 = _load(C + 3 * lda);

    CC5 = _load(C + 4 + 0 * lda);
    CC6 = _load(C + 4 + 1 * lda);
    CC7 = _load(C + 4 + 2 * lda);
    CC8 = _load(C + 4 + 3 * lda);
    for(int i = 0; i < lda; i++){
        AA1 = _load(A + i * aa);
        AA2 = _load(A + 4 + i * aa);
        
        BB1 = _mm256_broadcast_sd(B + i * bb + 0);
        BB2 = _mm256_broadcast_sd(B + i * bb + 1);
        BB3 = _mm256_broadcast_sd(B + i * bb + 2);
        BB4 = _mm256_broadcast_sd(B + i * bb + 3);

        CC1 = _mm256_fmadd_pd(AA1, BB1, CC1);
        CC2 = _mm256_fmadd_pd(AA1, BB2, CC2);
        CC3 = _mm256_fmadd_pd(AA1, BB3, CC3);
        CC4 = _mm256_fmadd_pd(AA1, BB4, CC4);

        CC5 = _mm256_fmadd_pd(AA2, BB1, CC5);
        CC6 = _mm256_fmadd_pd(AA2, BB2, CC6);
        CC7 = _mm256_fmadd_pd(AA2, BB3, CC7);
        CC8 = _mm256_fmadd_pd(AA2, BB4, CC8);
    }
    _store(C + 0 * lda, CC1);
    _store(C + 1 * lda, CC2);
    _store(C + 2 * lda, CC3);
    _store(C + 3 * lda, CC4);

    _store(C + 4 + 0 * lda, CC5);
    _store(C + 4 + 1 * lda, CC6);
    _store(C + 4 + 2 * lda, CC7);
    _store(C + 4 + 3 * lda, CC8);
}

inline void multiply_8x1(double *A, double *B, double *C, int lda, int aa, int bb){
    encode_t AA1, AA2, BB, CC1, CC2;    
    CC1 = _load(C);
    CC2 = _load(C + 4);
    for(int i = 0; i < lda; i++){
        AA1 = _load(A + i * aa);
        AA2 = _load(A + 4 + i * aa);
        BB = _mm256_set1_pd(B[i * bb]);
        CC1 = _mm256_fmadd_pd(AA1, BB, CC1);
        CC2 = _mm256_fmadd_pd(AA2, BB, CC2);
    }
    _store(C, CC1);
    _store(C + 4, CC2);
}

inline void pack_A(double *A, double *_A, int r, int c){
    for(int i = 0; i < c; i++){
        memmove(_A + i * r, A + i * c, sizeof(double) * r);
    }
}

inline void pack_B(double *B, double *_B, int r, int c){
    for(int i = 0; i < c; i++){
        for(int j = 0; j < r; j++){
            _B[i * r + j] = B[i + j * c];
        }
    }
}

//divied into 4 * 4 * 4, 4 * 4 * 1, 1 * 4 * 4, 1 * 1 * 1
void square_dgemm(int lda, double *A, double *B, double *C)
{
    for(int i = 0; i < lda / 4 * 4; i += 4){
        pack_B(B + i * lda, _B + i * lda, 4, lda);
    }
    for(int i = lda / 4 * 4; i < lda; i++){
        pack_B(B + i * lda, _B + i * lda, 1, lda);
    }
    // 8 * 4
    for(int i = 0; i < lda / 8 * 8; i += 8){
        pack_A(A + i, _A, 8, lda);
        for(int j = 0; j < lda / 4 * 4; j += 4){
            multiply_8x4(_A, _B + j * lda, C + i + j * lda, lda, 8, 4);
        }
        for(int j = lda / 4 * 4; j < lda; j++){
            multiply_8x1(_A, _B + j * lda, C + i + j * lda, lda, 8, 1);
        }
    }

    // 4* 4
    for(int i = lda / 8 * 8; i < lda / 4 * 4; i += 4){
        pack_A(A + i, _A, 4, lda);
        for(int j = 0; j < lda / 4 * 4; j += 4){
            multiply_4x4(_A, _B + j * lda, C + i + j * lda, lda, 4, 4);
        }
        for(int j = lda / 4 * 4; j < lda; j++){
            multiply_4x1(_A, _B + j * lda, C + i + j * lda, lda, 4, 1);
        }
    }

    // 2 * 4
    for(int i = lda / 4 * 4; i < lda / 2 * 2; i += 2){
        pack_A(A + i, _A, 2, lda);
        for(int j = 0; j < lda / 4 * 4; j += 4){
            multiply_2x4(_A, _B + j * lda, C + i + j * lda, lda, 2, 4);
        }
        for(int j = lda / 4 * 4; j < lda; j++){
            multiply_2x1(_A, _B + j * lda, C + i + j * lda, lda, 2, 1);
        }
    }

    //1 * 4
    for(int i = lda / 2 * 2; i < lda; i++){
        pack_A(A + i, _A, 1, lda);
        for(int j = 0; j < lda / 4 * 4; j += 4){
            multiply_1x4(_A, _B + j * lda, C + i + j * lda, lda, 1, 4);
        }
        for(int j = lda / 4 * 4; j < lda; j++){
            multiply_1x1(_A, _B + j * lda, C + i + j * lda, lda, 1, 1);
        }
    }
}
