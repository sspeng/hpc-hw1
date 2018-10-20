const char *dgemm_desc = "Simple blocked dgemm.";
#define _POSIX_C_SOURCE 200809L
#include "arch.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#define BLOCK_SIZE 4

#define min(a, b) (((a) < (b)) ? (a) : (b))

double _A[8 * 1025 + 10];

inline void multiply_1x1(double *A, double *B, double *C, int lda){
    double tmp = C[0];
    for(int i = 0; i < lda; i++){
        tmp += A[i] * B[i];
    }
    C[0] = tmp;
}

inline void multiply_1x4(double *A, double *B, double *C, int lda){
    for(int i = 0; i < lda; i++){
        double tmp = A[i];
        C[0 * lda] += tmp * B[i + 0 * lda];
        C[1 * lda] += tmp * B[i + 1 * lda];
        C[2 * lda] += tmp * B[i + 2 * lda];
        C[3 * lda] += tmp * B[i + 3 * lda];
    }
}

inline void multiply_2x1(double *A, double *B, double *C, int lda){
    __m128d AA, BB, CC;
    CC = _load128(C);
    for(int i = 0; i < lda; i++){
        AA = _load128(A + i * 2);
        BB = _mm_set1_pd(B[i]);
        CC = _mm_fmadd_pd(AA, BB, CC);
    }
    _store128(C, CC);
}

inline void multiply_2x4(double *A, double *B, double *C, int lda){
    __m128d BB1, BB2, BB3, BB4;
    __m128d CC1, CC2, CC3, CC4;
    CC1 = _load128(C + 0 * lda);
    CC2 = _load128(C + 1 * lda);
    CC3 = _load128(C + 2 * lda);
    CC4 = _load128(C + 3 * lda);
    for(int i = 0; i < lda; i++){
        __m128d AA = _load128(A + i * 2);
        BB1 = _mm_set1_pd(B[i + 0 * lda]);
        BB2 = _mm_set1_pd(B[i + 1 * lda]);
        BB3 = _mm_set1_pd(B[i + 2 * lda]);
        BB4 = _mm_set1_pd(B[i + 3 * lda]);

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

inline void multiply_4x1(double *A, double *B, double *C, int lda){
    encode_t AA, BB, CC;
    CC = _load(C);
    for(int i = 0; i < lda; i++){
        AA = _load(A + i * 4);
        BB = _mm256_broadcast_sd(B + i);
        CC = _mm256_fmadd_pd(AA, BB, CC);
    }
    _store(C, CC);
}

inline void multiply_4x4(double *A, double *B, double *C, int lda){
    encode_t BB1, BB2, BB3, BB4;
    encode_t CC1, CC2, CC3, CC4;
    CC1 = _load(C + 0 * lda);
    CC2 = _load(C + 1 * lda);
    CC3 = _load(C + 2 * lda);
    CC4 = _load(C + 3 * lda);
    for(int i = 0; i < lda; i++){
        encode_t AA = _load(A + i * 4);
        BB1 = _mm256_broadcast_sd(B + i + 0 * lda);
        BB2 = _mm256_broadcast_sd(B + i + 1 * lda);
        BB3 = _mm256_broadcast_sd(B + i + 2 * lda);
        BB4 = _mm256_broadcast_sd(B + i + 3 * lda);

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

inline void multiply_8x4(double *A, double *B, double *C, int lda){
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
        AA1 = _load(A + i * 8);
        AA2 = _load(A + 4 + i * 8);
        
        BB1 = _mm256_broadcast_sd(B + i + 0 * lda);
        BB2 = _mm256_broadcast_sd(B + i + 1 * lda);
        BB3 = _mm256_broadcast_sd(B + i + 2 * lda);
        BB4 = _mm256_broadcast_sd(B + i + 3 * lda);

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

inline void multiply_8x1(double *A, double *B, double *C, int lda){
    encode_t AA1, AA2, BB, CC1, CC2;    
    CC1 = _load(C);
    CC2 = _load(C + 4);
    for(int i = 0; i < lda; i++){
        AA1 = _load(A + i * 8);
        AA2 = _load(A + 4 + i * 8);
        BB = _mm256_set1_pd(B[i]);
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

// small matrix not do pack------------------

inline void _multiply_1x1(double *A, double *B, double *C, int lda){
    double tmp = C[0];
    for(int i = 0; i < lda; i++){
        tmp += A[i * lda] * B[i];
    }
    C[0] = tmp;
}

inline void _multiply_1x4(double *A, double *B, double *C, int lda){
    for(int i = 0; i < lda; i++){
        double tmp = A[i*lda];
        C[0 * lda] += tmp * B[i + 0 * lda];
        C[1 * lda] += tmp * B[i + 1 * lda];
        C[2 * lda] += tmp * B[i + 2 * lda];
        C[3 * lda] += tmp * B[i + 3 * lda];
    }
}

inline void _multiply_2x1(double *A, double *B, double *C, int lda){
    __m128d AA, BB, CC;
    CC = _load128(C);
    for(int i = 0; i < lda; i++){
        AA = _load128(A + i * lda);
        BB = _mm_set1_pd(B[i]);
        CC = _mm_fmadd_pd(AA, BB, CC);
    }
    _store128(C, CC);
}

inline void _multiply_2x4(double *A, double *B, double *C, int lda){
    __m128d BB1, BB2, BB3, BB4;
    __m128d CC1, CC2, CC3, CC4;
    CC1 = _load128(C + 0 * lda);
    CC2 = _load128(C + 1 * lda);
    CC3 = _load128(C + 2 * lda);
    CC4 = _load128(C + 3 * lda);
    for(int i = 0; i < lda; i++){
        __m128d AA = _load128(A + i * lda);
        BB1 = _mm_set1_pd(B[i + 0 * lda]);
        BB2 = _mm_set1_pd(B[i + 1 * lda]);
        BB3 = _mm_set1_pd(B[i + 2 * lda]);
        BB4 = _mm_set1_pd(B[i + 3 * lda]);

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

inline void _multiply_4x1(double *A, double *B, double *C, int lda){
    encode_t AA, BB, CC;
    CC = _load(C);
    for(int i = 0; i < lda; i++){
        AA = _load(A + i * lda);
        BB = _mm256_broadcast_sd(B + i);
        CC = _mm256_fmadd_pd(AA, BB, CC);
    }
    _store(C, CC);
}

inline void _multiply_4x4(double *A, double *B, double *C, int lda){
    encode_t BB1, BB2, BB3, BB4;
    encode_t CC1, CC2, CC3, CC4;
    CC1 = _load(C + 0 * lda);
    CC2 = _load(C + 1 * lda);
    CC3 = _load(C + 2 * lda);
    CC4 = _load(C + 3 * lda);
    for(int i = 0; i < lda; i++){
        encode_t AA = _load(A + i * lda);
        BB1 = _mm256_broadcast_sd(B + i + 0 * lda);
        BB2 = _mm256_broadcast_sd(B + i + 1 * lda);
        BB3 = _mm256_broadcast_sd(B + i + 2 * lda);
        BB4 = _mm256_broadcast_sd(B + i + 3 * lda);

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

inline void _multiply_8x4(double *A, double *B, double *C, int lda){
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
        AA1 = _load(A + i * lda);
        AA2 = _load(A + 4 + i * lda);
        
        BB1 = _mm256_broadcast_sd(B + i + 0 * lda);
        BB2 = _mm256_broadcast_sd(B + i + 1 * lda);
        BB3 = _mm256_broadcast_sd(B + i + 2 * lda);
        BB4 = _mm256_broadcast_sd(B + i + 3 * lda);

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

inline void _multiply_8x1(double *A, double *B, double *C, int lda){
    encode_t AA1, AA2, BB, CC1, CC2;    
    CC1 = _load(C);
    CC2 = _load(C + 4);
    for(int i = 0; i < lda; i++){
        AA1 = _load(A + i * lda);
        AA2 = _load(A + 4 + i * lda);
        BB = _mm256_set1_pd(B[i]);
        CC1 = _mm256_fmadd_pd(AA1, BB, CC1);
        CC2 = _mm256_fmadd_pd(AA2, BB, CC2);
    }
    _store(C, CC1);
    _store(C + 4, CC2);
}


void do_small_dgemm(int lda, double *A, double *B, double *C){
    //8 * 4
    for(int i = 0; i < lda / 8 * 8; i += 8){
        for(int j = 0; j < lda / 4 * 4; j += 4){
            _multiply_8x4(A + i, B + j * lda, C + i + j * lda, lda);
        }
        for(int j = lda / 4 * 4; j < lda; j++){
            _multiply_8x1(A + i, B + j * lda, C + i + j * lda, lda);
        }
    }

    // 4* 4
    for(int i = lda / 8 * 8; i < lda / 4 * 4; i += 4){
        for(int j = 0; j < lda / 4 * 4; j += 4){
            _multiply_4x4(A + i, B + j * lda, C + i + j * lda, lda);
        }
        for(int j = lda / 4 * 4; j < lda; j++){
            _multiply_4x1(A + i, B + j * lda, C + i + j * lda, lda);
        }
    }

    // 2 * 4
    for(int i = lda / 4 * 4; i < lda / 2 * 2; i += 2){
        for(int j = 0; j < lda / 4 * 4; j += 4){
            _multiply_2x4(A + i, B + j * lda, C + i + j * lda, lda);
        }
        for(int j = lda / 4 * 4; j < lda; j++){
            _multiply_2x1(A + i, B + j * lda, C + i + j * lda, lda);
        }
    }

    //1 * 4
    for(int i = lda / 2 * 2; i < lda; i++){
        for(int j = 0; j < lda / 4 * 4; j += 4){
            _multiply_1x4(A + i, B + j * lda, C + i + j * lda, lda);
        }
        for(int j = lda / 4 * 4; j < lda; j++){
            _multiply_1x1(A + i, B + j * lda, C + i + j * lda, lda);
        }
    }
}

//divied into 4 * 4 * 4, 4 * 4 * 1, 1 * 4 * 4, 1 * 1 * 1
void square_dgemm(int lda, double *A, double *B, double *C)
{
    if(lda < 64){
        do_small_dgemm(lda, A, B, C);
        return;
    }
    // 8 * 4
    for(int i = 0; i < lda / 8 * 8; i += 8){
        pack_A(A + i, _A, 8, lda);
        for(int j = 0; j < lda / 4 * 4; j += 4){
            multiply_8x4(_A, B + j * lda, C + i + j * lda, lda);
        }
        for(int j = lda / 4 * 4; j < lda; j++){
            multiply_8x1(_A, B + j * lda, C + i + j * lda, lda);
        }
    }

    // 4* 4
    for(int i = lda / 8 * 8; i < lda / 4 * 4; i += 4){
        pack_A(A + i, _A, 4, lda);
        for(int j = 0; j < lda / 4 * 4; j += 4){
            multiply_4x4(_A, B + j * lda, C + i + j * lda, lda);
        }
        for(int j = lda / 4 * 4; j < lda; j++){
            multiply_4x1(_A, B + j * lda, C + i + j * lda, lda);
        }
    }

    // 2 * 4
    for(int i = lda / 4 * 4; i < lda / 2 * 2; i += 2){
        pack_A(A + i, _A, 2, lda);
        for(int j = 0; j < lda / 4 * 4; j += 4){
            multiply_2x4(_A, B + j * lda, C + i + j * lda, lda);
        }
        for(int j = lda / 4 * 4; j < lda; j++){
            multiply_2x1(_A, B + j * lda, C + i + j * lda, lda);
        }
    }

    //1 * 4
    for(int i = lda / 2 * 2; i < lda; i++){
        pack_A(A + i, _A, 1, lda);
        for(int j = 0; j < lda / 4 * 4; j += 4){
            multiply_1x4(_A, B + j * lda, C + i + j * lda, lda);
        }
        for(int j = lda / 4 * 4; j < lda; j++){
            multiply_1x1(_A, B + j * lda, C + i + j * lda, lda);
        }
    }
}
