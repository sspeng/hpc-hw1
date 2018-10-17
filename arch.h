//
// by msn
//
#ifndef ARCH_H
#define ARCH_H

#define AVX2

#ifdef AVX2
#include <immintrin.h>
typedef  __m256d encode_t;
encode_t multiply_256(encode_t a, double b){
    encode_t c = _mm256_set1_pd(b);
    return _mm256_mul_pd(a, c);
}

encode_t add_256(encode_t a, encode_t b){
    return _mm256_add_pd(a, b);
}

void store(double *addr, encode_t a){
    _mm256_stream_pd(addr, a);
}

encode_t zero(){
    return _mm256_setzero_pd();
}

#endif //immintrin.h
#endif //ARCH_H