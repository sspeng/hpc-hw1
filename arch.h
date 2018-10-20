//
// by msn
//
#ifndef ARCH_H
#define ARCH_H
#include <stdlib.h>
#include <stdint.h>
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

encode_t _load(double *addr){
    if((unsigned long long )addr % 32 == 0) return _mm256_load_pd(addr);
    else return _mm256_loadu_pd(addr);
}

void _store(double *addr, encode_t a){
    if((unsigned long long )addr % 32 == 0) _mm256_store_pd(addr, a);
    else _mm256_storeu_pd(addr, a);
}

__m128d _load128(double *addr){
    return _mm_loadu_pd(addr);
}

void _store128(double *addr, __m128d a){
    _mm_storeu_pd(addr, a);
}

#endif //immintrin.h
#endif //ARCH_H