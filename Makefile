# We will benchmark you against Intel MKL implementation, the default processor vendor-tuned implementation.
# This makefile is intended for the Intel C compiler.
# Your code must compile (with icc) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = icc 
OPT = -Ofast  -march=native
CFLAGS = -Wall -DGETTIMEOFDAY -std=c99 $(OPT)
LDFLAGS = -Wall
# mkl is needed for blas implementation
LDLIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

targets = benchmark-naive benchmark-blocked benchmark-blocked-1 benchmark-blocked-2 benchmark-blocked-3 benchmark-blocked-4 benchmark-blocked-5 benchmark-blocked-6 benchmark-blocked-7 benchmark-blas 
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blocked-1.o dgemm-blocked-2.o dgemm-blocked-3.o dgemm-blocked-4.o dgemm-blocked-5.o dgemm-blocked-6.o dgemm-blocked-7.o dgemm-blas.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-1 : benchmark.o dgemm-blocked-1.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-2 : benchmark.o dgemm-blocked-2.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-3 : benchmark.o dgemm-blocked-3.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-4 : benchmark.o dgemm-blocked-4.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-5 : benchmark.o dgemm-blocked-5.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-6 : benchmark.o dgemm-blocked-6.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-7 : benchmark.o dgemm-blocked-7.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
