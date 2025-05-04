#include "2mm.h"

#include <omp.h>

#include <stdio.h>

#include <stdlib.h>

#include <string.h>

#include <sys/time.h>



#ifndef BS

#define BS 512

#endif



double bench_t_start, bench_t_end;



static double rtclock()

{

    struct timeval Tp;

    int stat = gettimeofday(&Tp, NULL);

    if (stat != 0)

        printf("Error return from gettimeofday: %d", stat);

    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);

}



void bench_timer_start()

{

    bench_t_start = rtclock();

}



void bench_timer_stop()

{

    bench_t_end = rtclock();

}



void bench_timer_print()

{

    printf("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);

}



static void init_array(int ni, int nj, int nk, int nl,

                       double *alpha,

                       double *beta,

                       double *A,

                       double *B,

                       double *C,

                       double *D)

{

    int i, j;

    *alpha = 1.5;

    *beta = 1.2;



    #pragma omp parallel

    {

        #pragma omp for nowait

        for (i = 0; i < ni; i++)

            for (j = 0; j < nk; j++)

                A[i * nk + j] = (double)((i * j + 1) % ni) / ni;



        #pragma omp for nowait

        for (i = 0; i < nk; i++)

            for (j = 0; j < nj; j++)

                B[i * nj + j] = (double)(i * (j + 1) % nj) / nj;



        #pragma omp for nowait

        for (i = 0; i < nj; i++)

            for (j = 0; j < nl; j++)

                C[i * nl + j] = (double)((i * (j + 3) + 1) % nl) / nl;



        #pragma omp for nowait

        for (i = 0; i < ni; i++)

            for (j = 0; j < nl; j++)

                D[i * nl + j] = (double)(i * (j + 2) % nk) / nk;

    }

}



static void kernel_2mm(int ni, int nj, int nk, int nl,

                       double alpha,

                       double beta,

                       double *tmp,

                       double *A,

                       double *B,

                       double *C,

                       double *D)

{

    int i, j, k;



    #pragma omp parallel

    {

        #pragma omp for private(i, j, k) collapse(2)

        for (int i0 = 0; i0 < ni; i0 += BS) {

            for (int j0 = 0; j0 < nj; j0 += BS) {



                int iMax = (i0 + BS < ni) ? (i0 + BS) : ni;

                int jMax = (j0 + BS < nj) ? (j0 + BS) : nj;



                for (i = i0; i < iMax; i++) {

                    for (j = j0; j < jMax; j++) {

                        tmp[i * nj + j] = 0.0;

                    }

                }



                for (int k0 = 0; k0 < nk; k0 += BS) {

                    int kMax = (k0 + BS < nk) ? (k0 + BS) : nk;



                    for (i = i0; i < iMax; i++) {

                        for (k = k0; k < kMax; k++) {

                            double aik = alpha * A[i * nk + k];

                            for (j = j0; j < jMax; j++) {

                                tmp[i * nj + j] += aik * B[k * nj + j];

                            }

                        }

                    }

                }

            }

        }

    }



    #pragma omp parallel for private(i, j)

    for (i = 0; i < ni; i++) {

        for (j = 0; j < nl; j++) {

            D[i * nl + j] *= beta;

        }

    }



    #pragma omp parallel

    {

        #pragma omp for private(i, j, k) collapse(2)

        for (int i0 = 0; i0 < ni; i0 += BS) {

            for (int j0 = 0; j0 < nl; j0 += BS) {



                int iMax = (i0 + BS < ni) ? (i0 + BS) : ni;

                int jMax = (j0 + BS < nl) ? (j0 + BS) : nl;



                for (int k0 = 0; k0 < nj; k0 += BS) {

                    int kMax = (k0 + BS < nj) ? (k0 + BS) : nj;



                    for (i = i0; i < iMax; i++) {

                        for (k = k0; k < kMax; k++) {

                            double tik = tmp[i * nj + k];

                            for (j = j0; j < jMax; j++) {

                                D[i * nl + j] += tik * C[k * nl + j];

                            }

                        }

                    }

                }

            }

        }

    }

}



int main(int argc, char** argv)

{

    int ni = NI;

    int nj = NJ;

    int nk = NK;

    int nl = NL;



    double alpha, beta;



    double *tmp = (double*) malloc(ni * nj * sizeof(double));

    double *A = (double*) malloc(ni * nk * sizeof(double));

    double *B = (double*) malloc(nk * nj * sizeof(double));

    double *C = (double*) malloc(nj * nl * sizeof(double));

    double *D = (double*) malloc(ni * nl * sizeof(double));



    init_array(ni, nj, nk, nl, &alpha, &beta, A, B, C, D);



    bench_timer_start();



    kernel_2mm(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D);



    bench_timer_stop();

    bench_timer_print();



    free(tmp);

    free(A);

    free(B);

    free(C);

    free(D);



    return 0;

}

