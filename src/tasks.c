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

        printf("Error return from gettimeofday: %d\n", stat);

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

    #pragma omp parallel

    {

        #pragma omp single

        {

            *alpha = 1.5;

            *beta = 1.2;



            for (int i0 = 0; i0 < ni; i0 += BS) {

                int iMax = (i0 + BS < ni) ? (i0 + BS) : ni;

                for (int j0 = 0; j0 < nk; j0 += BS) {

                    int jMax = (j0 + BS < nk) ? (j0 + BS) : nk;



                    #pragma omp task firstprivate(i0, j0, iMax, jMax)

                    {

                        for (int i = i0; i < iMax; i++) {

                            for (int j = j0; j < jMax; j++) {

                                A[i * nk + j] = (double)((i * j + 1) % ni) / ni;

                            }

                        }

                    }

                }

            }



            for (int i0 = 0; i0 < nk; i0 += BS) {

                int iMax = (i0 + BS < nk) ? (i0 + BS) : nk;

                for (int j0 = 0; j0 < nj; j0 += BS) {

                    int jMax = (j0 + BS < nj) ? (j0 + BS) : nj;



                    #pragma omp task firstprivate(i0, j0, iMax, jMax)

                    {

                        for (int i = i0; i < iMax; i++) {

                            for (int j = j0; j < jMax; j++) {

                                B[i * nj + j] = (double)(i * (j + 1) % nj) / nj;

                            }

                        }

                    }

                }

            }



            for (int i0 = 0; i0 < nj; i0 += BS) {

                int iMax = (i0 + BS < nj) ? (i0 + BS) : nj;

                for (int j0 = 0; j0 < nl; j0 += BS) {

                    int jMax = (j0 + BS < nl) ? (j0 + BS) : nl;



                    #pragma omp task firstprivate(i0, j0, iMax, jMax)

                    {

                        for (int i = i0; i < iMax; i++) {

                            for (int j = j0; j < jMax; j++) {

                                C[i * nl + j] = (double)((i * (j + 3) + 1) % nl) / nl;

                            }

                        }

                    }

                }

            }



            for (int i0 = 0; i0 < ni; i0 += BS) {

                int iMax = (i0 + BS < ni) ? (i0 + BS) : ni;

                for (int j0 = 0; j0 < nl; j0 += BS) {

                    int jMax = (j0 + BS < nl) ? (j0 + BS) : nl;



                    #pragma omp task firstprivate(i0, j0, iMax, jMax)

                    {

                        for (int i = i0; i < iMax; i++) {

                            for (int j = j0; j < jMax; j++) {

                                D[i * nl + j] = (double)((i * (j + 2)) % nk) / nk;

                            }

                        }

                    }

                }

            }

        }

        #pragma omp taskwait

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

    #pragma omp parallel

    {

        #pragma omp single

        {

            for (int i0 = 0; i0 < ni; i0 += BS) {

                int iMax = (i0 + BS < ni) ? (i0 + BS) : ni;

                for (int j0 = 0; j0 < nj; j0 += BS) {

                    int jMax = (j0 + BS < nj) ? (j0 + BS) : nj;



                    #pragma omp task firstprivate(i0, j0, iMax, jMax)

                    {

                        for (int i = i0; i < iMax; i++) {

                            for (int j = j0; j < jMax; j++) {

                                tmp[i * nj + j] = 0.0;

                            }

                        }

                        for (int k = 0; k < nk; k++) {

                            for (int i = i0; i < iMax; i++) {

                                double aik = alpha * A[i * nk + k];

                                for (int j = j0; j < jMax; j++) {

                                    tmp[i * nj + j] += aik * B[k * nj + j];

                                }

                            }

                        }

                    }

                }

            }

        }

        #pragma omp taskwait

    }



    #pragma omp parallel

    {

        #pragma omp single

        {

            for (int i0 = 0; i0 < ni; i0 += BS) {

                int iMax = (i0 + BS < ni) ? (i0 + BS) : ni;

                for (int j0 = 0; j0 < nl; j0 += BS) {

                    int jMax = (j0 + BS < nl) ? (j0 + BS) : nl;



                    #pragma omp task firstprivate(i0, j0, iMax, jMax)

                    {

                        for (int i = i0; i < iMax; i++) {

                            for (int j = j0; j < jMax; j++) {

                                D[i * nl + j] *= beta;

                            }

                        }

                    }

                }

            }

        }

        #pragma omp taskwait

    }



    #pragma omp parallel

    {

        #pragma omp single

        {

            for (int i0 = 0; i0 < ni; i0 += BS) {

                int iMax = (i0 + BS < ni) ? (i0 + BS) : ni;

                for (int j0 = 0; j0 < nl; j0 += BS) {

                    int jMax = (j0 + BS < nl) ? (j0 + BS) : nl;



                    #pragma omp task firstprivate(i0, j0, iMax, jMax)

                    {

                        for (int k = 0; k < nj; k++) {

                            for (int i = i0; i < iMax; i++) {

                                double tik = tmp[i * nj + k];

                                for (int j = j0; j < jMax; j++) {

                                    D[i * nl + j] += tik * C[k * nl + j];

                                }

                            }

                        }

                    }

                }

            }

        }

        #pragma omp taskwait

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

