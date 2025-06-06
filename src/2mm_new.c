#include "2mm.h"

#include <stdio.h>

#include <stdlib.h>

#include <string.h>

#include <sys/time.h>



#define BS 64



double bench_t_start, bench_t_end;



static double rtclock()

{

    struct timeval Tp;

    int stat;

    stat = gettimeofday(&Tp, NULL);

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

                       double A[ni][nk],

                       double B[nk][nj],

                       double C[nj][nl],

                       double D[ni][nl])

{

    int i, j;

    *alpha = 1.5;

    *beta = 1.2;

    for (i = 0; i < ni; i++)

        for (j = 0; j < nk; j++)

            A[i][j] = (double)((i * j + 1) % ni) / ni;

    for (i = 0; i < nk; i++)

        for (j = 0; j < nj; j++)

            B[i][j] = (double)(i * (j + 1) % nj) / nj;

    for (i = 0; i < nj; i++)

        for (j = 0; j < nl; j++)

            C[i][j] = (double)((i * (j + 3) + 1) % nl) / nl;

    for (i = 0; i < ni; i++)

        for (j = 0; j < nl; j++)

            D[i][j] = (double)(i * (j + 2) % nk) / nk;

}



static void kernel_2mm(int ni, int nj, int nk, int nl,

                       double alpha,

                       double beta,

                       double tmp[ni][nj],

                       double A[ni][nk],

                       double B[nk][nj],

                       double C[nj][nl],

                       double D[ni][nl])

{

    int i, j, k;

    for (int i0 = 0; i0 < ni; i0 += BS) {

        int iMax = (i0 + BS < ni) ? (i0 + BS) : ni;

        for (int j0 = 0; j0 < nj; j0 += BS) {

            int jMax = (j0 + BS < nj) ? (j0 + BS) : nj;

            for (i = i0; i < iMax; i++) {

                for (j = j0; j < jMax; j++) {

                    tmp[i][j] = 0.0;

                }

            }

            for (int k0 = 0; k0 < nk; k0 += BS) {

                int kMax = (k0 + BS < nk) ? (k0 + BS) : nk;

                for (i = i0; i < iMax; i++) {

                    for (k = k0; k < kMax; k++) {

                        double aik = alpha * A[i][k];

                        for (j = j0; j < jMax; j++) {

                            tmp[i][j] += aik * B[k][j];

                        }

                    }

                }

            }

        }

    }

    for (i = 0; i < ni; i++) {

        for (j = 0; j < nl; j++) {

            D[i][j] *= beta;

        }

    }

    for (int i0 = 0; i0 < ni; i0 += BS) {

        int iMax = (i0 + BS < ni) ? (i0 + BS) : ni;

        for (int j0 = 0; j0 < nl; j0 += BS) {

            int jMax = (j0 + BS < nl) ? (j0 + BS) : nl;

            for (int k0 = 0; k0 < nj; k0 += BS) {

                int kMax = (k0 + BS < nj) ? (k0 + BS) : nj;

                for (i = i0; i < iMax; i++) {

                    for (k = k0; k < kMax; k++) {

                        double tik = tmp[i][k];

                        for (j = j0; j < jMax; j++) {

                            D[i][j] += tik * C[k][j];

                        }

                    }

                }

            }

        }

    }

}



void save_matrix_to_file(int ni, int nl, double D[ni][nl], const char *filename)

{

    FILE *f = fopen(filename, "w");

    if (!f) {

        printf("Error opening file: %s\n", filename);

        return;

    }

    for (int i = 0; i < ni; i++) {

        for (int j = 0; j < nl; j++) {

            fprintf(f, "%0.6lf", D[i][j]);

            if (j < nl - 1)

                fprintf(f, ",");

        }

        fprintf(f, "\n");

    }

    fclose(f);

}



int main(int argc, char **argv)

{

    int ni = NI;

    int nj = NJ;

    int nk = NK;

    int nl = NL;

    double alpha;

    double beta;

    double(*tmp)[ni][nj];

    tmp = (double(*)[ni][nj])malloc((ni) * (nj) * sizeof(double));

    double(*A)[ni][nk];

    A = (double(*)[ni][nk])malloc((ni) * (nk) * sizeof(double));

    double(*B)[nk][nj];

    B = (double(*)[nk][nj])malloc((nk) * (nj) * sizeof(double));

    double(*C)[nj][nl];

    C = (double(*)[nj][nl])malloc((nj) * (nl) * sizeof(double));

    double(*D)[ni][nl];

    D = (double(*)[ni][nl])malloc((ni) * (nl) * sizeof(double));

    init_array(ni, nj, nk, nl, &alpha, &beta, *A, *B, *C, *D);

    bench_timer_start();

    kernel_2mm(ni, nj, nk, nl, alpha, beta, *tmp, *A, *B, *C, *D);

    bench_timer_stop();

    bench_timer_print();



    free((void *)tmp);

    free((void *)A);

    free((void *)B);

    free((void *)C);

    free((void *)D);

    return 0;

}

