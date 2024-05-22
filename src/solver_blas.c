/*
 * Tema 2 ASC
 * 2024 Spring
 */
#include "utils.h"
#include <cblas.h>

/* 
 * Add your BLAS implementation here
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

    unsigned int matrix_dimension = N * N;

    // Calculate A^T x B
    double *AtB = calloc(matrix_dimension, sizeof(double));
    cblas_dcopy(matrix_dimension, B, 1, AtB, 1); // Copy B into AtB
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, AtB, N);

    // Calculate B x A
    double *BA = calloc(matrix_dimension, sizeof(double));
    cblas_dcopy(matrix_dimension, B, 1, BA, 1); // Copy B into BA
    cblas_dtrmm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, BA, N);

    // Calculate (A^T x B) + (B x A)
    double *sum = calloc(matrix_dimension, sizeof(double));
    cblas_dcopy(matrix_dimension, AtB, 1, sum, 1);
    cblas_daxpy(matrix_dimension, 1, BA, 1, sum, 1);

    // Calculate ((A^T x B) + (B x A)) x B^T
    double *result = calloc(matrix_dimension, sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, sum, N, B, N, 0, result, N);

    // Free the intermediate matrices
    free(AtB);
    free(BA);
    free(sum);

    return result;
}
