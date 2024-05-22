/*
 * Tema 2 ASC
 * 2024 Spring
 */
#include "utils.h"
#include <string.h>

/*
 * Add your unoptimized implementation here
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	
    unsigned int matrix_dimension = N * N;

    // Calculate A transpose
    double *transpose_A = calloc(matrix_dimension, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            transpose_A[j * N + i] = A[i * N + j];
        }
    }

    // Calculate B transpose
    double *transpose_B = calloc(matrix_dimension, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            transpose_B[j * N + i] = B[i * N + j];
        }
    }

    // Calculate A^T x B
    double *AtB = calloc(matrix_dimension, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k <= i; k++) {
                AtB[i * N + j] += transpose_A[i * N + k] * B[k * N + j];
            }
        }
    }

    // Calculate B x A
    double *BA = calloc(matrix_dimension, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k <= j; k++) {
                BA[i * N + j] += B[i * N + k] * A[k * N + j];
            }
        }
    }

    // Calculate (A^T x B) + (B x A)
    double *sum = calloc(matrix_dimension, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sum[i * N + j] = AtB[i * N + j] + BA[i * N + j];
        }
    }

    // Calculate ((A^T x B) + (B x A)) x B^T
    double *result = calloc(matrix_dimension, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                result[i * N + j] += sum[i * N + k] * transpose_B[k * N + j];
            }
        }
    }

    // Free the intermediate matrices
    free(transpose_A);
    free(transpose_B);
    free(AtB);
    free(BA);
    free(sum);

    return result;
}
