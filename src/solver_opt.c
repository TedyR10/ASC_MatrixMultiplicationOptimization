/*
 * Tema 2 ASC
 * 2024 Spring
 */
#include "utils.h"
#include <string.h>
#include <stdlib.h>

#define BLOCK_SIZE 100
#define MATRIX_SIZE (N * N)

/*
 * Add your optimized implementation here
 */
double* my_solver(int N, double *A, double *B) {
	printf("OPT SOLVER\n");

    // Allocate memory for intermediate and result matrices
    double *transpose_A = calloc(MATRIX_SIZE, sizeof(double));
    double *transpose_B = calloc(MATRIX_SIZE, sizeof(double));
    double *AtB = calloc(MATRIX_SIZE, sizeof(double));
    double *BA = calloc(MATRIX_SIZE, sizeof(double));
    double *sum = calloc(MATRIX_SIZE, sizeof(double));
    double *result = calloc(MATRIX_SIZE, sizeof(double));

    // Calculate A transpose
    for (register int i = 0; i < N; i++) {
        for (register int j = i; j < N; j++) {
            transpose_A[j * N + i] = A[i * N + j];
        }
    }

    // Calculate B transpose
    for (register int i = 0; i < N; i++) {
        for (register int j = 0; j < N; j++) {
            transpose_B[j * N + i] = B[i * N + j];
        }
    }

    // Blocked matrix multiplication for A^T * B
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
		for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
			for (register int bk = 0; bk <= bi; bk += BLOCK_SIZE) {
				for (register int i = bi; i < bi + BLOCK_SIZE && i < N; i++) {
					double *restrict orig_pa = transpose_A + i * N + bk;
					for (register int j = bj; j < bj + BLOCK_SIZE && j < N; j++) {
						register double *restrict pa = orig_pa;
						register double *restrict pb = B + bk * N + j;
						register double sum = 0;
						for (register int k = bk; k < bk + BLOCK_SIZE && k <= i; k++) {
							sum += *pa * *pb;
							pa++;
							pb += N;
						}
						AtB[i * N + j] += sum;
					}
				}
			}
		}
	}

    // Blocked matrix multiplication for B * A
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
		for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
			for (register int bk = 0; bk <= bj; bk += BLOCK_SIZE) {
				for (register int i = bi; i < bi + BLOCK_SIZE && i < N; i++) {
					double *restrict orig_pa = B + i * N + bk;
					for (register int j = bj; j < bj + BLOCK_SIZE && j < N; j++) {
						register double *restrict pa = orig_pa;
						register double *restrict pb = A + bk * N + j;
						register double sum = 0;
						for (register int k = bk; k < bk + BLOCK_SIZE && k <= j; k++) {
							sum += *pa * *pb;
							pa++;
							pb += N;
						}
						BA[i * N + j] += sum;
					}
				}
			}
		}
	}

    // Calculate (A^T * B) + (B * A)
    for (register int i = 0; i < N; i++) {
		for (register int j = 0; j < N; j++) {
			sum[i * N + j] = AtB[i * N + j] + BA[i * N + j];
		}
	}

	// Blocked matrix multiplication for ((A^T * B) + (B * A)) * B^T
	for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
		for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
			for (register int bk = 0; bk < N; bk += BLOCK_SIZE) {
				for (register int i = bi; i < bi + BLOCK_SIZE && i < N; i++) {
					double *restrict orig_pa = sum + i * N + bk;
					for (register int j = bj; j < bj + BLOCK_SIZE && j < N; j++) {
						register double *restrict pa = orig_pa;
						register double *restrict pb = transpose_B + bk * N + j;
						register double sum = 0;
						for (register int k = bk; k < bk + BLOCK_SIZE && k < N; k++) {
							sum += *pa * *pb;
							pa++;
							pb += N;
						}
						result[i * N + j] += sum;
					}
				}
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
