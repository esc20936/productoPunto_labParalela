//--------------------------------------------------------------
// prodAx.c
//--------------------------------------------------------------
// Calcula el producto de una matriz m x n por un vector tamaño n
//--------------------------------------------------------------
// Auth.  JJCelada - Universidad del Valle de Guatemala
// Date   2021-10-06
// Vers.  1.0

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void Ax_b(int m, int n, double *restrict A, double *restrict x,
          double *restrict b);

int main(int argc, char *argv[])
{
    // start clock

    double *A, *x, *b;
    int i, j, m, n;

    printf("Ingrese las dimensiones m y n de la matriz: ");
    scanf("%d %d", &m, &n);

    clock_t start = clock();

    //---- Asignación de memoria para la matriz A ----
    if ((A = (double *)malloc(m * n * sizeof(double))) == NULL)
        perror("memory allocation for A");

    //---- Asignación de memoria para el vector x ----
    if ((x = (double *)malloc(n * sizeof(double))) == NULL)
        perror("memory allocation for x");

    //---- Asignación de memoria para el vector b ----
    if ((b = (double *)malloc(m * sizeof(double))) == NULL)
        perror("memory allocation for b");

    printf("Initializing matrix A and vector x\n");

    //---- Inicialización con elementos aleatorios entre 1-7 y 1-13
    for (j = 0; j < n; j++)
        x[j] = rand() % 7 + 1;

    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            A[i * n + j] = rand() % 13 + 1;

    printf("Calculando el producto Ax para m = %d n = %d\n", m, n);
    (void)Ax_b(m, n, A, x, b);

    // printf("\nb: \n");
    // for (j = 0; j < n; j++)
    //     printf("\t%0.0f ", b[j]);
    // printf("\n\n");

    free(A);
    free(x);
    free(b);

    // end clock
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time spent: %f\n", time_spent);

    return (0);
}

/* ------------------------
 * prodAx
 * ------------------------
 */
void Ax_b(int m, int n, double *A, double *x, double *b)
{
    int i, j;
#pragma omp parallel for
    for (i = 0; i < m; i++)
    {
        b[i] = 0.0; // inicialización elemento i del vec.
        for (j = 0; j < n; j++)
        {
            b[i] += A[i * n + j] * x[j]; // producto punto
        }
    } /*−−-Fin de parallel for−−−*/
}