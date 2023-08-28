//--------------------------------------------------------------
// prodAx_optimizado.c
//--------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void Ax_b(int m, int n, double *A, double *x, double *b);

int main(int argc, char *argv[])
{
    // Inicializa la semilla aleatoria para los números pseudoaleatorios
    srand(time(NULL));

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

    // Liberar memoria
    free(A);
    free(x);
    free(b);

    // Medir el tiempo transcurrido
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
#pragma omp parallel for shared(m, n, A, x, b) private(i, j) schedule(guided, 10)
    for (i = 0; i < m; i++)
    {
        b[i] = 0.0; // inicialización elemento i del vec.
        for (j = 0; j < n; j += 4)
        {
            b[i] += A[i * n + j] * x[j];         // producto punto
            b[i] += A[i * n + j + 1] * x[j + 1];
            b[i] += A[i * n + j + 2] * x[j + 2];
            b[i] += A[i * n + j + 3] * x[j + 3];
        }
    } /*−−-Fin de parallel for−−−*/
}
