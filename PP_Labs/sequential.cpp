#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 950000

void addArrays(int* c, const int* a, const int* b, unsigned int size) {
	for (unsigned int i = 0; i < size; ++i) {
		c[i] = a[i] + b[i];
	}
}

int main() {
	int n = N;
	printf("n = %u\n", n);

	int* a = (int*)malloc(n * sizeof(int));
	int* b = (int*)malloc(n * sizeof(int));
	int* c = (int*)malloc(n * sizeof(int));

	// Инициализация массивов
	for (unsigned int i = 0; i < n; ++i) {
		a[i] = 1;
		b[i] = 1;
	}

	// Замер времени выполнения
	clock_t start_time = clock();

	// Запуск ядра
	for (int i = 0; i < 12; ++i) {
		addArrays(c, a, b, n);
	}

	clock_t end_time = clock();
	double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

	printf("Time spent executing kernel: %.9f seconds\n", elapsed_time);

	// Очищение памяти
	free(a);
	free(b);
	free(c);

	return 0;
}