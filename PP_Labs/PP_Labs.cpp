/* Лабораторная работа 1 */

//#include "mpi.h"
//#include "stdio.h"
//#include <time.h>
//#include <math.h>
//#include <omp.h>
//
//int main(int argc, char* argv[])
//{
//    int rank, ranksize, i;
//    double start_time, end_time;
//
//    MPI_Init(&argc, &argv);
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
//
//    start_time = MPI_Wtime(); // Засекаем начальное время
//
//    printf("Hello world from process %d out of %d\n", rank + 1, ranksize);
//
//    MPI_Barrier(MPI_COMM_WORLD); // Ждем, пока все процессы дойдут до этой точки
//
//    end_time = MPI_Wtime(); // Засекаем конечное время
//
//    // Вычисляем и выводим время выполнения в микросекундах
//    if (rank == 0) {
//        printf("Total execution time: %f microseconds\n", (end_time - start_time) * 1e6);
//    }
//
//    MPI_Finalize();
//    return 0;
//}

//#include <math.h>
//#include <omp.h>
//#include <time.h>
//#include <stdlib.h>
//#include <locale.h>
//#include <stdio.h>
//
//int main(int argc, char* argv[])
//{
//    omp_set_num_threads(6);
//    int nThreads, threadNum;
//    double start_time, end_time;
//
//    // Засекаем начальное время в микросекундах
//    start_time = omp_get_wtime();
//
//#pragma omp parallel private(nThreads, threadNum)
//    {
//        nThreads = omp_get_num_threads();
//        threadNum = omp_get_thread_num();
//        printf("OpenMP thread %d out of %d threads\n", threadNum + 1, nThreads);
//    }
//
//    // Засекаем конечное время в микросекундах
//    end_time = omp_get_wtime();
//
//    // Вычисляем и выводим общее время выполнения в микросекундах
//    printf("Total execution time: %f microseconds\n", (end_time - start_time) * 1e6);
//
//    return 0;
//}

//#include <stdio.h>
//#include <chrono>
//
//int main() {
//    int nThreads = 3;
//
//    // Засекаем начальное время
//    auto start_time = std::chrono::high_resolution_clock::now();
//
//    // Эмулируем работу с тремя процессами
//    for (int threadNum = 0; threadNum < nThreads; ++threadNum) {
//        printf("Iteration %d out of %d threads\n", threadNum + 1, nThreads);
//    }
//
//    // Засекаем конечное время
//    auto end_time = std::chrono::high_resolution_clock::now();
//
//    // Вычисляем и выводим общее время выполнения в микросекундах
//    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
//    printf("Total execution time: %lld microseconds\n", duration);
//
//    return 0;
//}

/* Лабораторная работа 2 */

// OpenMP
//#include <omp.h>
//#define CHUNK 100
//#define NMAX 7600000
//#include "stdio.h"
//
//int main(int argc, char* argv[]) {
//	int num_proccesses[3] = {2, 4, 8};
//
//	for (int k = 0; k < 3; ++k) {
//		printf("\nNum proccesses = %d", num_proccesses[k]);
//		omp_set_num_threads(num_proccesses[k]);
//		int i, j;
//		int* a = new int[NMAX];
//		double sum;
//		int Q = 22;
//		for (i = 0; i < NMAX; i++) {
//			a[i] = 1.0;
//		}
//		double st_time, end_time;
//		st_time = omp_get_wtime();
//		sum = 0;
//
//#pragma omp parallel for shared(a) private(i, j) reduction(+:sum)
//		for (i = 0; i < NMAX; i++) {
//			for (j = 0; j < Q; j++) {
//				a[i] = (a[i] + a[i]) / 2;
//			}
////#pragma omp atomic
////#pragma omp critical
//			sum += a[i];
//		}
//		end_time = omp_get_wtime();
//		end_time = end_time - st_time;
//		printf("\nQ = %d", Q);
//		printf("\nTotal Sum = %10.2f", sum);
//		printf("\nTIME OF WORK IS %f ", end_time);
//		delete[] a;
//	}
//
//	return 0;
//}

// MPI
//#include <math.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include "mpi.h"
//
//int main(int argc, char* argv[])
//{
//	int N = 7600000;
//	int* x = new int[N];
//	*x = 0;
//	double TotalSum, ProcSum = 0.0;
//	int ProcRank, ProcNum, j, i;
//	MPI_Status Status;
//	double st_time, end_time;
//	int Q = 22;
//
//	MPI_Init(&argc, &argv);
//	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
//	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
//
//	if (ProcRank == 0) {
//		delete[] x;
//		x = new int[N];
//		for (i = 0; i < N; i++) {
//			x[i] = 1.0;
//		}
//	}
//	int* x_loc = new int[N / ProcNum];
//	MPI_Scatter(x, N / ProcNum, MPI_FLOAT, x_loc, N / ProcNum, MPI_FLOAT, 0, MPI_COMM_WORLD);
//	st_time = MPI_Wtime();
//
//
//	for (i = 0; i < N / ProcNum; i++) {
//		for (j = 0; j < Q; j++) {
//			x_loc[i] = (x_loc[i] + x_loc[i]) / 2;
//		}
//		ProcSum += x_loc[i];
//	}
//
//	TotalSum = 0;
//	MPI_Reduce(&ProcSum, &TotalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//	if (ProcRank == 0)
//	{
//		TotalSum = ProcSum;
//		for (i = 1; i < ProcNum; i++)
//		{
//			MPI_Recv(&ProcSum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &Status);
//			TotalSum = TotalSum + ProcSum;
//		}
//	}
//	else
//		MPI_Send(&ProcSum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
//
//	MPI_Barrier(MPI_COMM_WORLD);
//
//	end_time = MPI_Wtime();
//	end_time = end_time - st_time;
//
//	if (ProcRank == 0)
//	{
//		printf("\nQ = %d", Q);
//		printf("\nTotal Sum = %10.2f", TotalSum);
//		printf("\nTIME OF WORK IS %f ", end_time);
//	}
//
//	MPI_Finalize();
//	return 0;
//}

// Sequential
#include "stdio.h"
#include <time.h>

int main(int argc, char* argv[]) {
	int Q = 22;
	int i, j, N = 7600000;
	int* x = new int[N];
	double sum = 0;
	for (int i = 0; i < N; ++i) {
		x[i] = 1.0;
	}
	double start = clock();
	for (i = 0; i < N; ++i) {
		for (j = 0; j < Q; ++j) {
			sum += x[i];
		}
	}
	//sum /= Q;
	double end = clock();
	double t = (end - start) / CLOCKS_PER_SEC;
	printf("Sum: %f\n", sum);
	printf("Time: %f", t);
}
