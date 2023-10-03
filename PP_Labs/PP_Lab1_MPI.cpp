//#include "mpi.h"
//#include "stdio.h"
//#include <time.h>
//#include <math.h>
//#include <omp.h>
//
//int main(int argc, char* argv[])
//{
//	int rank, ranksize, i;
//	double start_time, end_time;
//
//	MPI_Init(&argc, &argv);
//	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//	MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
//
//	start_time = MPI_Wtime(); // Засекаем начальное время
//
//	printf("Hello world from process %d out of %d\n", rank, ranksize);
//
//	end_time = MPI_Wtime(); // Засекаем конечное время
//
//	// Вычисляем и выводим время выполнения в микросекундах
//	if (rank == 0) {
//		printf("Total execution time: %f microseconds\n", (end_time - start_time) * 1e6);
//	}
//
//	MPI_Finalize();
//	return 0;
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
//	omp_set_num_threads(16);
//	int nThreads, threadNum;
//	double start_time, end_time;
//
//	// Засекаем начальное время
//	start_time = omp_get_wtime();
//
//#pragma omp parallel private(nThreads, threadNum)
//	{
//		nThreads = omp_get_num_threads();
//		threadNum = omp_get_thread_num();
//		printf("OpenMP thread %d out of %d threads\n", threadNum, nThreads);
//	}
//
//	// Засекаем конечное время
//	end_time = omp_get_wtime();
//
//	// Вычисляем и выводим общее время выполнения в секундах
//	printf("Total execution time: %f seconds\n", end_time - start_time);
//
//	return 0;
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
//        printf("Iteration %d out of %d threads\n", threadNum, nThreads);
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