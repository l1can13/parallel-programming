#pragma region Лабораторная работа 1

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

#pragma endregion

#pragma region Лабораторная работа 2

/* Лабораторная работа 2 */

// OpenMP

//#include <omp.h>
//#define CHUNK 100
//#define NMAX 7600000
//#include "stdio.h"
//
//int main(int argc, char* argv[]) {
//	int num_proccesses[3] = { 2, 4, 8 };
//	double time = 0.0;
//
//
//	for (int k = 0; k < 3; ++k) {
//		time = 0.0;
//
//		for (int l = 0; l < 12; ++l) {
//			printf("\n\nNum proccesses = %d", num_proccesses[k]);
//			omp_set_num_threads(num_proccesses[k]);
//			int i, j;
//			int* a = new int[NMAX];
//			double sum;
//			int Q = 22;
//			for (i = 0; i < NMAX; i++) {
//				a[i] = 1.0;
//			}
//			double st_time, end_time;
//			st_time = omp_get_wtime();
//			sum = 0;
//
//#pragma omp parallel for shared(a) private(i, j) reduction(+:sum)
//			for (i = 0; i < NMAX; i++) {
//				for (j = 0; j < Q; j++) {
//					a[i] = (a[i] + a[i]) / 2;
//				}
//				//#pragma omp atomic
//				//#pragma omp critical
//				sum += a[i];
//			}
//			end_time = omp_get_wtime();
//			end_time = end_time - st_time;
//			printf("\nQ = %d", Q);
//			printf("\nTotal Sum = %10.2f", sum);
//			printf("\nTIME OF WORK IS %f ", end_time);
//			time += end_time;
//			delete[] a;
//		}
//
//		printf("\n\nNum Proccesses: %d\nAvg time for 12 starts: %f ", num_proccesses[k], (time / 12));
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
//	MPI_Scatter(x, N / ProcNum, MPI_INT, x_loc, N / ProcNum, MPI_INT, 0, MPI_COMM_WORLD);
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
//	MPI_Reduce(&ProcSum, &TotalSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
//	if (ProcRank == 0)
//	{
//		TotalSum = ProcSum;
//		for (i = 1; i < ProcNum; i++)
//		{
//			MPI_Recv(&ProcSum, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &Status);
//			TotalSum = TotalSum + ProcSum;
//		}
//	}
//	else
//		MPI_Send(&ProcSum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
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

//#include "stdio.h"
//#include <time.h>
//
//int main(int argc, char* argv[]) {
//	int Q = 22;
//	int i, j, N = 7600000;
//	int* x = new int[N];
//	double sum = 0;
//
//	for (int i = 0; i < N; ++i) {
//		x[i] = 1.0;
//	}
//
//	double start = clock();
//	for (int k = 0; k < 12; ++k) {
//		sum = 0.0;
//		for (i = 0; i < N; ++i) {
//			/*for (j = 0; j < Q; ++j) {
//				x[i] = (x[i] * 2) / 2;
//			}*/
//			sum += x[i];
//		}
//	}
//	double end = clock();
//	double t = (end - start) / CLOCKS_PER_SEC;
//	printf("\nTime: %f", t / 12);
//}

#pragma endregion

#pragma region Лабораторная работа 3

/* Лабораторная работа 3 */

// Sequential

//#define CHUNK 100 
//#define NMAX 7600000
//#define Q 22
//#define NUM_OF_THREADS 4
//
//#include <omp.h>
//#include <cstdlib>
//#include <time.h>
//#include "stdio.h"
//
//int main() {
//	int* a = new int[NMAX];
//	int* b = new int[NMAX];
//	int* sum = new int[NMAX];
//	int chunk, n, i, step, t, j, q, num;
//	chunk = CHUNK;
//	n = NMAX;
//	q = Q;
//	num = NUM_OF_THREADS;
//	//<инициализация данных>
//	for (i = 0; i < n; ++i) {
//		a[i] = i;
//		b[i] = i;
//	}
//	double st_time, end_time, time_sequential = 0.0;
//
//	for (t = 0; t < 12; t++) {
//		//последовательное выполнение
//		st_time = omp_get_wtime();
//		int* sum = new int[n];
//		for (int i = 0; i < n; ++i) {
//			for (int j = 0; j < Q; ++j) {
//				sum[i] = a[i] + b[i];
//			}
//		}
//		end_time = omp_get_wtime();
//		end_time = end_time - st_time;
//		time_sequential += end_time;
//	}
//	printf("Q: %d", q);
//	printf("\nTime of work SEQUENTIAL program: %f ", time_sequential / 12);
//	delete[] a;
//	delete[] b;
//	delete[] sum;
//	return 0;
//}

// OpenMP

//#define CHUNK 100 
//#define NMAX 7600000
//#define Q 22
//#define NUM_OF_THREADS 8
//#include <omp.h>
//#include <cstdlib>
//#include <time.h>
//#include "stdio.h"
//
//int main() {
//	int* a = new int[NMAX];
//	int* b = new int[NMAX];
//	int* sum = new int[NMAX];
//	int chunk, n, i, step, t, j, q, num;
//	chunk = CHUNK;
//	n = NMAX;
//	q = Q;
//	num = NUM_OF_THREADS;
//	omp_set_num_threads(num); // 2, 4, 8
//	//<инициализация данных>
//	for (i = 0; i < n; ++i) {
//		a[i] = i;
//		b[i] = i;
//	}
//	double start_time, end_time, result_time = 0.0;
//	for (t = 0; t < 12; t++) {
//		start_time = omp_get_wtime();
//		//параллельное выполнение
////#pragma omp parallel for shared (a,b,sum) schedule(static, chunk)
////		for (i = 0; i < n; ++i) {
////			for (j = 0; j < q; ++j) {
////				sum[i] = a[i] + b[i];
////			}
////		}
////#pragma omp parallel for shared (a,b,sum) schedule(dynamic, chunk)
////		for (i = 0; i < n; ++i) {
////			for (j = 0; j < q; ++j) {
////				sum[i] = a[i] + b[i];
////			}
////		}
//
//#pragma omp parallel for shared (a,b,sum) schedule(guided, chunk)
//		for (i = 0; i < n; ++i) {
//			for (j = 0; j < q; ++j) {
//				sum[i] = a[i] + b[i];
//			}
//		}
//		end_time = omp_get_wtime();
//
//		result_time += end_time - start_time;
//	}
//	printf("OpenMP program working on %d process", num);
//	printf("\nQ: %d", q);
//	//printf("\nTime of work STATIC program: %f ", result_time / 12);
//	//printf("\nTime of work DYNAMIC program: %f ", result_time / 12);
//	printf("\nTime of work GUIDED program: %f ", result_time / 12);
//	delete[] a;
//	delete[] b;
//	delete[] sum;
//	return 0;
//}

// MPI Scatter/Gather

//#include "mpi.h"
//#include <math.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <ctime>
//
//#define NMAX 7600000
//#define Q 22
//
//int main(int argc, char* argv[])
//{
//	int* a, * b, * c;
//	int* a_loc = NULL, * b_loc = NULL, * c_loc = NULL;
//	int ProcRank, ProcNum, N = NMAX, i, j, s, q = Q;
//	MPI_Status status;
//	int st_time, average_time = 0.0;
//	MPI_Init(&argc, &argv);
//	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
//	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
//	int count = N / ProcNum;
//	//инициализация
//	a = new int[N];
//	b = new int[N];
//	c = new int[N];
//	if (ProcRank == 0)
//	{
//		for (i = 0; i < N; ++i) {
//			a[i] = i;
//			b[i] = i;
//		}
//	}
//	for (s = 0; s < 12; ++s) {
//		a_loc = new int[N];
//		b_loc = new int[N];
//		c_loc = new int[N];
//
//		MPI_Scatter(a, count, MPI_INT, a_loc, count, MPI_INT, 0, MPI_COMM_WORLD);
//		MPI_Scatter(b, count, MPI_INT, b_loc, count, MPI_INT, 0, MPI_COMM_WORLD);
//
//		if (ProcRank == 0) {
//			st_time = MPI_Wtime();
//		}
//
//		for (i = 0; i < count; ++i) {
//			for (j = 0; j < q; ++j) {
//				c_loc[i] = a_loc[i] + b_loc[i];
//			}
//		}
//
//		MPI_Gather(c_loc, count, MPI_INT, c, count, MPI_INT, 0, MPI_COMM_WORLD);
//		MPI_Barrier(MPI_COMM_WORLD);
//		if (ProcRank == 0)
//			average_time += MPI_Wtime() - st_time;
//
//	}
//
//	if (ProcRank == 0)
//	{
//		printf("MPI program working on %d process", ProcNum);
//		printf("\nQ: %d", q);
//		printf("\nTime from work is %f\n", average_time / 12.0);
//		delete[] c;
//	}
//
//	delete[] a_loc;
//	delete[] b_loc;
//	delete[] c_loc;
//
//	MPI_Finalize();
//
//	return 0;
//}

// MPI Scatterv/Gatherv

//#include "mpi.h"
//#include <math.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <ctime>
//
//#define NMAX 7600000
//#define Q 22
//
//int main(int argc, char* argv[])
//{
//	int* a, * b, * c, s = 0, j;
//	int* a_loc = NULL, * b_loc = NULL, * c_loc = NULL;
//	int ProcRank, ProcNum, N = NMAX, i, q = Q;
//	int* sendCounts;
//	int* displs;
//	double st_time, average_time = 0.0;
//	MPI_Status status;
//	MPI_Init(&argc, &argv);
//	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
//	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
//	int count = N / ProcNum;
//	a = new int[N];
//	b = new int[N];
//	c = new int[N];
//	if (ProcRank == 0)
//	{
//
//		for (i = 0; i < N; ++i) {
//			a[i] = i;
//			b[i] = i;
//		}
//	}
//
//	sendCounts = new int[ProcNum];
//	displs = new int[ProcNum];
//
//	for (i = 0; i < ProcNum; ++i) {
//		sendCounts[i] = count;
//	}
//
//	for (i = 0; i < N % ProcNum; ++i) {
//		sendCounts[i] += 1;
//	}
//
//	displs[0] = 0; //массив смещений
//	for (i = 1; i < ProcNum; ++i) {
//		displs[i] = displs[i - 1] + sendCounts[i - 1];
//	}
//
//	for (s = 0; s < 12; ++s) {
//		a_loc = new int[sendCounts[ProcRank]];
//		b_loc = new int[sendCounts[ProcRank]];
//		c_loc = new int[sendCounts[ProcRank]];
//
//		MPI_Scatterv(a, sendCounts, displs, MPI_INT, a_loc, sendCounts[ProcRank], MPI_INT, 0, MPI_COMM_WORLD);
//		MPI_Scatterv(b, sendCounts, displs, MPI_INT, b_loc, sendCounts[ProcRank], MPI_INT, 0, MPI_COMM_WORLD);
//
//
//		if (ProcRank == 0) {
//			st_time = MPI_Wtime();
//		}
//
//		for (i = 0; i < sendCounts[ProcRank]; ++i) {
//			for (j = 0; j < q; ++j) {
//				c_loc[i] = a_loc[i] + b_loc[i];
//			}
//		}
//
//		MPI_Gatherv(c_loc, sendCounts[ProcRank], MPI_INT, c, sendCounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
//		MPI_Barrier(MPI_COMM_WORLD);
//		if (ProcRank == 0)
//			average_time += MPI_Wtime() - st_time;
//	}
//
//	if (ProcRank == 0)
//	{
//		printf("MPI program with Scatterv and Gatherv working on %d process", ProcNum);
//		printf("\nAmount of counting: %d", q);
//		printf("\nTime from work is %f\n", average_time / 12.0);
//		delete[] c;
//
//	}
//
//	delete[] a_loc;
//	delete[] b_loc;
//	delete[] c_loc;
//	delete[] sendCounts;
//	delete[] displs;
//
//	MPI_Finalize();
//
//	return 0;
//}

#pragma endregion