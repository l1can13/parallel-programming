#include <math.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <locale.h>
#include <stdio.h>


int main(int argc, char* argv[])
{
	omp_set_num_threads(16);
	int nTheads, theadNum;
#pragma omp parallel  private(nTheads, theadNum)
	{
		nTheads = omp_get_num_threads();
		theadNum = omp_get_thread_num();
		printf("OpenMP thread ı%d from %d threads \n", theadNum, nTheads);
	}
	return 0;
}
