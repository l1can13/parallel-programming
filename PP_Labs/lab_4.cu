#include <cublas_v2.h>
#include "device_launch_parameters.h"
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>


__global__ void addKernel(int* c, int* a, int* b, unsigned int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    for (; index < size; index += (blockDim.x * gridDim.x)) {
        c[index] = a[index] + b[index];
    }
}

#define kernel addKernel
#define GRID_SIZE 1024
#define BLOCK_SIZE 1024
#define N 7600000


int main(int argc, char* argv[])
{

    int n = N;

    printf("n = %d\n", n);

    int n2b = n * sizeof(int);
    int n2 = n;

    // Выделение памяти на хосте
    int* a = (int*)calloc(n2, sizeof(int));
    int* b = (int*)calloc(n2, sizeof(int));
    int* c = (int*)calloc(n2, sizeof(int));
    // Инициализация массивов
    for (int i = 0; i < n; i++) {
        a[i] = 1;
        b[i] = 1;
    }

    // Выделение памяти на устройстве
    int* adev = NULL;
    cudaError_t cuerr = cudaMalloc((void**)&adev, n2b);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for a: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    int* bdev = NULL;
    cuerr = cudaMalloc((void**)&bdev, n2b);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for b: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    int* cdev = NULL;
    cuerr = cudaMalloc((void**)&cdev, n2b);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for c: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Создание обработчиков событий
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cuerr = cudaEventCreate(&start);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot create CUDA start event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    cuerr = cudaEventCreate(&stop);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot create CUDA end event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Копирование данных с хоста на девайс
    cuerr = cudaMemcpy(adev, a, n2b, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy a array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    cuerr = cudaMemcpy(bdev, b, n2b, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy b array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Установка точки старта
    cuerr = cudaEventRecord(start, 0);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot record CUDA event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    //Запуск ядра
    for (int i = 0; i < 12; ++i) {
        kernel <<< GRID_SIZE, BLOCK_SIZE >>> (cdev, adev, bdev, n);
    }

    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Синхронизация устройств
    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Установка точки окончания
    cuerr = cudaEventRecord(stop, 0);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Копирование результата на хост
    cuerr = cudaMemcpy(c, cdev, n2b, cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Расчет времени
    cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
    printf("time spent executing %s: %.9f seconds\n", "kernel", (gpuTime / 1000) / 12);

    // Очищение памяти
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
    free(a);
    free(b);
    free(c);

    return 0;
}