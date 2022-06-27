/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      src/mcmpc.cu
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#include "../include/managed.cuh"
#include "../include/cuda_check_error.cuh"

void* Managed::operator new(size_t len)
{
    void *ptr;
    CHECK( cudaMallocManaged(&ptr, len) );
    CHECK( cudaDeviceSynchronize() );
    // printf("called Managed object!!\n");
    return ptr;
}

void* Managed::operator new[](size_t len)
{
    void *ptr;
    CHECK( cudaMallocManaged(&ptr, len) );
    CHECK( cudaDeviceSynchronize() );
    // printf("called Managed object!!\n");
    return ptr;
}

void Managed::operator delete(void *ptr)
{
    CHECK( cudaDeviceSynchronize() );
    cudaFree(ptr);
}