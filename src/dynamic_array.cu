/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      include/dyanmic_array.cuh
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#include "../include/dynamic_array.cuh"
#include "../include/cuda_check_error.cuh"

// Default Constructor
DynamicArray::DynamicArray(){ }

// Copy Constructor
DynamicArray::DynamicArray(const DynamicArray& x){ }

// destructor
DynamicArray::~DynamicArray(){ 
    CHECK( cudaFree(data) );
}

DynamicArray& DynamicArray::operator=(const int size)
{
    _realloc_data(size);
    for(int i = 0; i < size; i++)
    {
        data[i] = 0.0;
    }
    return *this;
}

__host__ __device__ float& DynamicArray::operator[](int pos)
{
    return data[pos];
}

__host__ __device__ float* DynamicArray::d_pointer()
{
    return data;
}

void DynamicArray::_realloc_data(int len)
{
    cudaFree(data);
    length = len;
    CHECK( cudaMallocManaged(&data, sizeof(float) * length) );
}