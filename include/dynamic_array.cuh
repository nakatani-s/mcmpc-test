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

#include "common_header.cuh"
#include "managed.cuh"

#ifndef DYNAMIC_ARRAY_CUH
#define DYNAMIC_ARRAY_CUH

class DynamicArray : public Managed
{
    int length;
    float *data;
public:
    // Default Constructor
    DynamicArray();
    // Copy constructor
    DynamicArray(const DynamicArray& x);
    // Assignment operator
    DynamicArray& operator=(const int size);
    // destructor
    ~DynamicArray();
    // Access operator (from host or device)
    __host__ __device__ float& operator[](int pos);
    __host__ __device__ float* d_pointer();
private:
    void _realloc_data(int len);
};

#endif