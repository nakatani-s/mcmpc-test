/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      include/managed.cuh
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#include "common_header.cuh"

#ifndef MANAGED_CUH
#define MANAGED_CUH

class Managed
{
public:
    void *operator new(size_t len);
    void *operator new[](size_t len);
    void operator delete(void *ptr);
};

#endif