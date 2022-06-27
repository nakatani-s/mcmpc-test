/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      include/mcmpc.cu
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <iomanip>

// include header files for using cuda API
#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

// include header files for using thrust library
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/transform.h>