/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      include/numerical_integrator.cuh
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#include "common_header.cuh"

#ifndef NUMERICAL_INTEGRATOR_CUH
#define NUMERICAL_INTEGRATOR_CUH

__host__ __device__ void EularIntegration(float *state, float *d_state, float delta, int state_dimention);
__host__ __device__ void RungeKutta45(float *state, int state_dim, float *input, float *param, float delta);

#endif