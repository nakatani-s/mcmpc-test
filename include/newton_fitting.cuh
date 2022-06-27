/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      include/newton_fitting.cuh
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/
#ifndef NEWTON_FITTING_CUH
#define NEWTON_FITTING_CUH

#include "common_header.cuh"
#include "../include/data_structures.cuh"

__global__ void SetInputSequences(float *output, float *optima);
__global__ void GetTensortMatrices(float *matrix, float *vector, float *mean, float cost, SampleInfo *info, int *indices, IndexStructure *idx);
__global__ void GetHessinaAndGradient(float *hessian, float *gradient, float *lsm_result_vec, IndexStructure *idx);
__global__ void ComputeNewtonStep(float *ans, float *current_guess, float *newton_step);

#endif