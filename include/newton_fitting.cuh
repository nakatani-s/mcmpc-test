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

__global__ void GetNegativeEigenValue(int *indicator, float *eigen_value, IndexStructure *idx);
__global__ void ResetTensortMatrices(float *matrix, SampleInfo *info, int *indices, IndexStructure *idx);
__global__ void GetMeanAbsoluteError(float *mae, float *prev_value, SampleInfo *info, int *indices, IndexStructure *idx);
__global__ void GetMeanSquareError(float *mse, float *prev_value, SampleInfo *info, int *indices, IndexStructure *idx);
#endif