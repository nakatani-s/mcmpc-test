/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      include/data_structures.cuh
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#include "common_header.cuh"

#ifndef TEMPLATE_CUH
#define TEMPLATE_CUH

struct OCP_SETTINGS
{
    static const int SIMULATION_STEPS;

    static const int DIM_OF_STATE;
    static const int DIM_OF_INPUT;
    static const int DIM_OF_PARAMETER;
    static const int DIM_OF_REFERENCE;
    static const int DIM_OF_CONSTRAINTS;
    static const int DIM_OF_WEIGHT_MATRIX;

    static const int NUM_OF_PREDICTION_STEPS;
    static const float PREDICTION_INTERVAL;
    static const float CONTROL_CYCLE;

};

struct CONTROLLER_PARAM
{
    static const int NUM_OF_SAMPLE;
    static const int NUM_OF_ELITE_SAMPLE;
    static const int NUM_OF_MONTE_CARLO_ITERATION;
    
    static const float VARIANCE;
};

struct OPTIONAL_PARAM
{
    // For GPU settings
    static const int NUM_THREAD_PER_BLOCK;
    // 重みの計算に一貫性を持たせるパラメータ
    static const float LAMBDA_GAIN; // 推奨値 0.2f (--> 1/5)
    // For Sample-based Newton method
    static const int NUM_OF_NEWTON_ITERATION;
    static const int MAX_DIVISOR;
    static const float SBNEWTON_VARIANCE;
    // For using Geometric Cooling method
    static const float COOLING_RATE;
    // バリア関数で使用 (For Barrier Function)
    static const float BARIIER_ZETA;
    static const float BARIIER_RHO;
    static const float BARIIER_TAU;
    static const float BARIIER_MAX;
    // 黄金分割探索で使用
    static const int NUM_OF_GOLDEN_SEARCH_ITERATION;
};

// 
__host__ __device__ float GetSaturatedInput(float input, float lower_limit, float upper_limit, float zeta);
__host__ __device__ float LogBarrierConstraint(float object, float lower_limit, float upper_limit, float rho);

// User-defined function (ユーザーが定義する関数の宣言)
__host__ __device__ void InputSaturation(float *u, float *constraints, float zeta);
__host__ __device__ float GetBarrierTerm(float *x, float *u, float *constraints, float rho);
__host__ __device__ void DynamicalModel(float *dx, float *x, float *u, float *param);
__host__ __device__ float GetStageCostTerm(float *u, float *x, float *ref, float *weight);
__host__ __device__ float GetTerminalCostTerm(float *u, float *x, float *ref, float *weight);

#endif