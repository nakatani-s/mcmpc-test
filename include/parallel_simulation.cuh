/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      src/parallel_simulation.cu
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#include "common_header.cuh"
#include "data_structures.cuh"
#include "template.cuh"
#include "numerical_integrator.cuh"

unsigned int CountBlocks(unsigned int a, unsigned int b);

float GetCostValue(float *input, float *state, float *param, float *ref, float *cnstrnt, float *weight, IndexStructure *idx);

__device__ float GenerateRadomInput(unsigned int id, curandState *seed, float mean, float variance);

__global__ void SetRandomSeed(curandState *random_seed_vec, int seed);

__global__ void ParallelMonteCarloSimulation(SampleInfo *info, float *cost_vec, int *indices, float var, float *state, float *param, float *ref, float *cnstrnt, float *weight, float *mean, curandState *seed, IndexStructure *idx);
__global__ void GetWeightFromEliteSample(SampleInfo *info, float *weight_vec, IndexStructure *idx, int *indices);
__global__ void GetWeightedAverageInParallel(float *ret_value, SampleInfo *info, float *weight, float *denom_vec, int *indices, IndexStructure *idx);