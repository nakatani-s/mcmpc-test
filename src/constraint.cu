/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      src/constraint.cu
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#include "../include/template.cuh"

__host__ __device__ float GetSaturatedInput(float input, float lower_limit, float upper_limit, float zeta)
{
    float ret_value = input;
    if(input < lower_limit) ret_value = lower_limit + zeta;
    if(input > upper_limit) ret_value = upper_limit - zeta;

    return ret_value;
}

__host__ __device__ float LogBarrierConstraint(float object, float lower_limit, float upper_limit, float rho)
{
    float ret_value = 0.0f;
    ret_value += -logf(object - lower_limit) - logf(upper_limit - object);
    ret_value += rho * (upper_limit - lower_limit);

    return ret_value;
}
