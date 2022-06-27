/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      src/numerical_integrator.cu
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#include "../include/numerical_integrator.cuh"
#include "../include/template.cuh"

__host__ __device__ void EularIntegration(float *state, float *d_state, float delta, int state_dimention)
{
    for(int state_id = 0; state_id < state_dimention; state_id++)
    {
        state[state_id] = state[state_id] + (delta * d_state[state_id]);
    }
}

__host__ __device__ void RungeKutta45(float *state, int state_dim, float *input, float *param, float delta)
{
    float *d_state, *yp_1, *next_state;
    float *yp_2, *yp_3, *yp_4;
    d_state = (float *)malloc(sizeof(float) * state_dim);
    yp_1 = (float *)malloc(sizeof(float) * state_dim);
    yp_2 = (float *)malloc(sizeof(float) * state_dim);
    yp_3 = (float *)malloc(sizeof(float) * state_dim);
    yp_4 = (float *)malloc(sizeof(float) * state_dim);
    next_state = (float *)malloc(sizeof(float) * state_dim);

    for(int i = 0; i < state_dim; i++)
    {
        yp_1[i] = 0.0f;
        yp_2[i] = 0.0f;
        yp_3[i] = 0.0f;
        yp_4[i] = 0.0f;
    }

    DynamicalModel(d_state, state, input, param);
    EularIntegration(yp_1, d_state, delta, state_dim);
    for(int i = 0; i < state_dim; i++)
    {
        next_state[i] = state[i] + yp_1[i] / 2;
    }

    DynamicalModel(d_state, next_state, input, param);
    EularIntegration(yp_2, d_state, delta, state_dim);
    for(int i = 0; i < state_dim; i++)
    {
        next_state[i] = state[i] + yp_2[i] / 2;
    }

    DynamicalModel(d_state, next_state, input, param);
    EularIntegration(yp_3, d_state, delta, state_dim);
    for(int i = 0; i < state_dim; i++)
    {
        next_state[i] = state[i] + yp_3[i];
    }

    DynamicalModel(d_state, next_state, input, param);
    EularIntegration(yp_4,  d_state, delta, state_dim);

    for(int i = 0; i < state_dim; i++)
    {
        state[i] = state[i] +(yp_1[i] + 2 * yp_2[i] + 2 * yp_3[i] + yp_4[i]) / 6.0;
    }

    free(d_state);
    free(yp_1);
    free(yp_2);
    free(yp_3);
    free(yp_4);
    free(next_state);
}
