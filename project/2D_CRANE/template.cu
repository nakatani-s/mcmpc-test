
/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      project/example_inverted_pendulum/template.cu
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#include "../../include/mcmpc_toolkit.cuh"

const int OCP_SETTINGS::SIMULATION_STEPS        = 1000;
const int OCP_SETTINGS::NUM_OF_PREDICTION_STEPS = 20;
const float OCP_SETTINGS::PREDICTION_INTERVAL   = 1.0f;
const float OCP_SETTINGS::CONTROL_CYCLE         = 0.02f;
const int OCP_SETTINGS::DIM_OF_STATE            = 6;
const int OCP_SETTINGS::DIM_OF_INPUT            = 2;
const int OCP_SETTINGS::DIM_OF_PARAMETER        = 2;
const int OCP_SETTINGS::DIM_OF_REFERENCE        = 6;
const int OCP_SETTINGS::DIM_OF_CONSTRAINTS      = 4;
const int OCP_SETTINGS::DIM_OF_WEIGHT_MATRIX    = 8;


/*****  *****/ 
const int CONTROLLER_PARAM::NUM_OF_SAMPLE                 = 9000;
const int CONTROLLER_PARAM::NUM_OF_ELITE_SAMPLE             = 100;
const int CONTROLLER_PARAM::NUM_OF_MONTE_CARLO_ITERATION    = 1;
const float CONTROLLER_PARAM::VARIANCE                      = 1.25f;


/***** OPTIONAL PARAMETERS *****/
const int OPTIONAL_PARAM::NUM_THREAD_PER_BLOCK      = 10;
const float OPTIONAL_PARAM::LAMBDA_GAIN             = 2e-1;

/***** PARAMETERS FOR SAMPLE-BASED NEWTON METHOD *****/
const int OPTIONAL_PARAM::NUM_OF_NEWTON_ITERATION   = 1;
const float OPTIONAL_PARAM::SBNEWTON_VARIANCE       = 1.2f;
const int OPTIONAL_PARAM::MAX_DIVISOR               = 50;

const float OPTIONAL_PARAM::COOLING_RATE            = 0.98f;

const float OPTIONAL_PARAM::BARIIER_ZETA            = 0.001f;
const float OPTIONAL_PARAM::BARIIER_RHO             = 1e-4;
const float OPTIONAL_PARAM::BARIIER_TAU             = 1e-2;
const float OPTIONAL_PARAM::BARIIER_MAX             = 1e7;

const int OPTIONAL_PARAM::NUM_OF_GOLDEN_SEARCH_ITERATION = 5;

/***** DYNAMIC MODEL REPRESENTING STATE TRANSITION dot{x} = "f(x,u,t,p)" *****/
__host__ __device__ void DynamicalModel(float *dx, float *x, float *u, float *param)
{
    float a[6];
    a[0] = sinf(x[4]);
    a[1] = 2.0f * x[3] * x[5];
    a[2] = cosf(x[4]);
    a[3] = param[1] * a[0] + a[1];
    a[4] = -a[3] / x[2];
    a[5] = -a[2] / x[2];

    dx[0] = x[1];
    dx[1] = param[0] * u[0];
    dx[2] = x[3];
    dx[3] = param[0] * u[1];
    dx[4] = x[5];
    dx[5] = a[4] + a[5] * u[0];
} 

/***** FOR COMPUTING STAGE COST COST *****/
__host__ __device__ float GetStageCostTerm(float *u, float *x, float *reference, float *weight)
{
    float stage_cost = 0.0f;
    
    for(int i = 0; i < 6; i++)
    {
        stage_cost += weight[i] * (x[i] - reference[i]) * (x[i] - reference[i]);
    }
    stage_cost += weight[6] * (u[0] - reference[5]) * (u[0] - reference[5]);
    stage_cost += weight[7] * (u[1] - reference[5]) * (u[0] - reference[5]);
    stage_cost = stage_cost / 2;

    return stage_cost;
}

/***** FOR COMPUTING TERMINAL COST COST *****/
__host__ __device__ float GetTerminalCostTerm(float *u, float *x, float *ref, float *weight)
{
    float terminal_cost = 0.0f;

    return terminal_cost;
}

/***** FOR HANDLING INPUT CONSTRAINTS ON RANDOMLY GENERATED INPUT *****/ 
__host__ __device__ void InputSaturation(float *u, float *constraints, float zeta)
{
    u[0] = GetSaturatedInput(u[0], constraints[2], constraints[3], zeta);
    u[1] = GetSaturatedInput(u[1], constraints[2], constraints[3], zeta);
}

/***** BARRIER FUNCTION FOR HANDILNG CONSTRAINTS ON STATE OR OTHERS *****/
__host__ __device__ float GetBarrierTerm(float *x, float *u, float *constraints, float rho)
{
    float log_barrier = 0.0f;
    
    float a[7] ={ };
    a[0] = x[2] * sinf(x[4]);
    a[1] = 0.2f;
    a[2] = powf(x[0] + a[0], 2);
    a[3] = a[1] * a[2];
    a[4] = 1.25f;
    a[5] = a[3] + a[4];
    a[6] = x[2] * cosf(x[4]);

    log_barrier += LogBarrierConstraint(a[6], 0.0f, a[5], rho);
    log_barrier += LogBarrierConstraint(x[5], constraints[0], constraints[1], rho);
    // log_barrier += LogBarrierConstraint(a[6], 0.0f, a[5], rho);
    return log_barrier;
}


