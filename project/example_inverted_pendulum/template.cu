
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

const int OCP_SETTINGS::SIMULATION_STEPS        = 500;
const int OCP_SETTINGS::NUM_OF_PREDICTION_STEPS = 35;
const float OCP_SETTINGS::PREDICTION_INTERVAL   = 0.7f;
const float OCP_SETTINGS::CONTROL_CYCLE         = 0.02f;
const int OCP_SETTINGS::DIM_OF_STATE            = 4;
const int OCP_SETTINGS::DIM_OF_INPUT            = 1;
const int OCP_SETTINGS::DIM_OF_PARAMETER        = 7;
const int OCP_SETTINGS::DIM_OF_REFERENCE        = 4;
const int OCP_SETTINGS::DIM_OF_CONSTRAINTS      = 4;
const int OCP_SETTINGS::DIM_OF_WEIGHT_MATRIX    = 5;


/*****  *****/ 
const int CONTROLLER_PARAM::NUM_OF_SAMPLE                 = 9000;
const int CONTROLLER_PARAM::NUM_OF_ELITE_SAMPLE             = 100;
const int CONTROLLER_PARAM::NUM_OF_MONTE_CARLO_ITERATION    = 1;
const float CONTROLLER_PARAM::VARIANCE                      = 1.0f;


/***** OPTIONAL PARAMETERS *****/
const int OPTIONAL_PARAM::NUM_THREAD_PER_BLOCK      = 10;
const float OPTIONAL_PARAM::LAMBDA_GAIN             = 2e-1;

/***** PARAMETERS FOR SAMPLE-BASED NEWTON METHOD *****/
const int OPTIONAL_PARAM::NUM_OF_NEWTON_ITERATION   = 1;
const float OPTIONAL_PARAM::SBNEWTON_VARIANCE       = 1.0f;
const int OPTIONAL_PARAM::MAX_DIVISOR               = 30;

const float OPTIONAL_PARAM::COOLING_RATE            = 0.98f;

const float OPTIONAL_PARAM::BARIIER_ZETA            = 0.001f;
const float OPTIONAL_PARAM::BARIIER_RHO             = 1e-4;
const float OPTIONAL_PARAM::BARIIER_TAU             = 1e-2;
const float OPTIONAL_PARAM::BARIIER_MAX             = 1e7;

const int OPTIONAL_PARAM::NUM_OF_GOLDEN_SEARCH_ITERATION = 50;

/***** DYNAMIC MODEL REPRESENTING STATE TRANSITION dot{x} = "f(x,u,t,p)" *****/
__host__ __device__ void DynamicalModel(float *dx, float *x, float *u, float *param)
{
    float o[14];
    o[0] = param[3] + powf(param[2], 2) * param[1];
    o[1] = u[0] - x[1] * param[4] + powf(x[3], 2) * param[2] * param[1] * sinf(x[2]);
    o[2] = cosf(x[2]) * param[2] * param[1];
    o[3] = x[3] * param[5] - param[6] * param[2] * param[1] * sinf(x[2]);
    o[4] = -(o[0] * o[1] + o[2] * o[3]);

    o[5] = powf(cosf(x[2]) * param[1] * param[2], 2);
    o[6] = param[0] + param[1];
    o[7] = param[3] + powf(param[2], 2) * param[1];
    o[8] = o[5] - (o[6] * o[7]);

    o[9] = -(o[2] * o[1] + o[6] * o[3]);
    o[10] = param[3] * (param[0] + param[1]);
    o[11] = powf(param[2], 2) * param[1];
    o[12] = param[0] + param[1] - powf(cosf(x[2]), 2) * param[1];
    o[13] = o[10] + o[11] * o[12];

    dx[0] = x[1]; // dx
    dx[2] = x[3]; // dth
    dx[1] = o[4] / o[8]; // ddx
    dx[3] = o[9] / o[13]; //ddthta
} 

/***** FOR COMPUTING STAGE COST COST *****/
__host__ __device__ float GetStageCostTerm(float *u, float *x, float *reference, float *weight)
{
    float stage_cost = 0.0f;
    stage_cost += weight[0] * (x[0] - reference[0]) * (x[0] - reference[0]);
    stage_cost += weight[1] * (x[1] - reference[1]) * (x[1] - reference[1]);
    stage_cost += weight[2] * (sinf(x[2]/2.0) - reference[2]) * (sinf(x[2]/ 2.0) - reference[2]);
    stage_cost += weight[3] * (x[3] - reference[3]) * (x[3] - reference[3]);
    stage_cost += weight[4] * u[0] * u[0];
    
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
    u[0] = GetSaturatedInput(u[0], constraints[0], constraints[1], zeta);
}

/***** BARRIER FUNCTION FOR HANDILNG CONSTRAINTS ON STATE OR OTHERS *****/
__host__ __device__ float GetBarrierTerm(float *x, float *u, float *constraints, float rho)
{
    float log_barrier = 0.0f;
    
    log_barrier += LogBarrierConstraint(x[0], constraints[2], constraints[3], rho);

    return log_barrier;
}


