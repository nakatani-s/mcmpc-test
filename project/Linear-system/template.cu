
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

const int OCP_SETTINGS::SIMULATION_STEPS        = 1;
const int OCP_SETTINGS::NUM_OF_PREDICTION_STEPS = 30;
const float OCP_SETTINGS::PREDICTION_INTERVAL   = 0.3f;
const float OCP_SETTINGS::CONTROL_CYCLE         = 0.01f;
const int OCP_SETTINGS::DIM_OF_STATE            = 4;
const int OCP_SETTINGS::DIM_OF_INPUT            = 1;
const int OCP_SETTINGS::DIM_OF_PARAMETER        = 9;
const int OCP_SETTINGS::DIM_OF_REFERENCE        = 4;
const int OCP_SETTINGS::DIM_OF_CONSTRAINTS      = 4;
const int OCP_SETTINGS::DIM_OF_WEIGHT_MATRIX    = 5;


/*****  *****/ 
const int CONTROLLER_PARAM::NUM_OF_SAMPLE                 = 3000;
const int CONTROLLER_PARAM::NUM_OF_ELITE_SAMPLE             = 2400;
const int CONTROLLER_PARAM::NUM_OF_MONTE_CARLO_ITERATION    = 300;
const float CONTROLLER_PARAM::VARIANCE                      = sqrt(4.0f);


/***** OPTIONAL PARAMETERS *****/
const int OPTIONAL_PARAM::NUM_THREAD_PER_BLOCK      = 10;
const float OPTIONAL_PARAM::LAMBDA_GAIN             = 2e-1;

/***** PARAMETERS FOR SAMPLE-BASED NEWTON METHOD *****/
const int OPTIONAL_PARAM::NUM_OF_NEWTON_ITERATION   = 1;
const float OPTIONAL_PARAM::SBNEWTON_VARIANCE       = 0.25f;
const int OPTIONAL_PARAM::MAX_DIVISOR               = 50;

const float OPTIONAL_PARAM::COOLING_RATE            = 0.95f;

const float OPTIONAL_PARAM::BARIIER_ZETA            = 0.00001f;
const float OPTIONAL_PARAM::BARIIER_RHO             = 1e-4;
const float OPTIONAL_PARAM::BARIIER_TAU             = 1e-2;
const float OPTIONAL_PARAM::BARIIER_MAX             = 1e7;

const int OPTIONAL_PARAM::NUM_OF_GOLDEN_SEARCH_ITERATION = 4;

/***** DYNAMIC MODEL REPRESENTING STATE TRANSITION dot{x} = "f(x,u,t,p)" *****/
__host__ __device__ void DynamicalModel(float *dx, float *x, float *u, float *param)
{
    // float param[OCP_SETTINGS::DIM_OF_PARAMETER] = {1.0, -1.6658, -11.934, 3.5377e-6, 43.0344, 44.7524, -9.1392e-5, 9.43, 35.3774};
    dx[0] = param[0] * x[2];
    dx[1] = param[0] * x[3];
    dx[2] = param[1] * x[1] + param[2] * x[2] + param[3] * x[3] + param[7] * u[0];
    dx[3] = param[4] * x[1] + param[5] * x[2] + param[6] * x[3] + param[8] * u[0];    
} 

/***** FOR COMPUTING STAGE COST COST *****/
__host__ __device__ float GetStageCostTerm(float *u, float *x, float *reference, float *weight)
{
    float stage_cost = 0.0f;
    stage_cost += weight[0] * (x[0] - reference[0]) * (x[0] - reference[0]);
    stage_cost += weight[1] * (x[1] - reference[1]) * (x[1] - reference[1]);
    stage_cost += weight[2] * (x[2] - reference[2]) * (x[2] - reference[2]);
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
    // u[0] = GetSaturatedInput(u[0], constraints[0], constraints[1], zeta);
    u[0] = u[0];
}

/***** BARRIER FUNCTION FOR HANDILNG CONSTRAINTS ON STATE OR OTHERS *****/
__host__ __device__ float GetBarrierTerm(float *x, float *u, float *constraints, float rho)
{
    float log_barrier = 0.0f;
    
    // log_barrier += LogBarrierConstraint(x[0], constraints[2], constraints[3], rho);
    log_barrier += 0.0f;

    return log_barrier;
}


