
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

const int OCP_SETTINGS::SIMULATION_STEPS        = 750;
const int OCP_SETTINGS::NUM_OF_PREDICTION_STEPS = 40;
const float OCP_SETTINGS::PREDICTION_INTERVAL   = 1.2f;
const float OCP_SETTINGS::CONTROL_CYCLE         = 0.02f;
const int OCP_SETTINGS::DIM_OF_STATE            = 6;
const int OCP_SETTINGS::DIM_OF_INPUT            = 1;
const int OCP_SETTINGS::DIM_OF_PARAMETER        = 8;
const int OCP_SETTINGS::DIM_OF_REFERENCE        = 6;
const int OCP_SETTINGS::DIM_OF_CONSTRAINTS      = 4;
const int OCP_SETTINGS::DIM_OF_WEIGHT_MATRIX    = 7;


/*****  *****/ 
const int CONTROLLER_PARAM::NUM_OF_SAMPLE                 = 9000;
const int CONTROLLER_PARAM::NUM_OF_ELITE_SAMPLE             = 150;
const int CONTROLLER_PARAM::NUM_OF_MONTE_CARLO_ITERATION    = 4;
const float CONTROLLER_PARAM::VARIANCE                      = 0.8f;


/***** OPTIONAL PARAMETERS *****/
const int OPTIONAL_PARAM::NUM_THREAD_PER_BLOCK      = 10;
const float OPTIONAL_PARAM::LAMBDA_GAIN             = 2e-1;

/***** PARAMETERS FOR SAMPLE-BASED NEWTON METHOD *****/
const int OPTIONAL_PARAM::NUM_OF_NEWTON_ITERATION   = 1;
const float OPTIONAL_PARAM::SBNEWTON_VARIANCE       = 0.25f;
const int OPTIONAL_PARAM::MAX_DIVISOR               = 50;

const float OPTIONAL_PARAM::COOLING_RATE            = 0.98f;

const float OPTIONAL_PARAM::BARIIER_ZETA            = 0.00001f;
const float OPTIONAL_PARAM::BARIIER_RHO             = 1e-4;
const float OPTIONAL_PARAM::BARIIER_TAU             = 1e-2;
const float OPTIONAL_PARAM::BARIIER_MAX             = 1e7;

const int OPTIONAL_PARAM::NUM_OF_GOLDEN_SEARCH_ITERATION = 4;

/***** PARAMETERS FOR MPC with CMA-ES  *****/
const int OPTIONAL_PARAM::SAMPLE_SIZE_CMA       = 40;
const int OPTIONAL_PARAM::ELITE_SAMPLE_CMA      = 7;
const float OPTIONAL_PARAM::CMA_XI              = 0.5f;
const float OPTIONAL_PARAM::LEARNING_RATE_Z     = 1.0f;
const float OPTIONAL_PARAM::LEARNING_RATE_C     = 1.0f;
const float OPTIONAL_PARAM::DAMPING_COEFFICIENT = 1.0f;
const float OPTIONAL_PARAM::PATH_UPDATE_RATE_Z  = 1.0f;
const float OPTIONAL_PARAM::PATH_UPDATE_RATE_C  = 1.0f;

/***** DYNAMIC MODEL REPRESENTING STATE TRANSITION dot{x} = "f(x,u,t,p)" *****/
__host__ __device__ void DynamicalModel(float *dx, float *x, float *u, float *param)
{
    float o[7] = { };
    o[0] = sinf(x[2] - x[4]);
    o[1] = -param[2] * o[0] * powf(x[5], 2) - (param[5] + param[6]) * x[3] + param[6] * x[5] + param[3] * param[7] * sinf(x[2]) - param[3] * cosf(x[2]) * u[0];
    o[2] = param[2] * o[0] * powf(x[3], 2) + param[6] * (x[3] - x[5]) + param[4] * param[7] * sinf(x[4]) - param[4] * cosf(x[4]) * u[0];
    o[3] = param[0] * param[1] - powf(param[2] * cosf(o[2] - o[4]), 2);
    o[4] = param[1] / o[3];
    o[5] = -param[2] * cosf(o[2] - o[4]) /o[3];
    o[6] = param[0] / o[3];
    
    dx[0] = x[1]; // dx
    dx[1] = u[0]; // ddx
    dx[2] = x[3]; // dtheta1
    dx[3] = o[4] * o[1] + o[5] * o[2]; // ddtheta1
    dx[4] = x[5]; // dtheta2
    dx[5] = o[5] * o[0] + o[6] * o[2];
    
} 

/***** FOR COMPUTING STAGE COST COST *****/
__host__ __device__ float GetStageCostTerm(float *u, float *x, float *reference, float *weight)
{
    float stage_cost = 0.0f;
    float o[2] = {};
    o[0] = sinf(x[2]/2.0f);
    o[1] = sinf(x[4]/2.0f);
    // 状態に関するステージコスト
    stage_cost += weight[0] * (x[0] - reference[0]) * (x[0] - reference[0]);
    stage_cost += weight[1] * (x[1] - reference[1]) * (x[1] - reference[1]);
    stage_cost += weight[2] * (o[0] - reference[2]) * (o[0] - reference[2]);
    stage_cost += weight[3] * (x[3] - reference[3]) * (x[3] - reference[3]);
    stage_cost += weight[4] * (o[1] - reference[4]) * (o[1] - reference[4]);
    stage_cost += weight[5] * (x[5] - reference[5]) * (x[5] - reference[5]);
    // 入力に関するステージコスト
    stage_cost += weight[6] * u[0] * u[0];
    
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


