
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
const int OCP_SETTINGS::NUM_OF_PREDICTION_STEPS = 15;
const float OCP_SETTINGS::PREDICTION_INTERVAL   = 0.7f;
const float OCP_SETTINGS::CONTROL_CYCLE         = 0.02f;
const int OCP_SETTINGS::DIM_OF_STATE            = 13;
const int OCP_SETTINGS::DIM_OF_INPUT            = 4;
const int OCP_SETTINGS::DIM_OF_PARAMETER        = 11;
const int OCP_SETTINGS::DIM_OF_REFERENCE        = 4;
const int OCP_SETTINGS::DIM_OF_CONSTRAINTS      = 6;
const int OCP_SETTINGS::DIM_OF_WEIGHT_MATRIX    = 16;


/*****  *****/ 
const int CONTROLLER_PARAM::NUM_OF_SAMPLE                 = 9000;
const int CONTROLLER_PARAM::NUM_OF_ELITE_SAMPLE             = 100;
const int CONTROLLER_PARAM::NUM_OF_MONTE_CARLO_ITERATION    = 2;
const float CONTROLLER_PARAM::VARIANCE                      = 1.25f;


/***** OPTIONAL PARAMETERS *****/
const int OPTIONAL_PARAM::NUM_THREAD_PER_BLOCK      = 10;
const float OPTIONAL_PARAM::LAMBDA_GAIN             = 2e-1;

/***** PARAMETERS FOR SAMPLE-BASED NEWTON METHOD *****/
const int OPTIONAL_PARAM::NUM_OF_NEWTON_ITERATION   = 1;
const float OPTIONAL_PARAM::SBNEWTON_VARIANCE       = 1.2f;
const int OPTIONAL_PARAM::MAX_DIVISOR               = 100;

const float OPTIONAL_PARAM::COOLING_RATE            = 0.98f;

const float OPTIONAL_PARAM::BARIIER_ZETA            = 0.001f;
const float OPTIONAL_PARAM::BARIIER_RHO             = 1e-4;
const float OPTIONAL_PARAM::BARIIER_TAU             = 1e-2;
const float OPTIONAL_PARAM::BARIIER_MAX             = 1e7;

const int OPTIONAL_PARAM::NUM_OF_GOLDEN_SEARCH_ITERATION = 50;

/***** DYNAMIC MODEL REPRESENTING STATE TRANSITION dot{x} = "f(x,u,t,p)" *****/
__host__ __device__ void DynamicalModel(float *dx, float *x, float *u, float *param)
{
    float o[10] = { };
    o[0] = param[9] * param[3];
    o[1] = param[1] + u[0] - u[2] + u[3];  // u1 = u[0], u2 = u[1], u3 = u[2], u4 = u[3]
    o[2] = param[1] + u[0] + u[2] + u[3];
    o[3] = param[1] + u[0] + u[1] - u[3];
    o[4] = param[1] + u[0] - u[1] - u[3];
    o[5] = o[1] * fabs(o[1]);
    o[6] = o[2] * fabs(o[2]);
    o[7] = o[3] * fabs(o[3]);
    o[8] = o[4] * fabs(o[4]);
    o[9] = o[5] + o[6] + o[7] + o[8];

    // dot{X}
    dx[0] = x[1];
    // dot{dot{X}}
    dx[1] = 2.0 * param[4] * (x[9] * x[11] + x[10] * x[12]) * o[9] / o[0];
    // dot{Y}
    dx[2] = x[3];
    // dot{dot{Y}}
    dx[3] = -2.0 * param[4] * (x[9] * x[10] - x[11] * x[12]) * o[9] / o[0];
    // dot{Z}
    dx[4] = x[5];
    // dot{dot{Z}}
    dx[5] = (param[4]*(2.0 * x[9] * x[9] + 2.0 * x[12] * x[12] - 1.0) * o[9] / o[0]) - param[0];
    // dot{Gamma}
    dx[6] = 0.5 * ( 2.0 * (param[7] - param[8]) * x[7] * x[8] + param[10] * param[4]* (o[7]-o[8]) / param[3]) / param[6];
    // dot{Beta}
    dx[7] = -0.5 * ( 2.0 * (param[6] - param[8]) * x[6] * x[8] + param[10] * param[4]* (o[5]-o[6]) / param[3]) / param[7];
    // dot{alpha}
    dx[8] = ((param[6] - param[7]) * x[6] * x[7] - param[5] * (o[7]+o[8]-o[5]-o[6])) / param[8];
    // dot{Quaternion_W}
    dx[9] = -0.5 * x[10] * x[6] - 0.5 * x[11] * x[7] - 0.5 * x[12] * x[8];
    // dot{Quaternion_X}
    dx[10] = 0.5 * x[9] * x[6] + 0.5 * x[11] * x[8] - 0.5 * x[12] * x[7];
    // dot{Quaternion_Y}
    dx[11] = 0.5 * x[9] * x[7] + 0.5 * x[12] * x[6] - 0.5 * x[10] * x[8];
    // dot{Quaternion_Z}
    dx[12] = 0.5 * x[9] * x[8] + 0.5 * x[10] * x[7] - 0.5 * x[11] * x[6];
} 

/***** FOR COMPUTING STAGE COST COST *****/
__host__ __device__ float GetStageCostTerm(float *u, float *x, float *reference, float *weight)
{
    float stage_cost = 0.0f;
    stage_cost += weight[0] * (x[0] - reference[1]) * (x[0] - reference[1]); // q_11 * (x - ref{x})^2
    stage_cost += weight[2] * (x[2] - reference[2]) * (x[2] - reference[2]); // q_33 * (y - ref{y})^2
    stage_cost += weight[4] * (x[4] - reference[3]) * (x[4] - reference[3]); // q_55 * (z - ref{z})^2
    int index = 0;
    for(int i = 0; i < 3; i++)
    { 
        if(index < 5) index = 2 * i + 1;
        stage_cost += weight[index] * x[index] * x[index];
    }
    index += 1;
    while(index < 9)
    {
        stage_cost += weight[index] * x[index] * x[index];
        index += 1;
    }
    stage_cost += weight[9] * x[10] * x[10] + weight[10] * x[11] * x[11] + weight[11] * x[12] * x[12];
    stage_cost += weight[12] * (u[0] - reference[0]) * (u[0] - reference[0]);
    stage_cost += weight[13] * u[1] * u[1] + weight[14] * u[2] * u[2] + weight[15] * u[3] * u[3];
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
    for(int i = 1; i < 4; i++)
    {
        u[i] = GetSaturatedInput(u[i], constraints[2], constraints[3], zeta);
    }
    u[0] = GetSaturatedInput(u[0], constraints[4], constraints[5], zeta);
}

/***** BARRIER FUNCTION FOR HANDILNG CONSTRAINTS ON STATE OR OTHERS *****/
__host__ __device__ float GetBarrierTerm(float *x, float *u, float *constraints, float rho)
{
    float log_barrier = 0.0f;
    for(int i = 1; i < 4; i++)
    {
        log_barrier += LogBarrierConstraint(u[i], constraints[2], constraints[3], rho);
    }

    log_barrier += LogBarrierConstraint(u[0], constraints[4], constraints[5], rho);

    return log_barrier;
}


