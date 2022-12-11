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
#include "managed.cuh"
#include "dynamic_array.cuh"
#include "template.cuh"

#ifndef DATA_STRUCTURES_CUH
#define DATA_STRUCTURES_CUH

// #define CMA_DEFAULT

struct SampleInfo : public Managed
{
    float cost;
    float weight;
    DynamicArray input;
    DynamicArray dev_state;
    DynamicArray dev_input;
    DynamicArray dev_dstate;
    DynamicArray dev_ref;
};

struct SampleInfoCMA : public Managed
{
    float cost;
    float weight;
    DynamicArray input;
    DynamicArray dy;
    DynamicArray dev_input;
    DynamicArray dev_state;
    DynamicArray dev_dstate;
    DynamicArray dev_ref;
};

typedef struct{
    int horizon;
    int dim_of_input;
    int dim_of_state;
    int dim_of_reference;
    int dim_of_parameter;
    int dim_of_constraints;
    int dim_of_weight_matrix;
    int sample_size;
    int elite_sample_size;
    int monte_calro_iteration;
    int newton_iteration;

    unsigned int input_by_horizon;
    
    unsigned int size_of_hessian;
    unsigned int size_of_hessian_element;
    unsigned int size_of_quadrtic_curve;
    unsigned int pow_hessian_elements;
    unsigned int sample_size_for_fitting;

    float control_cycle;
    float prediction_interval;
    float sigma;

    int max_divisor;
    unsigned int thread_per_block;
    float newton_search_sigma;
    float zeta;
    float rho;
    float cooling_rate;
    float lambda_gain;
    float barrier_tau;
    float barrier_max;

    int golden_search_iteration;
    float golden_ratio;

}IndexStructure;

typedef struct
{
    int sample_size_cma;
    int elite_sample_cma;
    float cma_xi;
    float cma_chi;
    float learning_rate_zeta;
    float learning_rate_c;
    float learning_rate_muW;
    float update_rate_top;
    float update_rate_mu;
    float damping_ratio;
}IndexCMA;

enum ValueType{
    SET_STATE, SET_INPUT, SET_PARAMETER, SET_CONSTRAINT, SET_WEIGHT_MATRIX, SET_REFERENCE, SET_COOLING_METHOD, SET_REFERENCE_TYPE, SET_STEP_WIDTH_ADJUSTING_METHOD,
    SET_SOLVER
};

enum ReferenceType{
    TIME_INVARIANT, TIME_VARIANT
};

enum CoolingMethod{
    GEOMETRIC, HYPERBOLIC, NOTHING
};

enum IntegralMethod{
    EULAR, RUNGE_KUTTA_45
};

enum StepWidthDecisiveMethod{
    GOLDEN_SECTION, TERNARY, NOT_SETTING
};

enum LinearEquationSolver{
    EIGEN_VALUE_DECOM, QR_DECOM
};

void SetupIndices(IndexStructure *idx);
void SetupIndicesSampleBasedNewton(IndexStructure *idx);
void SetupStructure(SampleInfo *info, int num, IndexStructure *idx);
void SetupIndicesCMA(IndexCMA *c_idx, IndexStructure *idx);
void SetupStructureCMA(SampleInfoCMA *cinfo, IndexCMA *c_idx, IndexStructure *idx);

#endif