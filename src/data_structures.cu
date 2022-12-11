/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      src/mcmpc.cu
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#include "../include/data_structures.cuh"

void SetupIndices(IndexStructure *idx)
{
    int input_by_horizon_temp = OCP_SETTINGS::DIM_OF_INPUT * OCP_SETTINGS::NUM_OF_PREDICTION_STEPS;

    idx->horizon = OCP_SETTINGS::NUM_OF_PREDICTION_STEPS;
    idx->dim_of_input = OCP_SETTINGS::DIM_OF_INPUT;
    idx->dim_of_state = OCP_SETTINGS::DIM_OF_STATE;
    idx->dim_of_reference = OCP_SETTINGS::DIM_OF_REFERENCE;
    idx->dim_of_weight_matrix = OCP_SETTINGS::DIM_OF_WEIGHT_MATRIX;
    idx->dim_of_constraints = OCP_SETTINGS::DIM_OF_CONSTRAINTS;
    idx->dim_of_parameter = OCP_SETTINGS::DIM_OF_PARAMETER;

    idx->sample_size = CONTROLLER_PARAM::NUM_OF_SAMPLE;
    idx->elite_sample_size = CONTROLLER_PARAM::NUM_OF_ELITE_SAMPLE;
    idx->monte_calro_iteration = CONTROLLER_PARAM::NUM_OF_MONTE_CARLO_ITERATION;
    idx->newton_iteration = OPTIONAL_PARAM::NUM_OF_NEWTON_ITERATION;

    idx->thread_per_block = OPTIONAL_PARAM::NUM_THREAD_PER_BLOCK;
    idx->input_by_horizon = input_by_horizon_temp;
    idx->control_cycle = OCP_SETTINGS::CONTROL_CYCLE;
    idx->prediction_interval = OCP_SETTINGS::PREDICTION_INTERVAL;
    idx->sigma = CONTROLLER_PARAM::VARIANCE;

    idx->cooling_rate = OPTIONAL_PARAM::COOLING_RATE;
    idx->zeta = OPTIONAL_PARAM::BARIIER_ZETA;
    idx->rho = OPTIONAL_PARAM::BARIIER_RHO;
    idx->lambda_gain = OPTIONAL_PARAM::LAMBDA_GAIN;
    idx->barrier_max = OPTIONAL_PARAM::BARIIER_MAX;
    idx->barrier_tau = OPTIONAL_PARAM::BARIIER_TAU;
    // idx->dim_of_hessian = input_by_horizon_temp * input_by_horizon_temp;

}

void SetupIndicesSampleBasedNewton(IndexStructure *idx)
{
    int input_by_horizon_temp = OCP_SETTINGS::DIM_OF_INPUT * OCP_SETTINGS::NUM_OF_PREDICTION_STEPS;
    int size_hessian_element;
    int size_coe_quadric_curve;
    int minimum_require;
    int counter = 0;
    int tpb = OPTIONAL_PARAM::NUM_THREAD_PER_BLOCK;

    idx->horizon = OCP_SETTINGS::NUM_OF_PREDICTION_STEPS;
    idx->dim_of_input = OCP_SETTINGS::DIM_OF_INPUT;
    idx->dim_of_state = OCP_SETTINGS::DIM_OF_STATE;
    idx->dim_of_reference = OCP_SETTINGS::DIM_OF_REFERENCE;
    idx->dim_of_weight_matrix = OCP_SETTINGS::DIM_OF_WEIGHT_MATRIX;
    idx->dim_of_constraints = OCP_SETTINGS::DIM_OF_CONSTRAINTS;
    idx->dim_of_parameter = OCP_SETTINGS::DIM_OF_PARAMETER;

    idx->sample_size = CONTROLLER_PARAM::NUM_OF_SAMPLE;
    idx->elite_sample_size = CONTROLLER_PARAM::NUM_OF_ELITE_SAMPLE;
    idx->monte_calro_iteration = CONTROLLER_PARAM::NUM_OF_MONTE_CARLO_ITERATION;
    idx->newton_iteration = OPTIONAL_PARAM::NUM_OF_NEWTON_ITERATION;

    idx->thread_per_block = OPTIONAL_PARAM::NUM_THREAD_PER_BLOCK;
    idx->input_by_horizon = input_by_horizon_temp;
    idx->control_cycle = OCP_SETTINGS::CONTROL_CYCLE;
    idx->prediction_interval = OCP_SETTINGS::PREDICTION_INTERVAL;
    idx->sigma = CONTROLLER_PARAM::VARIANCE;

    idx->size_of_hessian = input_by_horizon_temp * input_by_horizon_temp;
    size_hessian_element = (int)( (input_by_horizon_temp * (input_by_horizon_temp + 1) ) / 2);
    idx->size_of_hessian_element = size_hessian_element;
    size_coe_quadric_curve = size_hessian_element + input_by_horizon_temp + 1;
    idx->size_of_quadrtic_curve = size_coe_quadric_curve;
    idx->pow_hessian_elements = size_coe_quadric_curve * size_coe_quadric_curve;
    
    minimum_require = size_coe_quadric_curve;
    while(!(minimum_require % tpb == 0) || counter < OPTIONAL_PARAM::MAX_DIVISOR)
    {
        minimum_require++;
        if(minimum_require % tpb == 0)
        {
            counter++;
        }
    }

    idx->sample_size_for_fitting = minimum_require;
    idx->newton_search_sigma = OPTIONAL_PARAM::SBNEWTON_VARIANCE;
 
    idx->cooling_rate = OPTIONAL_PARAM::COOLING_RATE;
    idx->zeta = OPTIONAL_PARAM::BARIIER_ZETA;
    idx->rho = OPTIONAL_PARAM::BARIIER_RHO;
    idx->lambda_gain = OPTIONAL_PARAM::LAMBDA_GAIN;
    idx->barrier_max = OPTIONAL_PARAM::BARIIER_MAX;
    idx->barrier_tau = OPTIONAL_PARAM::BARIIER_TAU;

    idx->golden_search_iteration = OPTIONAL_PARAM::NUM_OF_GOLDEN_SEARCH_ITERATION;
    idx->golden_ratio = 0.38196601f;
}

void SetupStructure(SampleInfo *info, int num, IndexStructure *idx)
{
    for(int i = 0; i < num; i++)
    {
        info[i].cost = 0.0;
        info[i].weight = 0.0;
        info[i].input = idx->input_by_horizon;
        info[i].dev_state = idx->dim_of_state;
        info[i].dev_input = idx->dim_of_input;
        info[i].dev_ref = idx->dim_of_reference;
        info[i].dev_dstate = idx->dim_of_state;
    }
    printf("Success ----- Set up SampleInfo Structure!!! -----\n");
}

void SetupStructureCMA(SampleInfoCMA *cinfo, IndexCMA *c_idx, IndexStructure *idx)
{
    for(int i = 0; i < c_idx->sample_size_cma; i++)
    {
        cinfo[i].cost = 0.0;
        cinfo[i].weight = 0.0;
        cinfo[i].input = idx->input_by_horizon;
        cinfo[i].dy = idx->input_by_horizon;
        cinfo[i].dev_state = idx->dim_of_state;
        cinfo[i].dev_input = idx->dim_of_input;
        cinfo[i].dev_ref = idx->dim_of_reference;
        cinfo[i].dev_dstate = idx->dim_of_state;
    }
}

void SetupIndicesCMA(IndexCMA *c_idx, IndexStructure *idx)
{
    float l_rate_zeta;
    float sqrt_input_by_horizon;
    sqrt_input_by_horizon = sqrt(idx->input_by_horizon);
    l_rate_zeta = 4.0 / (idx->input_by_horizon + 4);
// #ifdef CMA_DEFAULT
    int temp_sample_size;
    temp_sample_size = (int)(3.0 * log(idx->input_by_horizon));

    c_idx->cma_xi = OPTIONAL_PARAM::CMA_XI;
    c_idx->sample_size_cma =  (4 + temp_sample_size);
    c_idx->elite_sample_cma = (int)((4 + temp_sample_size) / 2);
    c_idx->learning_rate_zeta = l_rate_zeta;
    c_idx->learning_rate_c = l_rate_zeta;
    c_idx->update_rate_top = sqrt(l_rate_zeta*(2-l_rate_zeta));
    c_idx->update_rate_mu = 1.0; //調べて修正
    c_idx->damping_ratio = (1/l_rate_zeta) + 1;
    c_idx->cma_chi = sqrt_input_by_horizon *(1 - ( 1 / ( 4 * idx->input_by_horizon ) ) + ( 1 / ( 21 * powf(idx->input_by_horizon,2) ) ) );
// #else
//     c_idx->sample_size_cma = OPTIONAL_PARAM::SAMPLE_SIZE_CMA;
//     c_idx->elite_sample_cma = OPTIONAL_PARAM::ELITE_SAMPLE_CMA;
//     c_idx->cma_xi = OPTIONAL_PARAM::CMA_XI;
//     c_idx->learning_rate_zeta = OPTIONAL_PARAM::LEARNING_RATE_Z;
//     c_idx->learning_rate_c = OPTIONAL_PARAM::LEARNING_RATE_C;
//     c_idx->update_rate_top = OPTIONAL_PARAM::PATH_UPDATE_RATE_Z;
//     c_idx->update_rate_mu = OPTIONAL_PARAM::PATH_UPDATE_RATE_C;
//     c_idx->damping_ratio = OPTIONAL_PARAM::DAMPING_COEFFICIENT;
//     c_idx->cma_chi = sqrt_input_by_horizon *(1 - ( 1 / ( 4 * hst_idx->input_by_horizon ) ) + ( 1 / ( 21 * powf(hst_idx->input_by_horizon) ) ) );
// #endif
}