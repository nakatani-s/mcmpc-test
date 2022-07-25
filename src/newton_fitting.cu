/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      src/newton_fitting.cu
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#include "../include/newton_fitting.cuh"

__global__ void SetInputSequences(float *output, float *optima)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    float temp_value;
    temp_value = output[id];
    output[id] = optima[id];
    optima[id] = temp_value;
}

__global__ void GetTensortMatrices(float *matrix, float *vector, float *mean, float cost, SampleInfo *info, int *indices, IndexStructure *idx)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < idx->sample_size_for_fitting)
    {
        unsigned int sample_id = indices[id];
        unsigned int next_indices = 0;
        unsigned int row_idx;
        int forward_id;
        int backward_id;
        row_idx = id * idx->size_of_quadrtic_curve;

        float denominator = idx->lambda_gain * info[indices[idx->size_of_quadrtic_curve - 1]].cost;
        float fitting_weight = exp(-info[sample_id].cost / denominator);
        float f_weight = sqrt(fitting_weight);

        vector[id] = f_weight *  info[sample_id].cost / idx->sample_size_for_fitting;

        for(int i = 0; i < idx->horizon; i++)
        {
            for(int j = 0; j < idx->dim_of_input; j++)
            {
                for(int k = i; k < idx->horizon; k++)
                {
                    if(k==i){
                        for(int h = j; h < idx->dim_of_input; h++)
                        {
                            forward_id = i * idx->dim_of_input + j;
                            backward_id = k * idx->dim_of_input + h;
                            // matrix[row_idx + next_indices] = (info[sample_id].input[forward_id] - mean[forward_id]) * (info[sample_id].input[backward_id] - mean[backward_id]);
                            matrix[row_idx + next_indices] = f_weight * (info[sample_id].input[forward_id] - mean[forward_id]) * (info[sample_id].input[backward_id] - mean[backward_id]);
                            next_indices += 1;
                        }
                    }else{
                        for(int h = 0; h < idx->dim_of_input; h++)
                        {
                            forward_id = i * idx->dim_of_input + j;
                            backward_id = k * idx->dim_of_input + h;
                            // matrix[row_idx + next_indices] = (info[sample_id].input[forward_id] - mean[forward_id]) * (info[sample_id].input[backward_id] - mean[backward_id]);
                            matrix[row_idx + next_indices] = f_weight * (info[sample_id].input[forward_id] - mean[forward_id]) * (info[sample_id].input[backward_id] - mean[backward_id]);
                            next_indices += 1;
                        }
                    }
                }
            }
        }

        for(int i = 0; i < idx->horizon; i++)
        {
            for(int j = 0; j < idx->dim_of_input; j++)
            {
                forward_id = i * idx->dim_of_input + j;
                // matrix[row_idx + next_indices] = (info[sample_id].input[forward_id] - mean[forward_id]);
                matrix[row_idx + next_indices] = f_weight * (info[sample_id].input[forward_id] - mean[forward_id]);
                next_indices += 1;
            }
        }

        // matrix[row_idx + next_indices] = 1.0;
        matrix[row_idx + next_indices] = f_weight;
    }
    
}

__global__ void GetHessinaAndGradient(float *hessian, float *gradient, float *lsm_result_vec, IndexStructure *idx)
{
    unsigned int id  = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int diag_term_id;
    float temp_value;

    int vector_id = blockIdx.x;
    if(threadIdx.x <= blockIdx.x)
    {
        int offset;
        for(int temp_id = 0; temp_id < threadIdx.x; temp_id++)
        {
            offset = temp_id + 1;
            vector_id += (idx->input_by_horizon - offset);
        }
        temp_value = lsm_result_vec[vector_id];
        diag_term_id = blockIdx.x + threadIdx.x * blockDim.x;

        if(threadIdx.x != blockIdx.x)
        {
            if(isnan(temp_value) || isinf(temp_value))
            {
                hessian[id] = 0.0f;
                hessian[diag_term_id] = 0.0f;
            }else{
                hessian[id] = temp_value;
                hessian[diag_term_id] = temp_value;
            }
        }else{
            if(isnan(temp_value) || isinf(temp_value))
            {
                hessian[id] = 1.0f;
            }else{
                hessian[id] = 2.0f * temp_value;
            }
        }
    }
    unsigned int gradient_id = threadIdx.x;
    if(blockIdx.x == blockDim.x - 1)
    {
        gradient[gradient_id] = lsm_result_vec[idx->size_of_hessian_element + gradient_id];
    }
}

__global__ void ComputeNewtonStep(float *ans, float *current_guess, float *newton_step)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    ans[id] = current_guess[id] + newton_step[id];
}