/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      src/parallel_simulation.cu
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#include "../include/parallel_simulation.cuh"

unsigned int CountBlocks(unsigned int a, unsigned int b)
{
    unsigned int num;
	num = a / b;
	if (a < b || a % b > 0)
		num++;
	return num;
}

__device__ float GenerateRadomInput(unsigned int id, curandState *seed, float mean, float variance)
{
    float ret_value;
    curandState local_seed;
    local_seed = seed[id];
    ret_value = curand_normal(&local_seed) * variance + mean;
    return ret_value;
}

__device__ void GenerateInputCMA(float *input, float *dy, unsigned int id, curandState *seed, float *mean, float *sqrtV, IndexStructure *idx)
{
    int input_leading_id;
    unsigned int shifted_seq_id;
    unsigned int mat_id;
    // int ref_leading_id;
    float temp;
    // if(id == 0)
    // {
    //     for(int row = 0; row < idx->input_by_horizon; row++)
    //     {
    //         for(int col = 0; col < idx->input_by_horizon; col++){
    //             if(col < idx->input_by_horizon){
    //                 printf("%f ", sqrtV[row*idx->input_by_horizon + col]);
    //             }else{
    //                 printf("%f\n", sqrtV[row*idx->input_by_horizon + col]);
    //             }
    //         }
    //     }
    // }
    for(int t = 0; t < idx->horizon; t++)
    {
        input_leading_id = t * idx->dim_of_input;
        for(int input_id = 0; input_id < idx->dim_of_input; input_id++)
        {
            // shifted_seq_id = id + input_id *(idx->horizon * idx->dim_of_input);
            shifted_seq_id = id * (idx->input_by_horizon) + (t * idx->dim_of_input + input_id);
            // dy[input_leading_id + input_id] = GenerateRadomInput(shifted_seq_id, seed, 0.0, 1.0);
            temp = GenerateRadomInput(shifted_seq_id, seed, 0.0, 1.0);
            // if(input_id == idx->dim_of_input - 1) printf("horizon = %d ::: myid = %d && shifted_id = %d  temp == %f\n", t,id, shifted_seq_id, temp);
            for(int sum_id = 0; sum_id < idx->input_by_horizon; sum_id++)
            {
                if(input_leading_id + input_id == 0) 
                {
                    input[sum_id] = mean[sum_id]; // Initialization
                    dy[sum_id] = 0.0f;
                }
                mat_id = sum_id * idx->input_by_horizon + (input_leading_id + input_id);
                // if(id == 0) printf("sum_id = [%d] <===> mat_id = [%d]\n", sum_id, mat_id);
                dy[sum_id] += sqrtV[mat_id] * temp; 
                input[sum_id] += sqrtV[mat_id] * temp;
                // if(id == 0) printf("sum_id = [%d] <===> mat_id = [%d]  input[%d] [@ %f @]  dy[%d] [@ %f @]\n", sum_id, mat_id, sum_id, input[sum_id], sum_id, dy[sum_id]);
                // input[sum_id] += sqrtV[mat_id] * dy[input_leading_id + input_id];
            }
        }
    }
}

__global__ void SetRandomSeed(curandState *random_seed_vec, int seed)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &random_seed_vec[id]);
}

__global__ void GetWeightFromEliteSample(SampleInfo *info, float *weight_vec, IndexStructure *idx, int *indices)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < idx->elite_sample_size)
    {
        float lambda, s_cost;
        lambda = idx->lambda_gain * info[indices[idx->elite_sample_size- 1]].cost;
        // lambda = idx->lambda_gain * info[indices[50]].cost;
        // lambda = 9 / 5;
        // lambda = 50;
        s_cost = info[indices[id]].cost / lambda;
        info[indices[id]].weight = exp(-s_cost);
        if(isnan(exp(-s_cost)) || isinf(exp(-s_cost)))
        {
            weight_vec[id] = 0.0f;
        }else{
            weight_vec[id] = exp(-s_cost);
        }
    }
}

__global__ void GetWeightedAverageInParallel(float *ret_value, SampleInfo *info, float *weight, float *denom_vec, int *indices, IndexStructure *idx)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < idx->horizon * idx->dim_of_input)
    {
        float denominator = denom_vec[idx->elite_sample_size - 1];
        ret_value[id] = 0.0f;
        for(int i = 0; i < idx->elite_sample_size; i++)
        {
            if(isnan(denominator) || isinf(denominator)) break;
            if(isnan(weight[i]) || isinf(weight[i]))
            {
                ret_value[id] += 0.0f;
            }else{
                ret_value[id] += info[indices[i]].input[id] * weight[i] / denominator;
            }
        }
    }
}

float GetCostValue(float *input, float *state, float *param, float *ref, float *cnstrnt, float *weight, IndexStructure *idx)
{
    int input_leading_id, ref_leading_id;
    float stage_cost = 0.0f;
    float total_cost = 0.0f;
    float log_barrier_term = 0.0f;
    float delta_time = idx->prediction_interval / idx->horizon;
    float *simulate_d_state, *simulate_input, *simulate_state, *simulate_ref;
    simulate_d_state = (float *)malloc(sizeof(float) * idx->dim_of_state);
    simulate_state = (float *)malloc(sizeof(float) * idx->dim_of_state);
    simulate_input = (float *)malloc(sizeof(float) * idx->dim_of_input);
    simulate_ref = (float *)malloc(sizeof(float) * idx->dim_of_reference);

    for(int i = 0; i < idx->dim_of_state; i++)
    {
        simulate_state[i] = state[i];
    }
    // 評価値を計算するためのシミュレーションのループ
    for(int t = 0; t < idx->horizon; t++)
    {
        input_leading_id = t * idx->dim_of_input;
        ref_leading_id = t * idx->dim_of_reference;
        for(int input_id = 0; input_id < idx->dim_of_input; input_id++)
        {
            simulate_input[input_id] = input[input_leading_id + input_id];
        }
        for(int ref_id = 0; ref_id < idx->dim_of_reference; ref_id++)
        {
            simulate_ref[ref_id] = ref[ref_leading_id + ref_id];
        }
        InputSaturation(simulate_input, cnstrnt, idx->zeta);
        for(int input_id = 0; input_id < idx->dim_of_input; input_id++)
        {
            input[input_leading_id + input_id] = simulate_input[input_id];
        }
        DynamicalModel(simulate_d_state, simulate_state, simulate_input, param);
        EularIntegration(simulate_state, simulate_d_state, delta_time, idx->dim_of_state);

        log_barrier_term = GetBarrierTerm(simulate_state, simulate_input, cnstrnt, idx->rho);
        stage_cost = GetStageCostTerm(simulate_input, simulate_state, simulate_ref, weight);

        total_cost += stage_cost;
        if(isnan(log_barrier_term) || isinf(log_barrier_term))
        {
            total_cost += idx->barrier_max;
        }else if(total_cost - log_barrier_term < 0){
            total_cost += 1e-2 * idx->rho;
        }else{
            total_cost += idx->barrier_tau * idx->rho * log_barrier_term;
        }
    }
    free(simulate_d_state);
    free(simulate_input);
    free(simulate_state);
    free(simulate_ref);

    return total_cost;
}

void GetCostValueNewton(float &cost, int &check, float *input, float *state, float *param, float *ref, float *cnstrnt, float *weight, IndexStructure *idx)
{
    int input_leading_id, ref_leading_id;
    float stage_cost = 0.0f;
    float total_cost = 0.0f;
    float log_barrier_term = 0.0f;
    float delta_time = idx->prediction_interval / idx->horizon;
    float *simulate_d_state, *simulate_input, *simulate_state, *simulate_ref;
    simulate_d_state = (float *)malloc(sizeof(float) * idx->dim_of_state);
    simulate_state = (float *)malloc(sizeof(float) * idx->dim_of_state);
    simulate_input = (float *)malloc(sizeof(float) * idx->dim_of_input);
    simulate_ref = (float *)malloc(sizeof(float) * idx->dim_of_reference);

    for(int i = 0; i < idx->dim_of_state; i++)
    {
        simulate_state[i] = state[i];
    }

    check = 0;
    // 評価値を計算するためのシミュレーションのループ
    for(int t = 0; t < idx->horizon; t++)
    {
        input_leading_id = t * idx->dim_of_input;
        ref_leading_id = t * idx->dim_of_reference;
        for(int input_id = 0; input_id < idx->dim_of_input; input_id++)
        {
            simulate_input[input_id] = input[input_leading_id + input_id];
        }
        for(int ref_id = 0; ref_id < idx->dim_of_reference; ref_id++)
        {
            simulate_ref[ref_id] = ref[ref_leading_id + ref_id];
        }
        InputSaturation(simulate_input, cnstrnt, idx->zeta);
        for(int input_id = 0; input_id < idx->dim_of_input; input_id++)
        {
            if(input[input_leading_id + input_id] != simulate_input[input_id] && check == 0) check = 1; 
            // input[input_leading_id + input_id] = simulate_input[input_id];
        }
        DynamicalModel(simulate_d_state, simulate_state, simulate_input, param);
        EularIntegration(simulate_state, simulate_d_state, delta_time, idx->dim_of_state);

        log_barrier_term = GetBarrierTerm(simulate_state, simulate_input, cnstrnt, idx->rho);
        stage_cost = GetStageCostTerm(simulate_input, simulate_state, simulate_ref, weight);

        total_cost += stage_cost;
        if(isnan(log_barrier_term) || isinf(log_barrier_term))
        {
            total_cost += idx->barrier_max;
        }else if(total_cost - log_barrier_term < 0){
            total_cost += 1e-2 * idx->rho;
        }else{
            total_cost += idx->barrier_tau * idx->rho * log_barrier_term;
        }
    }
    free(simulate_d_state);
    free(simulate_input);
    free(simulate_state);
    free(simulate_ref);

    cost = total_cost;
}

__global__ void ParallelMonteCarloSimulation(SampleInfo *info, float *cost_vec, int *indices, float var, float *state, float *param, float *ref, float *cnstrnt, float *weight, float *mean, curandState *seed, IndexStructure *idx)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < idx->sample_size)
    {
        unsigned int seq = id;
        int input_leading_id;
        int ref_leading_id;
        unsigned int shifted_seq_id;

        float stage_cost = 0.0f;
        float total_cost = 0.0f;
        float log_barrier_term = 0.0f;
        float delta_time = idx->prediction_interval / idx->horizon;

        for(int i = 0; i < idx->dim_of_state; i++)
        {
            info[id].dev_state[i] = state[i];
        }

        // Monte Carlo Simulation Start
        for(int t = 0; t < idx->horizon; t++)
        {
            input_leading_id = t * idx->dim_of_input;
            ref_leading_id = t * idx->dim_of_reference;

            for(int input_id = 0; input_id < idx->dim_of_input; input_id++)
            {
                if(isnan(mean[input_leading_id + input_id]) || isinf(mean[input_leading_id + input_id]) )
                {
                    info[id].dev_input[input_id] = 0.0f;
                }else{
                    info[id].dev_input[input_id] = mean[input_leading_id + input_id];
                }
                shifted_seq_id = seq + input_id *(idx->horizon * idx->dim_of_input);
                info[id].dev_input[input_id] = GenerateRadomInput(shifted_seq_id, seed, info[id].dev_input[input_id], var);
            }
            seq += idx->sample_size;
            for(int ref_id = 0; ref_id < idx->dim_of_reference; ref_id++)
            {
                info[id].dev_ref[ref_id] = ref[ref_leading_id + ref_id];
            }
            InputSaturation(info[id].dev_input.d_pointer(), cnstrnt, idx->zeta);
            // if(isnan(info[id].dev_dstate[0]) || isnan(info[id].dev_dstate[1]) || isnan(info[id].dev_dstate[2]) || isnan(info[id].dev_dstate[3]) || isnan(info[id].dev_input[0]))
            // {
            //     printf("id = %d input[0]::%f mean[0]::%f\n", id, info[id].dev_input[0], mean[input_leading_id]);
            //     printf("id = %d, dstate[0]::%f dstate[1]::%f dstate[2]::%f dstate[3]::%f\n", id, info[id].dev_dstate[0], info[id].dev_dstate[1], info[id].dev_dstate[2], info[id].dev_dstate[3]);
            // }
            DynamicalModel(info[id].dev_dstate.d_pointer(), info[id].dev_state.d_pointer(), info[id].dev_input.d_pointer(), param);
            EularIntegration(info[id].dev_state.d_pointer(), info[id].dev_dstate.d_pointer(), delta_time, idx->dim_of_state);

            log_barrier_term = GetBarrierTerm(info[id].dev_state.d_pointer(), info[id].dev_input.d_pointer(), cnstrnt, idx->rho);
            stage_cost = GetStageCostTerm(info[id].dev_input.d_pointer(), info[id].dev_state.d_pointer(), info[id].dev_ref.d_pointer(), weight);

            for(int input_id = 0; input_id < idx->dim_of_input; input_id++)
            {
                info[id].input[input_leading_id + input_id] = info[id].dev_input[input_id];
            }

            total_cost += stage_cost;
            if(isnan(log_barrier_term))
            {
                total_cost += idx->barrier_max;
            }else if(total_cost - log_barrier_term < 0){
                total_cost += 1e-2 * idx->rho;
            }else{
                total_cost += idx->barrier_tau * idx->rho * log_barrier_term;
            }
        }
        info[id].cost = total_cost;
        cost_vec[id] = total_cost;
        indices[id] = id;
    }
}

__global__ void ParallelSimulationCMA(SampleInfoCMA *cinfo, float *cost_vec, int *indices, float *state, float *param, float *ref, float *cnstrnt, float *weight, float *mean, float *sqrtVar, curandState *seed, IndexStructure *idx, IndexCMA *cidx)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < cidx->sample_size_cma)
    {
        unsigned int seq = id;
        int input_leading_id;
        int ref_leading_id;
        // unsigned int shifted_seq_id;

        float stage_cost = 0.0f;
        float total_cost = 0.0f;
        float log_barrier_term = 0.0f;
        float delta_time = idx->prediction_interval / idx->horizon;

        for(int i = 0; i < idx->dim_of_state; i++)
        {
            cinfo[id].dev_state[i] = state[i];
        }
        // if(id == 0) printf("called generated input function in 0 thread\n");
        GenerateInputCMA(cinfo[id].input.d_pointer(), cinfo[id].dy.d_pointer(), seq, seed, mean, sqrtVar, idx);
        // __syncthreads( );
        // if(id==0){
        //     for(int i = 0; i < idx->input_by_horizon; i++){
        //         if(id%5==0) printf("Generated input[%d] == %f  <==> dy[%d] == %f\n", i, cinfo[id].input[i], i, cinfo[id].dy[i]);
        //     }
        // }
        
        for(int t = 0; t < idx->horizon; t++)
        {
            input_leading_id = t * idx->dim_of_input;
            ref_leading_id = t * idx->dim_of_reference;

            for(int input_id = 0; input_id < idx->dim_of_input; input_id++)
            {
                cinfo[id].dev_input[input_id] = cinfo[id].input[input_leading_id + input_id];
            }
            // 入力飽和処理
            InputSaturation(cinfo[id].dev_input.d_pointer(), cnstrnt, idx->zeta);
            // 1予測ステップ分の予測シミュレーション
            DynamicalModel(cinfo[id].dev_dstate.d_pointer(), cinfo[id].dev_state.d_pointer(), cinfo[id].dev_input.d_pointer(), param);
            EularIntegration(cinfo[id].dev_state.d_pointer(), cinfo[id].dev_dstate.d_pointer(), delta_time, idx->dim_of_state);
            for(int ref_id = 0; ref_id < idx->dim_of_reference; ref_id++)
            {
                cinfo[id].dev_ref[ref_id] = ref[ref_leading_id + ref_id];
            }
            log_barrier_term = GetBarrierTerm(cinfo[id].dev_state.d_pointer(), cinfo[id].dev_input.d_pointer(), cnstrnt, idx->rho);
            stage_cost = GetStageCostTerm(cinfo[id].dev_input.d_pointer(), cinfo[id].dev_state.d_pointer(), cinfo[id].dev_ref.d_pointer(), weight);
            for(int input_id = 0; input_id < idx->dim_of_input; input_id++)
            {
                cinfo[id].input[input_leading_id + input_id] = cinfo[id].dev_input[input_id];
            }
            total_cost += stage_cost;
            if(isnan(log_barrier_term))
            {
                total_cost += idx->barrier_max;
            }else if(total_cost - log_barrier_term < 0){
                total_cost += 1e-2 * idx->rho;
            }else{
                total_cost += idx->barrier_tau * idx->rho * log_barrier_term;
            }
        }
        cinfo[id].cost = total_cost;
        cost_vec[id] = total_cost;
        indices[id] = id;
    }
}