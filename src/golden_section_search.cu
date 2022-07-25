/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      src/golden_section_search.cu
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#include "../include/cuda_check_error.cuh"
#include "../include/golden_section_search.cuh"

golden_section_search::golden_section_search( )
{
    gss_h_idx = (IndexStructure*)malloc(sizeof(IndexStructure));
    SetupIndicesSampleBasedNewton(gss_h_idx);
    CHECK(cudaMalloc((void**)&gss_d_idx, sizeof(IndexStructure)));
    CHECK( cudaMemcpy(gss_d_idx, gss_h_idx, sizeof(IndexStructure), cudaMemcpyHostToDevice) );

    g_sample = new GoldenSample[gss_h_idx->elite_sample_size + 1];
    SetupGoldenSample();
    thrust::host_vector<float> gss_sort_key_h_vec_temp(gss_h_idx->elite_sample_size);
    thrust::host_vector<int> gss_indices_h_vec_temp(gss_h_idx->elite_sample_size);
    gss_sort_key_d_vec = gss_sort_key_h_vec_temp;
    gss_indices_d_vec = gss_indices_h_vec_temp;
    gss_indices_h_vec = gss_indices_d_vec;

    gss_input_id = 0;
    gss_id_vec_id = 0;
    CHECK( cudaMallocManaged((void**)&copy_indices, sizeof(int) * gss_h_idx->sample_size ))
    CHECK( cudaMallocManaged((void**)&copy_newton_sequences, sizeof(float) * gss_h_idx->input_by_horizon));
    CHECK(cudaMallocManaged((void**)&copy_mcmpc_sequences, sizeof(float) * gss_h_idx->input_by_horizon));

    printf("Constructor of golden section search Called\n");
}

golden_section_search::~golden_section_search()
{
    CHECK(cudaFree(copy_newton_sequences));
}

golden_section_search::golden_section_search(const golden_section_search& old)
{
    *this = old;
}

golden_section_search& golden_section_search::operator=(const golden_section_search &old)
{
    if(this != &old)
    {
        printf("Why you call Assiment operator!!!\n");
    }
    return *this;
}


void golden_section_search::ExeGoldenSectionSearch( float &cost_value, float &cost_ref, float *newton_input_seq, float *mcmpc_input_seq, SampleInfo *sample, int *indices, float *_state, float *_param, float *_ref, float *_cnstrnt, float *_weight)
{
    gss_input_id = 0;
    gss_id_vec_id = 0;
    printf("@@@@@@@@ golden_section_search called @@@@@@@@@ \n");
    while(gss_input_id < gss_h_idx->input_by_horizon || gss_id_vec_id < gss_h_idx->sample_size)
    {
        if(gss_input_id < gss_h_idx->input_by_horizon){
            copy_newton_sequences[gss_input_id] = newton_input_seq[gss_input_id];
            copy_mcmpc_sequences[gss_input_id] = mcmpc_input_seq[gss_input_id];
            gss_input_id += 1;
        }
        if(gss_id_vec_id < gss_h_idx->sample_size)
        {
            copy_indices[gss_id_vec_id] = indices[gss_id_vec_id];
            gss_id_vec_id += 1;
        }
    }
    InitializeGoldenSearch<<<gss_h_idx->elite_sample_size, gss_h_idx->input_by_horizon>>>(g_sample, sample, copy_newton_sequences, copy_mcmpc_sequences, cost_ref, copy_indices, gss_d_idx);
    CHECK( cudaDeviceSynchronize() );

    // printf("!!!!!!! golden_section_search called  %d !!!!!!!\n", gss_indices_h_vec[hst_idx->elite_sample_size -1]);

    ParallelGoldenSectionSearch<<<gss_h_idx->elite_sample_size, 2>>>(thrust::raw_pointer_cast(gss_sort_key_d_vec.data()), thrust::raw_pointer_cast(gss_indices_d_vec.data()), g_sample, 
                                                                sample, _state, _param, _ref, _cnstrnt, _weight, copy_indices, gss_d_idx);                                                                                         

    CHECK( cudaDeviceSynchronize() );

    // g_sampleの評価値を昇順にソートして、結果のインデックスをgss_indices_h_vecで受ける
    thrust::sort_by_key(gss_sort_key_d_vec.begin(), gss_sort_key_d_vec.end(), gss_indices_d_vec.begin());
    gss_indices_h_vec = gss_indices_d_vec;

    if(g_sample[gss_indices_h_vec[0]].cost_limit < cost_value)
    {
        gss_input_id = 0;
        cost_value = g_sample[gss_indices_h_vec[0]].cost_limit;
        while(gss_input_id < gss_h_idx->input_by_horizon)
        {
            newton_input_seq[gss_input_id] = g_sample[gss_indices_h_vec[0]].input_limit[gss_input_id];
            gss_input_id += 1;
        }
    }

}

void golden_section_search::SetupGoldenSample( )
{
    for(int i = 0; i < gss_h_idx->elite_sample_size; i++)
    {
        g_sample[i].cost_left = 0.0f;
        g_sample[i].cost_right = 0.0f;
        g_sample[i].cost_limit = 0.0f;
        g_sample[i].input_left = gss_h_idx->input_by_horizon;
        g_sample[i].input_right = gss_h_idx->input_by_horizon;
        g_sample[i].input_limit = gss_h_idx->input_by_horizon;
        g_sample[i].dev_state_left = gss_h_idx->dim_of_state;
        g_sample[i].dev_state_right = gss_h_idx->dim_of_state;
        g_sample[i].dev_input_left = gss_h_idx->dim_of_input;
        g_sample[i].dev_input_right = gss_h_idx->dim_of_input;
        g_sample[i].dev_dstate_left = gss_h_idx->dim_of_state;
        g_sample[i].dev_dstate_right = gss_h_idx->dim_of_state;
        g_sample[i].dev_ref_left = gss_h_idx->dim_of_reference;
        g_sample[i].dev_ref_right = gss_h_idx->dim_of_reference;
    }
}

__device__ int CheckBoxConstraintViolation(float *after, float *before, int dim_of_input)
{
    int flag = 0;
    for(int i = 0; i < dim_of_input; i++)
    {
        if(after[i] != before[i])
        {
            flag = 1;
            break;
        }
    }
    return flag;
}

__global__ void InitializeGoldenSearch(GoldenSample *g_info, SampleInfo *info, float *newton_seq, float *mcmpc_seq, float cost_mc, int *indices, IndexStructure *idx)
{
    if(blockIdx.x == 0)
    {
        int worst_id = indices[idx->elite_sample_size - 1];
        info[worst_id].input[threadIdx.x] = mcmpc_seq[threadIdx.x];
        if(threadIdx.x == 0) info[worst_id].cost = cost_mc;
    }
    if(blockIdx.x < idx->elite_sample_size)
    {
        g_info[blockIdx.x].input_limit[threadIdx.x] = newton_seq[threadIdx.x];
    }
}

/*__global__ void CopyFirstGuessInfoToWorstSample(SampleInfo *info, float *mcmpc_seq, int *indices, IndexStructure *idx)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int worst_id = indices[idx->elite_sample_size - 1];
    if(id < idx->input_by_horizon)
    {
        info[worst_id].input[id] = mcmpc_seq[id];
    }   
}
__global__ void CopyLeftBoundaryPoint(GoldenSample *g_sample, float *newton_seq; IndexStructure *idx)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;


}*/

// __global__ void OverwriteInputSequences(float *input_sequences, GoldenSample *g_sample, int *indices, IndexStructure *idx)
// {
//     unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
//     if(id < idx->input_by_horizon)
//     {
//         input_sequences[id] = g_sample[indices[0]].input_limit[id];
//     }
// }

__global__ void ParallelGoldenSectionSearch(float *cost_vec, int *indices, GoldenSample *g_sample, SampleInfo *info, float *state, float *param, float *ref, float *cnstrnt, float *weight, int *s_indices, IndexStructure *idx)
{
    // unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    // if(blockIdx.x < 5) printf("block ID :: %d {g_sample[%d].cost_limit = %f}\n", blockIdx.x, blockIdx.x, g_sample[blockIdx.x].cost_limit);
    int info_id = s_indices[blockIdx.x];
    int input_index = 0;
    float barrier_term;
    float stage_cost;
    float delta_time = idx->prediction_interval / idx->horizon;
    const float ratio = idx->golden_ratio;
    indices[blockIdx.x] = blockIdx.x;
    if( blockIdx.x < idx->elite_sample_size ){
        for(int iter = 0; iter < idx->golden_search_iteration; iter++)
        {
            if(threadIdx.x == 0) g_sample[blockIdx.x].cv_flag = 0;
            __syncthreads( );

            if(threadIdx.x == 0){
                input_index = 0;
                g_sample[blockIdx.x].cost_left = 0.0f;
                for(int i = 0; i < idx->dim_of_state; i++)
                {
                    g_sample[blockIdx.x].dev_state_left[i] = state[i];
                }

                for(int i = 0; i < idx->dim_of_reference; i++)
                {
                    g_sample[blockIdx.x].dev_ref_left[i] = ref[i];
                }

                for(int i = 0; i < idx->horizon; i++)
                {
                    for(int k = 0; k < idx->dim_of_input; k++)
                    {
                        g_sample[blockIdx.x].dev_input_left[k] = (1 - ratio) * g_sample[blockIdx.x].input_limit[input_index] + ratio * info[info_id].input[input_index];
                        g_sample[blockIdx.x].input_left[input_index] =  g_sample[blockIdx.x].dev_input_left[k];
                        input_index += 1;
                    }

                    // 内側（モンテカルロ解側）の評価点がBox制約を違反した時はループを抜ける
                    if(g_sample[blockIdx.x].cv_flag != 0) break;
                    
                    DynamicalModel(g_sample[blockIdx.x].dev_dstate_left.d_pointer(), g_sample[blockIdx.x].dev_state_left.d_pointer(), g_sample[blockIdx.x].dev_input_left.d_pointer(), param);
                    EularIntegration(g_sample[blockIdx.x].dev_state_left.d_pointer(), g_sample[blockIdx.x].dev_dstate_left.d_pointer(), delta_time, idx->dim_of_state);
                    barrier_term = GetBarrierTerm(g_sample[blockIdx.x].dev_state_left.d_pointer(), g_sample[blockIdx.x].dev_input_left.d_pointer(), cnstrnt, idx->rho);
                    stage_cost = GetStageCostTerm(g_sample[blockIdx.x].dev_input_left.d_pointer(), g_sample[blockIdx.x].dev_state_left.d_pointer(), g_sample[blockIdx.x].dev_ref_left.d_pointer(), weight);
                    g_sample[blockIdx.x].cost_left += stage_cost;
                    if(isnan(barrier_term) || isinf(barrier_term))
                    {
                        g_sample[blockIdx.x].cost_left += idx->barrier_max;
                    }else if(g_sample[blockIdx.x].cost_left - barrier_term < 0){
                        g_sample[blockIdx.x].cost_left += 1e-2 * idx->rho;
                    }else{
                        g_sample[blockIdx.x].cost_left += idx->barrier_tau * idx->rho * barrier_term;
                    }
                }
            }else{
                input_index = 0;
                g_sample[blockIdx.x].cost_right = 0.0f;
                for(int i= 0; i < idx->dim_of_state; i++)
                {
                    g_sample[blockIdx.x].dev_state_right[i] = state[i];
                }

                for(int i = 0; i < idx->dim_of_reference; i++)
                {
                    g_sample[blockIdx.x].dev_ref_right[i] = ref[i];
                }

                for(int i = 0; i < idx->horizon; i++)
                {
                    for(int k =0; k < idx->dim_of_input; k++)
                    {
                        g_sample[blockIdx.x].dev_input_right[k] = ratio * g_sample[blockIdx.x].input_limit[input_index] + (1 - ratio) * info[info_id].input[input_index];
                        info[info_id].dev_input[k] = g_sample[blockIdx.x].dev_input_right[k];
                        g_sample[blockIdx.x].input_right[input_index] = g_sample[blockIdx.x].dev_input_right[k];
                        input_index += 1;
                    }
                    InputSaturation(g_sample[blockIdx.x].dev_input_right.d_pointer(), cnstrnt, idx->zeta);
                    g_sample[blockIdx.x].cv_flag = CheckBoxConstraintViolation(g_sample[blockIdx.x].dev_input_right.d_pointer(), info[info_id].dev_input.d_pointer(), idx->dim_of_input);
                    
                    // 内側（モンテカルロ解側）の評価点がBox制約を違反した時はループを抜ける
                    if(g_sample[blockIdx.x].cv_flag != 0) break;

                    DynamicalModel(g_sample[blockIdx.x].dev_dstate_right.d_pointer(), g_sample[blockIdx.x].dev_state_right.d_pointer(), g_sample[blockIdx.x].dev_input_right.d_pointer(), param);
                    EularIntegration(g_sample[blockIdx.x].dev_state_right.d_pointer(), g_sample[blockIdx.x].dev_dstate_right.d_pointer(), delta_time, idx->dim_of_state);
                    barrier_term = GetBarrierTerm(g_sample[blockIdx.x].dev_state_right.d_pointer(), g_sample[blockIdx.x].dev_input_right.d_pointer(), cnstrnt, idx->rho);
                    stage_cost = GetStageCostTerm(g_sample[blockIdx.x].dev_input_right.d_pointer(), g_sample[blockIdx.x].dev_state_right.d_pointer(), g_sample[blockIdx.x].dev_ref_right.d_pointer(), weight);
                    g_sample[blockIdx.x].cost_right += stage_cost;
                    if(isnan(barrier_term) || isinf(barrier_term))
                    {
                        g_sample[blockIdx.x].cost_right += idx->barrier_max;
                    }else if(g_sample[blockIdx.x].cost_right - barrier_term < 0){
                        g_sample[blockIdx.x].cost_right += 1e-2 * idx->rho;
                    }
                    else{
                        g_sample[blockIdx.x].cost_right += idx->barrier_tau * idx->rho * barrier_term;
                    }
                }
            }
            // if(blockIdx.x < 5) printf("block ID :: %d {g_sample[%d].cost_right = %f}\n", blockIdx.x, blockIdx.x, g_sample[blockIdx.x].cost_right);
            __syncthreads();
            if(g_sample[blockIdx.x].cv_flag != 0){
                if(threadIdx.x == 0)
                {
                    input_index = 0;
                    while(input_index < idx->input_by_horizon)
                    {
                        g_sample[blockIdx.x].input_limit[input_index] =  ratio * g_sample[blockIdx.x].input_limit[input_index] + (1 - ratio) * info[info_id].input[input_index];
                        input_index += 1;
                    }
                }
                if(iter == idx->golden_search_iteration - 1) cost_vec[blockIdx.x] = idx->barrier_max;
                continue;
            }
            // 内側（モンテカルロ解側）の評価点がBox制約を満たしている時だけ
            // if(threadIdx.x == 0){
            //     DynamicalModel(g_sample[blockIdx.x].dev_dstate_left.d_pointer(), g_sample[blockIdx.x].dev_state_left.d_pointer(), info[info_id].dev_input_left.d_pointer(), param);
            //     EularIntegration(g_sample[blockIdx.x].dev_state_left.d_pointer(), g_sample[blockIdx.x].dev_dstate_left.d_pointer(), delta_time, idx->dim_of_state);
            //     barrier_term = GetBarrierTerm(g_sample[blockIdx.x].dev_state_left.d_pointer(), g_sample[blockIdx.x].dev_input_left.d_pointer(), cnstrnt, idx->rho);
            //     stage_cost = GetStageCostTerm(g_sample[blockIdx.x].dev_input_left.d_pointer(), info[blockIdx.x].dev_state_left.d_pointer(), info[info_id].dev_ref_left.d_pointer(), weight);
            //     g_sample[blockIdx.x].cost_lest += stage_cost;
            //     if(isnan(barrier_term) || isinf(barrier_term))
            //     {
            //         g_sample[blockIdx.x].cost_lest += idx->barrier_max;
            //     }else{
            //         g_sample[blockIdx.x].cost_left += idx->barrier_tau * idx->rho * barrier_term;
            //     }
            // }else{
            //     DynamicalModel(g_sample[blockIdx.x].dev_dstate_right.d_pointer(), g_sample[blockIdx.x].dev_state_right.d_pointer(), info[info_id].dev_input_right.d_pointer(), param);
            //     EularIntegration(g_sample[blockIdx.x].dev_state_right.d_pointer(), g_sample[blockIdx.x].dev_dstate_right.d_pointer(), delta_time, idx->dim_of_state);
            //     barrier_term = GetBarrierTerm(g_sample[blockIdx.x].dev_state_right.d_pointer(), g_sample[blockIdx.x].dev_input_right.d_pointer(), cnstrnt, idx->rho);
            //     stage_cost = GetStageCostTerm(g_sample[blockIdx.x].dev_input_right.d_pointer(), info[blockIdx.x].dev_state_right.d_pointer(), info[info_id].dev_ref_right.d_pointer(), weight);
            //     g_sample[blockIdx.x].cost_right += stage_cost;
            //     if(isnan(barrier_term) || isinf(barrier_term))
            //     {
            //         g_sample[blockIdx.x].cost_right += idx->barrier_max;
            //     }else{
            //         g_sample[blockIdx.x].cost_right += idx->barrier_tau * idx->rho * barrier_term;
            //     }
            // }
            // __syncthreads();

            // 以下は、Iterationの最後は、評価値が小さい方をg_sample[].input_limitに返す
            // Iterationの最後までのループでは、右側（あるいは左側）の探索範囲を狭める処理
            if(threadIdx.x == 0)
            {
                input_index = 0;
                if(g_sample[blockIdx.x].cost_left > g_sample[blockIdx.x].cost_right)
                {
                    while(input_index < idx->input_by_horizon)
                    {
                        if(iter < idx->golden_search_iteration - 1 ){
                            g_sample[blockIdx.x].input_limit[input_index] = g_sample[blockIdx.x].input_left[input_index];
                        }else{
                            if( g_sample[blockIdx.x].cost_right < info[info_id].cost ) g_sample[blockIdx.x].input_limit[input_index] = g_sample[blockIdx.x].input_right[input_index];
                            if(g_sample[blockIdx.x].cost_right > info[info_id].cost) g_sample[blockIdx.x].input_limit[input_index] = info[info_id].input[input_index];
                            g_sample[blockIdx.x].cost_limit = g_sample[blockIdx.x].cost_right;                             
                        }
                        input_index += 1;
                    }
                    if( g_sample[blockIdx.x].cost_right > info[info_id].cost ){
                        cost_vec[blockIdx.x] = info[info_id].cost;
                        g_sample[blockIdx.x].cost_limit = info[info_id].cost;
                    }else{
                        cost_vec[blockIdx.x] = g_sample[blockIdx.x].cost_right;
                        g_sample[blockIdx.x].cost_limit = g_sample[blockIdx.x].cost_right;
                    }
                }else{
                    while(input_index < idx->input_by_horizon)
                    {
                        if(iter < idx->golden_search_iteration - 1)
                        {
                            if( g_sample[blockIdx.x].cost_right > info[info_id].cost ) g_sample[blockIdx.x].input_limit[input_index] = g_sample[blockIdx.x].input_right[input_index];
                            if( g_sample[blockIdx.x].cost_right <= info[info_id].cost ) info[info_id].input[input_index] = g_sample[blockIdx.x].input_right[input_index];
                        }else{
                            if( g_sample[blockIdx.x].cost_right > info[info_id].cost ) g_sample[blockIdx.x].input_limit[input_index] = info[info_id].input[input_index];
                            if( g_sample[blockIdx.x].cost_right <= info[info_id].cost ) g_sample[blockIdx.x].input_limit[input_index] = g_sample[blockIdx.x].input_left[input_index];                            
                            g_sample[blockIdx.x].cost_limit = g_sample[blockIdx.x].cost_left;
                        }
                        input_index += 1;
                    }
                    if( g_sample[blockIdx.x].cost_right > info[info_id].cost )
                    {
                        cost_vec[blockIdx.x] = info[info_id].cost;
                        g_sample[blockIdx.x].cost_limit = info[info_id].cost;
                    }else{
                        cost_vec[blockIdx.x] = g_sample[blockIdx.x].cost_left;
                        g_sample[blockIdx.x].cost_limit = g_sample[blockIdx.x].cost_left;
                    }
                }
            }
        }
    }
}