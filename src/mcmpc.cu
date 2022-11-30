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

#include "../include/mcmpc.cuh"
#include "../include/cuda_check_error.cuh"
#include "../include/parallel_simulation.cuh"

// デフォルトコンストラクタ
mcmpc::mcmpc()
{
    time_steps = 0; // 時刻の初期化
    cooling_method = NOTHING; // デフォルトの冷却方式（冷却なし）
    ref_type = TIME_INVARIANT; // デフォルト
    solver_type = QR_DECOM; // デフォルト

    cumsum_cost = 0.0f;

    /***** Setup IndexStructure Structure *****/ 
    hst_idx = (IndexStructure*)malloc(sizeof(IndexStructure));
    SetupIndicesSampleBasedNewton(hst_idx);
    CHECK( cudaMalloc(&dev_idx, sizeof(IndexStructure)) );
    CHECK( cudaMemcpy(dev_idx, hst_idx, sizeof(IndexStructure), cudaMemcpyHostToDevice) );

    control_cycle = hst_idx->control_cycle;

    /***** Set GPU parameter *****/
    num_random_seed = hst_idx->sample_size * (hst_idx->dim_of_input + 1) * hst_idx->horizon;
    num_blocks = CountBlocks(hst_idx->sample_size, hst_idx->thread_per_block);
    printf("##### Number of Blocks == [%d] ##### \n", num_blocks);
    thread_per_block = hst_idx->thread_per_block; 
    CHECK( cudaMalloc((void**)&dev_random_seed, num_random_seed * sizeof(curandState)) );
    

    CHECK( cudaMallocManaged((void**)&_state, sizeof(float) * hst_idx->dim_of_state) );
    CHECK( cudaMallocManaged((void**)&_ref, sizeof(float) * hst_idx->dim_of_reference) );
    CHECK( cudaMallocManaged((void**)&_param, sizeof(float) * hst_idx->dim_of_parameter) );
    CHECK( cudaMallocManaged((void**)&_cnstrnt, sizeof(float) * hst_idx->dim_of_constraints) );
    CHECK( cudaMallocManaged((void**)&_weight, sizeof(float) * hst_idx->dim_of_weight_matrix) );

    sample = new SampleInfo[hst_idx->sample_size + 1];
    SetupStructure(sample, hst_idx->sample_size + 1, hst_idx);

    thrust::host_vector<int> indices_hst_vec_temp( hst_idx->sample_size );
    indices_dev_vec = indices_hst_vec_temp;
    thrust::host_vector<float> sort_key_hst_vec_temp( hst_idx->sample_size, 0.0f );
    sort_key_dev_vec = sort_key_hst_vec_temp;
    thrust::host_vector<float> cumsum_weight_hst_vec_temp(hst_idx->elite_sample_size, 0.0f);
    cumsum_weight_dev_vec = cumsum_weight_hst_vec_temp;
    cumsum_weight_hst_vec = cumsum_weight_dev_vec;
    weight_dev_vec = cumsum_weight_hst_vec_temp;
    
    CHECK( cudaMallocManaged((void**)&mcmpc_input_sequences, sizeof(float) * hst_idx->input_by_horizon) );

    time_t time_value;
    struct tm *time_object;
    time(&time_value);
    time_object = localtime( &time_value );
    char filename_state[128], filename_input[128], filename_cost[128]; 
    sprintf(filename_state, "./output/data_state_%d%d_%d%d.txt", time_object->tm_mon + 1, time_object->tm_mday, time_object->tm_hour, time_object->tm_min);
    sprintf(filename_input, "./output/data_input_%d%d_%d%d.txt", time_object->tm_mon + 1, time_object->tm_mday, time_object->tm_hour, time_object->tm_min);
    sprintf(filename_cost, "./output/data_cost_%d%d_%d%d.txt", time_object->tm_mon + 1, time_object->tm_mday, time_object->tm_hour, time_object->tm_min);
    fp_state = fopen(filename_state, "w");
    fp_input = fopen(filename_input, "w");
    fp_cost = fopen(filename_cost, "w");

    // D論用　消して問題なし
    char filename_iteration[128];
    sprintf(filename_iteration, "./output/data_iteration_%d%d_%d%d.txt", time_object->tm_mon + 1, time_object->tm_mday, time_object->tm_hour, time_object->tm_min);
    fp_iteration = fopen(filename_iteration, "w");
    CHECK(cudaMallocManaged((void**)&_optimal, sizeof(float) * hst_idx->input_by_horizon));

    // SetRandomSeed<<<hst_idx->sample_size, (hst_idx->dim_of_input + 1) * hst_idx->horizon>>>(dev_random_seed, rand());
    SetRandomSeed<<<hst_idx->sample_size, (hst_idx->dim_of_input + 1) * hst_idx->horizon>>>(dev_random_seed, (int) rand() / (time_object->tm_min + 1));

    CHECK( cudaDeviceSynchronize() );
    
}

mcmpc::~mcmpc()
{
    FreeAllCudaArrayInBaseClass();
    free(hst_idx);
    fclose(fp_input);
    fclose(fp_state);
    fclose(fp_cost);
    printf("No Error\n");
}

void mcmpc::FreeAllCudaArrayInBaseClass()
{
    CHECK(cudaFree(dev_idx));
    CHECK(cudaFree(dev_random_seed));
    CHECK(cudaFree(_state));
    CHECK(cudaFree(_ref));
    CHECK(cudaFree(_param));
    CHECK(cudaFree(_cnstrnt));
    CHECK(cudaFree(_weight));
    CHECK(cudaFree(mcmpc_input_sequences));

    // D論用　消して問題なし
    CHECK(cudaFree(_optimal));
    fclose(fp_iteration);
}

void mcmpc::Set(float *a, ValueType type)
{
    int index = 0;
    switch(type)
    {
        case SET_STATE:
            for(int i = 0; i < hst_idx->dim_of_state; i++)
            {
                _state[i] = a[i];
            }
            break;
        case SET_INPUT:
            for(int t = 0; t < hst_idx->horizon; t++)
            {
                for(int i = 0; i < hst_idx->dim_of_input; i++)
                {
                    mcmpc_input_sequences[index] = a[i];
                    index++;
                }
            }
            break;
        case SET_PARAMETER:
            for(int i = 0; i < hst_idx->dim_of_parameter; i++)
            {
                _param[i] = a[i];
            }
            break;
        case SET_CONSTRAINT:
            for(int i = 0; i < hst_idx->dim_of_constraints; i++)
            {
                _cnstrnt[i] = a[i];
            }
            break;
        case SET_WEIGHT_MATRIX:
            for(int i = 0; i < hst_idx->dim_of_weight_matrix; i++)
            {
                _weight[i] = a[i];
            }
            break;
        case SET_REFERENCE:
            switch(ref_type)
            {
                case TIME_INVARIANT:
                    for(int t = 0; t < hst_idx->horizon; t++)
                    {
                        for(int i = 0; i < hst_idx->dim_of_reference; i++)
                        {
                            _ref[index] = a[i];
                            index++;
                        }
                    }
                    break;
                case TIME_VARIANT:
                    for(int i = 0; i < hst_idx->horizon * hst_idx->dim_of_reference; i++)
                    {
                        _ref[i] = a[i];
                    }
                    break;
                default:
                    break;
            }
            break;
        default:
            break;
    }
}

void mcmpc::Set(CoolingMethod method, ValueType type)
{
    if(type == SET_COOLING_METHOD && method != cooling_method) cooling_method = method;
}

void mcmpc::Set(ReferenceType method, ValueType type)
{
    if(type == SET_REFERENCE_TYPE && ref_type != method) ref_type = method;
}

void mcmpc::Set(StepWidthDecisiveMethod method, ValueType type)
{
    if(type == SET_STEP_WIDTH_ADJUSTING_METHOD) line_search = method;
}

void mcmpc::Set(LinearEquationSolver method, ValueType type)
{
    if(type == SET_SOLVER) solver_type = method;
}

void mcmpc::ExecuteMPC(float *current_input)
{
    clock_t start_t, stop_t;
    start_t = clock();

    MonteCarloSimulation();
    
    stop_t = clock();
    all_time = stop_t - start_t;
    cost_value_mcmpc = GetCostValue(mcmpc_input_sequences, _state, _param, _ref, _cnstrnt, _weight, hst_idx);
    printf("TIME STEP [ %f ] ******* cost value [ %f ]\n", time_steps * hst_idx->control_cycle, cost_value_mcmpc);   
    printf("PROCESSING TIME OF MCMPC [ %f s] \n", all_time / CLOCKS_PER_SEC);

    for(int i = 0; i < hst_idx->dim_of_input; i++)
    {
        current_input[i] = mcmpc_input_sequences[i];
    }
    printf("----- Ended %d -th control & optimization loop -----\n", time_steps);
    cumsum_cost += cost_value_mcmpc / hst_idx->horizon;
    time_steps++;
}

void mcmpc::MonteCarloSimulation()
{
    float var;
    printf("----- Start %d -th control & optimization loop -----\n", time_steps);
    for(int iter = 0; iter < hst_idx->monte_calro_iteration; iter++)
    {
        switch(cooling_method)
        {
            case GEOMETRIC:
                var = hst_idx->sigma * pow(hst_idx->cooling_rate, iter);
                break;
            case HYPERBOLIC:
                var = hst_idx->sigma / sqrt(iter + 1);
                break;
            case NOTHING:
                var = hst_idx->sigma;
                break;
            default:
                var = hst_idx->sigma;
                break;
        }

        // 並列モンテカルロシミュレーション
        ParallelMonteCarloSimulation<<<num_blocks, thread_per_block>>>(sample, thrust::raw_pointer_cast(sort_key_dev_vec.data()), thrust::raw_pointer_cast(indices_dev_vec.data()),
                                                                       var, _state, _param, _ref, _cnstrnt, _weight, mcmpc_input_sequences, dev_random_seed, dev_idx);
        CHECK( cudaDeviceSynchronize() );

        // 評価値の大小によるThrustを用いたソート
        thrust::sort_by_key(sort_key_dev_vec.begin(), sort_key_dev_vec.end(), indices_dev_vec.begin());
        // ソートの結果を利用して，"エリートサンプル”の重みを取得
        GetWeightFromEliteSample<<<num_blocks, thread_per_block>>>(sample, thrust::raw_pointer_cast(weight_dev_vec.data()), dev_idx, thrust::raw_pointer_cast(indices_dev_vec.data()));
        CHECK( cudaDeviceSynchronize() );

        // Thrustライブラリ関数を用いて、累積和を計算する
        thrust::inclusive_scan(weight_dev_vec.begin(), weight_dev_vec.end(), cumsum_weight_dev_vec.begin());

        // cumsum_weight_hst_vec = cumsum_weight_dev_vec;
        GetWeightedAverageInParallel<<<hst_idx->dim_of_input, hst_idx->horizon>>>(mcmpc_input_sequences, sample, thrust::raw_pointer_cast(weight_dev_vec.data()),
                                                                                 thrust::raw_pointer_cast(cumsum_weight_dev_vec.data()), thrust::raw_pointer_cast(indices_dev_vec.data()), dev_idx);
        CHECK( cudaDeviceSynchronize() );
        WriteIterationResult(iter);        
    }

}

void mcmpc::ExecuteForwardSimulation(float *state, float *input, IntegralMethod method)
{
    float *d_state;
    switch(method)
    {
        case EULAR:
            d_state = (float *)malloc(sizeof(float) * hst_idx->dim_of_state);
            DynamicalModel(d_state, state, input, _param);
            EularIntegration(state, d_state, hst_idx->control_cycle, hst_idx->dim_of_state);
            free(d_state);
            break;
        case RUNGE_KUTTA_45:
            RungeKutta45(state, hst_idx->dim_of_state, input, _param, hst_idx->control_cycle);
            break;
        default:
            d_state = (float *)malloc(sizeof(float) * hst_idx->dim_of_state);
            DynamicalModel(d_state, state, input, _param);
            EularIntegration(state, d_state, hst_idx->control_cycle, hst_idx->dim_of_state);
            free(d_state);
            break;
    }

}

// Virtual Function
void mcmpc::WriteDataToFile( )
{
    float current_time = time_steps * hst_idx->control_cycle;
    for(int i = 0; i < hst_idx->dim_of_state; i++)
    {
        if(i == 0)
        {
            fprintf(fp_state, "%f %f ", current_time, _state[i]);
        }else if(i == hst_idx->dim_of_state - 1){
            fprintf(fp_state, "%f\n", _state[i]);
        }else{
            fprintf(fp_state, "%f ", _state[i]);
        }
    }

    fprintf(fp_cost, "%f %f %f %f\n", current_time, cost_value_mcmpc, cumsum_cost, all_time / CLOCKS_PER_SEC);

    for(int i = 0; i < hst_idx->dim_of_input; i++)
    {
        if(i == 0)
        {
            if(!(hst_idx->dim_of_input - 1) == 0)
            {
                fprintf(fp_input, "%f %f ", current_time, mcmpc_input_sequences[i]);
            }else{
                fprintf(fp_input, "%f %f\n", current_time, mcmpc_input_sequences[i]);
            }
        }else if(i == hst_idx->dim_of_input - 1){
            fprintf(fp_input, "%f\n", mcmpc_input_sequences[i]);
        }else{
            fprintf(fp_input, "%f ", mcmpc_input_sequences[i]);
        }
    }
}

void mcmpc::WriteDataToFile(float *_input)
{
    float current_time = time_steps * hst_idx->control_cycle;
    for(int i = 0; i < hst_idx->dim_of_state; i++)
    {
        if(i == 0)
        {
            fprintf(fp_state, "%f %f ", current_time, _state[i]);
        }else if(i == hst_idx->dim_of_state - 1){
            fprintf(fp_state, "%f\n", _state[i]);
        }else{
            fprintf(fp_state, "%f ", _state[i]);
        }
    }

    fprintf(fp_cost, "%f %f %f %f\n", current_time, cost_value_mcmpc, cumsum_cost, all_time / CLOCKS_PER_SEC);

    for(int i = 0; i < hst_idx->dim_of_input; i++)
    {
        if(i == 0)
        {
            if(!(hst_idx->dim_of_input - 1) == 0)
            {
                fprintf(fp_input, "%f %f ", current_time, _input[i]);
            }else{
                fprintf(fp_input, "%f %f\n", current_time, _input[i]);
            }
        }else if(i == hst_idx->dim_of_input - 1){
            fprintf(fp_input, "%f\n", _input[i]);
        }else{
            fprintf(fp_input, "%f ", _input[i]);
        }
    }
}


// D論用　消して問題なし
void mcmpc::SetOptimal(float *opt)
{
    for(int i = 0; i < hst_idx->input_by_horizon; i++)
    {
        _optimal[i] = opt[i];
    }
}
void mcmpc::WriteIterationResult(int iterations)
{
    float Error = 0.0f;
    fprintf(fp_iteration,"%d ", iterations + 1);
    for(int i = 0; i < hst_idx->input_by_horizon - 1; i++)
    {
        Error = _optimal[i] - mcmpc_input_sequences[i];
        fprintf(fp_iteration,"%f ", Error);
    }
    Error = _optimal[hst_idx->input_by_horizon - 1] - mcmpc_input_sequences[hst_idx->input_by_horizon - 1];
    fprintf(fp_iteration,"%f\n", Error);
}

void mcmpc::WriteIterationLast(int iteration, float *input)
{
    float Error = 0.0f;
    fprintf(fp_iteration,"%d ", iteration + 1);
    for(int i = 0; i < hst_idx->input_by_horizon - 1; i++)
    {
        Error = input[i];
        fprintf(fp_iteration,"%f ", Error);
    }
    Error = input[hst_idx->input_by_horizon-1];
    fprintf(fp_iteration,"%f\n", Error);
}