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

#include "../include/cuda_check_error.cuh"
#include "../include/parallel_simulation.cuh"
#include "../include/sample_based_newton.cuh"
#include "../include/newton_fitting.cuh"
#include "../include/golden_section_search.cuh"

// コンストラクタ
sample_based_newton_method::sample_based_newton_method()
{
    // 推定入力時系列保存用配列
    CHECK( cudaMallocManaged((void**)&sbnewton_input_sequences, sizeof(float) * hst_idx->input_by_horizon) );
    // ヘシアンと勾配計算用
    CHECK( cudaMalloc(&hessian, sizeof(float) * hst_idx->size_of_hessian) );
    CHECK( cudaMalloc(&gradient, sizeof(float) * hst_idx->input_by_horizon) );

    // QR分解に使用する行列のための配列（最小二乗法、ヘシアンの逆行列の計算で使用）
    CHECK( cudaMallocManaged((void**)&coe_matrix, sizeof(float) * hst_idx->pow_hessian_elements) );
    CHECK( cudaMallocManaged((void**)&tensort_x, sizeof(float) * hst_idx->sample_size_for_fitting * hst_idx->size_of_quadrtic_curve) );
    CHECK( cudaMallocManaged((void**)&tensort_l, sizeof(float) * hst_idx->size_of_quadrtic_curve) );
    CHECK( cudaMallocManaged((void**)&b_vector, sizeof(float) * hst_idx->sample_size_for_fitting) );

    // Set up cublas & cusolver variables
    CHECK_CUSOLVER( cusolverDnCreate(&cusolverH), "Failed to Create cusolver handler" );
    CHECK_CUBLAS( cublasCreate(&cublasH), "Failed to create cublas handler" );
    row_x = hst_idx->sample_size_for_fitting;
    colunm_x = hst_idx->size_of_quadrtic_curve;
    row_c = hst_idx->size_of_quadrtic_curve;
    row_h = hst_idx->input_by_horizon;
    alpha = 1.0f;
    beta = 0.0f;
    m_alpha = -1.0;
    CHECK( cudaMalloc((void**)&cu_info, sizeof(int)) );
    CHECK( cudaMalloc((void**)&qr_tau, sizeof(float) * hst_idx->size_of_quadrtic_curve) );
    CHECK( cudaMalloc((void**)&hqr_tau, sizeof(float) * hst_idx->input_by_horizon) );
    uplo = CUBLAS_FILL_MODE_LOWER;
    side = CUBLAS_SIDE_LEFT;
    trans = CUBLAS_OP_T;
    trans_no = CUBLAS_OP_N;
    uplo_qr = CUBLAS_FILL_MODE_UPPER;
    cub_diag = CUBLAS_DIAG_NON_UNIT;

    geqrf_work_size = 0;
    ormqr_work_size = 0;
    qr_work_size = 0;
    hqr_work_size = 0;
    nrhs = 1;

    // printf("Before Generate COLDEN_SECTION_SEARCH_CLASS\n");
    // golden_section_search temp_gss;
    // gss = &temp_gss;
    // printf("After Generate COLDEN_SECTION_SEARCH_CLASS\n");
    newton_h_indices_vec = indices_dev_vec;
    block_size_qc_regression = CountBlocks(hst_idx->sample_size_for_fitting, hst_idx->thread_per_block);

    golden_section_search_flag = 0;
    mcmpc::line_search = NOT_SETTING;
}

sample_based_newton_method::~sample_based_newton_method()
{
    FreeAllCudaArrayInSBNewton();
}

void sample_based_newton_method::FreeAllCudaArrayInSBNewton()
{
    if(cusolverH) cusolverDnDestroy(cusolverH);
    if(cublasH) cublasDestroy(cublasH);
    CHECK( cudaFree( sbnewton_input_sequences ) );
    CHECK( cudaFree( hessian ) );
    CHECK( cudaFree( gradient ) );
    CHECK( cudaFree( coe_matrix ) );
    CHECK( cudaFree( tensort_x ) );
    CHECK( cudaFree( tensort_l ) );
    CHECK( cudaFree( b_vector ) );
    CHECK( cudaFree( qr_tau ) );
    CHECK( cudaFree( hqr_tau ) );
    CHECK( cudaFree( cu_info ) );
}

void sample_based_newton_method::ExecuteMPC(float *current_input)
{
    clock_t start_t, stop_t;
    start_t = clock();
    // Execute Monte Carlo Simulation
    printf("----- Start %d -th control & optimization loop -----\n", time_steps);
    mcmpc::MonteCarloSimulation();

    cost_value_mcmpc = GetCostValue(mcmpc_input_sequences, _state, _param, _ref, _cnstrnt, _weight, hst_idx);

    printf("----- Start Sample-based Newton Step Calculation -----\n");
    printf("----- Cost Value of MCMPC == %f -----\n", cost_value_mcmpc);

    float newton_variance;
    newton_variance = hst_idx->newton_search_sigma;
    
    ParallelMonteCarloSimulation<<<num_blocks, thread_per_block>>>(sample, thrust::raw_pointer_cast(sort_key_dev_vec.data()), thrust::raw_pointer_cast(indices_dev_vec.data()),
                                                                       newton_variance, _state, _param, _ref, _cnstrnt, _weight, mcmpc_input_sequences, dev_random_seed, dev_idx);
    CHECK( cudaDeviceSynchronize() );
    thrust::sort_by_key(sort_key_dev_vec.begin(), sort_key_dev_vec.end(), indices_dev_vec.begin());

    GetTensortMatrices<<<block_size_qc_regression, thread_per_block>>>(tensort_x, b_vector, mcmpc_input_sequences, cost_value_mcmpc, sample, thrust::raw_pointer_cast(indices_dev_vec.data()),
                                                                      dev_idx);
    CHECK( cudaDeviceSynchronize( ) );

    // Compute coe_matrix() = tensort_x^T * tensort_x 
    CHECK_CUBLAS( cublasSgemm(cublasH, trans_no, trans, colunm_x, colunm_x, row_x, &alpha, tensort_x, colunm_x, tensort_x, colunm_x, &beta, coe_matrix, row_c), "Failed to cublasDgemm for [coe_matrix]" );
    // CHECK( cudaDeviceSynchronize());
    // sprintf(mat_name, "C");
    // printMatrix(row_c, row_c, coe_matrix, row_c, mat_name);
    // Compute tensort_l = transpose(tensort_x) * b_vector
    CHECK_CUBLAS(cublasSgemm(cublasH, trans_no, trans, colunm_x, 1, row_x, &alpha, tensort_x, colunm_x, b_vector, 1, &beta, tensort_l, row_c), "Failed to cublasSgemm for [tensort_l]" );
    // CHECK(cudaDeviceSynchronize());

    if(time_steps == 0)
    {
        CHECK_CUSOLVER( cusolverDnSgeqrf_bufferSize(cusolverH, row_c, row_c, coe_matrix, row_c, &geqrf_work_size), "Failed to get buffersize of [coe_matrix] (1st step)" );
        CHECK_CUSOLVER( cusolverDnSormqr_bufferSize(cusolverH, side, trans, row_c, nrhs, row_c, coe_matrix, row_c, qr_tau, tensort_l, row_c, &ormqr_work_size), "Failed to get buffersize of [coe_matrix] (2nd step)" );
        qr_work_size = (geqrf_work_size > ormqr_work_size)? geqrf_work_size : ormqr_work_size;
        CHECK( cudaMalloc((void**)&ws_qr_ops, sizeof(float) * qr_work_size) );
    }
    // Execute QR decomposition for Least Square method ==>  Q = lower triangular of coe_matrix
    CHECK_CUSOLVER( cusolverDnSgeqrf(cusolverH, row_c, row_c, coe_matrix, row_c, qr_tau, ws_qr_ops, qr_work_size, cu_info), "Failed to compute QR factorization" );

    // Compute transpose(Q) * B for compute Ans = inv(R) * transpose(Q) * B by QR decomposition
    CHECK_CUSOLVER( cusolverDnSormqr(cusolverH, side, trans, row_c, nrhs, row_c, coe_matrix, row_c, qr_tau, tensort_l, row_c, ws_qr_ops, qr_work_size, cu_info), "Failed to compute Q^T*B" );

    // Compute vector Ans (store Hessian elements & gradient element & constant C) = inv(R) * transpose(Q) * B
    CHECK_CUBLAS( cublasStrsm(cublasH, side, uplo_qr, trans_no, cub_diag, row_c, nrhs, &alpha, coe_matrix, row_c, tensort_l, row_c), "Failed to compute X = R^-1Q^T*B" );

    GetHessinaAndGradient<<<hst_idx->input_by_horizon, hst_idx->input_by_horizon>>>(hessian, gradient, tensort_l, dev_idx);
    CHECK( cudaDeviceSynchronize( ) );

    if(time_steps == 0)
    {
        CHECK_CUSOLVER( cusolverDnSgeqrf_bufferSize(cusolverH, row_h, row_h, hessian, row_h, &geqrf_work_size), "Failed to get buffersize of [Hessian] (1st step)" );
        CHECK_CUSOLVER( cusolverDnSormqr_bufferSize(cusolverH, side, trans, row_h, nrhs, row_h, hessian, row_h, hqr_tau, gradient, row_h, &ormqr_work_size), "Failed to get buffersize for QR decom [2]" );
        hqr_work_size = (geqrf_work_size > ormqr_work_size)? geqrf_work_size : ormqr_work_size;
        CHECK( cudaMalloc(&ws_hqr_ops, sizeof(float) * hqr_work_size) );
    }
    // Execute QR decomposition to get inv(Hessian)  ==>  Q = lower triangular of coe_matrix
    CHECK_CUSOLVER( cusolverDnSgeqrf(cusolverH, row_h, row_h, hessian, row_h, hqr_tau, ws_hqr_ops, hqr_work_size, cu_info),"Failed to compute QR factorization of Hessain" );

    // Compute transpose(Q) * B for compute Ans (inv(Hessian) * Gradient) = inv(R) * transpose(Q) * B by QR decomposition
    CHECK_CUSOLVER( cusolverDnSormqr(cusolverH, side, trans, row_h, nrhs, row_h, hessian, row_h, hqr_tau, gradient, row_h, ws_hqr_ops, hqr_work_size, cu_info), "Failed to compute Q^T*B of Hessian" )

    // Compute estimated input sequences = inv(R) * transpose(Q) * B
    CHECK_CUBLAS( cublasStrsm(cublasH, side, uplo_qr, trans_no, cub_diag, row_h, nrhs, &m_alpha, hessian, row_h, gradient, row_h), "Failed to compute X = R^-1Q^T*B" );

    ComputeNewtonStep<<<hst_idx->input_by_horizon, 1>>>(sbnewton_input_sequences, mcmpc_input_sequences, gradient);
    CHECK( cudaDeviceSynchronize() );

    cost_value_newton = GetCostValue(sbnewton_input_sequences, _state, _param, _ref, _cnstrnt, _weight, hst_idx);

    printf("----- End Sample-based Newton Step Calculation -----\n");
    printf("----- Cost value of Sample-based Newton method == %f ----\n",cost_value_newton);
    
    cost_value_newton_after_gss = cost_value_newton;
    if(line_search == GOLDEN_SECTION)
    {
        golden_section_search::ExeGoldenSectionSearch( cost_value_newton_after_gss, cost_value_mcmpc, sbnewton_input_sequences, mcmpc_input_sequences, sample, 
                                                    thrust::raw_pointer_cast(newton_h_indices_vec.data()), _state, _param, _ref, _cnstrnt, _weight);
        printf("----- Cost value of Sample-based Newton method after golden search == %f ----\n",cost_value_newton_after_gss);
    }
    
    
    // printf("----- Cost value of Sample-based Newton method after golden search == %f ----\n",cost_value_newton);
    stop_t = clock();
    all_time = stop_t - start_t;
    printf("*** Computation time of Sample-based Newton method = [%f] ***\n", all_time / CLOCKS_PER_SEC);
    SelectOptimalSolution( current_input );
    time_steps++;
}

void sample_based_newton_method::SelectOptimalSolution( float *current_input )
{
    if(cost_value_newton <= cost_value_mcmpc || cost_value_newton_after_gss <= cost_value_mcmpc){
        // if(cost_value_newton_after_gss == cost_value_mcmpc) cost_value_newton_after_gss -= 1e-4;
        golden_section_search_flag = 1;
        SetInputSequences<<<hst_idx->input_by_horizon, 1>>>(mcmpc_input_sequences, sbnewton_input_sequences);
        CHECK( cudaDeviceSynchronize() );
        for(int i = 0; i < hst_idx->dim_of_input; i++)
        {
            current_input[i] = mcmpc_input_sequences[i];
        }
        printf("*** Newton method superior than MCMPC ***\n");
        cumsum_cost += cost_value_newton_after_gss / hst_idx->horizon;
    }else{
        golden_section_search_flag = 0;
        for(int i = 0; i < hst_idx->dim_of_input; i++)
        {
            current_input[i] = mcmpc_input_sequences[i];
        }
        printf("*** SB-Newton method inferior than MCMPC ***\n");
        cumsum_cost += cost_value_mcmpc / hst_idx->horizon;
    }
}

void sample_based_newton_method::WriteDataToFile( )
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

    if(line_search == GOLDEN_SECTION)
    {
        if(cost_value_mcmpc < cost_value_newton_after_gss) fprintf(fp_cost, "%f %f %f %f %f %f %d %f\n", current_time, cost_value_mcmpc, cumsum_cost, cost_value_mcmpc, cost_value_newton, cost_value_newton_after_gss, golden_section_search_flag, all_time / CLOCKS_PER_SEC);
        if(cost_value_newton_after_gss <= cost_value_mcmpc) fprintf(fp_cost, "%f %f %f %f %f %f %d %f\n", current_time, cost_value_newton_after_gss, cumsum_cost, cost_value_mcmpc, cost_value_newton, cost_value_newton_after_gss, golden_section_search_flag, all_time / CLOCKS_PER_SEC);
    }else{
        if(cost_value_mcmpc < cost_value_newton_after_gss) fprintf(fp_cost, "%f %f %f %f %f %f %d %f\n", current_time, cost_value_mcmpc, cumsum_cost, cost_value_mcmpc, cost_value_newton, cost_value_newton_after_gss, golden_section_search_flag, all_time / CLOCKS_PER_SEC);
        if(cost_value_newton_after_gss <= cost_value_mcmpc) fprintf(fp_cost, "%f %f %f %f %f %f %d %f\n", current_time, cost_value_newton_after_gss, cumsum_cost, cost_value_mcmpc, cost_value_newton, cost_value_newton_after_gss, golden_section_search_flag, all_time / CLOCKS_PER_SEC);
    }

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

void sample_based_newton_method::WriteDataToFile(float *_input)
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

    if(line_search == GOLDEN_SECTION)
    {
        if(cost_value_mcmpc < cost_value_newton_after_gss) fprintf(fp_cost, "%f %f %f %f %f %f %d %f\n", current_time, cost_value_mcmpc, cumsum_cost, cost_value_mcmpc, cost_value_newton, cost_value_newton_after_gss, golden_section_search_flag, all_time / CLOCKS_PER_SEC);
        if(cost_value_newton_after_gss <= cost_value_mcmpc) fprintf(fp_cost, "%f %f %f %f %f %f %d %f\n", current_time, cost_value_newton_after_gss, cumsum_cost, cost_value_mcmpc, cost_value_newton, cost_value_newton_after_gss, golden_section_search_flag, all_time / CLOCKS_PER_SEC);
    }else{
        if(cost_value_mcmpc < cost_value_newton_after_gss) fprintf(fp_cost, "%f %f %f %f %f %f %d %f\n", current_time, cost_value_mcmpc, cumsum_cost, cost_value_mcmpc, cost_value_newton, cost_value_newton_after_gss, golden_section_search_flag, all_time / CLOCKS_PER_SEC);
        if(cost_value_newton_after_gss <= cost_value_mcmpc) fprintf(fp_cost, "%f %f %f %f %f %f %d %f\n", current_time, cost_value_newton_after_gss, cumsum_cost, cost_value_mcmpc, cost_value_newton, cost_value_newton_after_gss, golden_section_search_flag, all_time / CLOCKS_PER_SEC);
    }
    

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

// void sample_based_newton_method::printMatrix(int m, int n, float *A, int lda, const char* name)
// {
//     FILE *temp_mat_file;
//     char f_m_n[128];
//     sprintf(f_m_n, "./output/matrix_%s_time_step_%d.txt", name, time_steps);
//     temp_mat_file = fopen(f_m_n, "w");
//     for(int row = 0 ; row < m ; row++){
//         for(int col = 0 ; col < n ; col++){
//             double Areg = A[row + col*lda];
//             printf("%s(%d,%d) = %lf\n", name, row+1, col+1, Areg);
//             //printf("%s[%d] = %lf\n", name, row + col*lda, Areg);
//             if(col == n-1) fprintf(temp_mat_file, "%lf\n", Areg);
//             if(col < n -1) fprintf(temp_mat_file, "%lf ", Areg);
//         }
//     }
//     // sleep(2);
//     fclose(temp_mat_file);
// }