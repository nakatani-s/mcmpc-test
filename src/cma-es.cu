/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      src/cma-es.cu
    [author]    Shintaro Nakatani
    [date]      2022.12.6
*/

#include "../include/cuda_check_error.cuh"
#include "../include/parallel_simulation.cuh"
#include "../include/cma-es.cuh"
// #include ""

// コンストラクター
cma_mpc::cma_mpc()
{
    // デフォルトパラメータの設定
    ref_type = TIME_INVARIANT;
    //　タイム情報の取得
    time_t time_value;
    struct tm *time_object;
    time(&time_value);
    time_object = localtime( &time_value );
    
    // データ出力用ファイルの定義
    char f_name[128];
    sprintf(f_name, "./output/data_state_%d%d_%d%d.txt", time_object->tm_mon + 1, time_object->tm_mday, time_object->tm_hour, time_object->tm_min);
    fp_state = fopen(f_name, "w");
    sprintf(f_name, "./output/data_input_%d%d_%d%d.txt", time_object->tm_mon + 1, time_object->tm_mday, time_object->tm_hour, time_object->tm_min);
    fp_input = fopen(f_name, "w");
    sprintf(f_name, "./output/data_cost_%d%d_%d%d.txt", time_object->tm_mon + 1, time_object->tm_mday, time_object->tm_hour, time_object->tm_min);
    fp_cost = fopen(f_name ,"w");

    // MPCおよびCMA-ESのための各種インデックス（サイズ・パラメータ含む）を取得
    hst_idx = (IndexStructure*)malloc(sizeof(IndexStructure));
    SetupIndicesSampleBasedNewton(hst_idx);
    CHECK( cudaMalloc(&dev_idx, sizeof(IndexStructure)) );
    CHECK( cudaMemcpy(dev_idx, hst_idx, sizeof(IndexStructure), cudaMemcpyHostToDevice) );
    cma_idx = (IndexCMA*)malloc(sizeof(IndexCMA));
    SetupIndicesCMA(cma_idx, hst_idx);
    // cmaで必要なサンプル数（100次元でも18サンプルとかに）なので、最低限thread_per_block(=10)以上のサンプルは利用するようにする（global関数の呼び出し実行を容易にするため）
    if(cma_idx->sample_size_cma < hst_idx->thread_per_block) cma_idx->sample_size_cma = hst_idx->thread_per_block;
    CHECK( cudaMalloc(&dev_cma_idx, sizeof(IndexCMA)) );
    CHECK( cudaMemcpy(dev_cma_idx, cma_idx, sizeof(IndexCMA), cudaMemcpyHostToDevice) );

    cma_xi = cma_idx->cma_xi;
    // Array (or Vector) For CMA-ES Algorithm
    CHECK( cudaMallocManaged((void**)&cma_es_input_sequences, sizeof(float) * hst_idx->input_by_horizon) );
    CHECK( cudaMallocManaged(&cma_es_dy, sizeof(float) * hst_idx->input_by_horizon) );
    // CHECK( cudaMalloc(&tensort_y_vector, sizeof(float) * cma_idx->sample_size_cma * hst_idx->input_by_horizon) );
    // CHECK( cudaMalloc(&tensort_Pc, sizeof(float) * hst_idx->size_of_hessian) );
    // CHECK( cudaMalloc(&tensort_y, sizeof(float) * hst_idx->size_of_hessian) );
    CHECK( cudaMallocManaged((void**)&path_zeta, sizeof(float) * hst_idx->input_by_horizon) );
    CHECK( cudaMalloc(&path_c, sizeof(float) * hst_idx->input_by_horizon) );

    SetupVector<<<hst_idx->input_by_horizon,1>>>(path_zeta, 0.0f);
    CHECK( cudaDeviceSynchronize() );
    SetupVector<<<hst_idx->input_by_horizon,1>>>(path_c, 0.0f);
    CHECK( cudaDeviceSynchronize() );

    CHECK( cudaMallocManaged(&Variance, sizeof(float) * hst_idx->size_of_hessian) );
    CHECK( cudaMalloc(&sqrtVariance, sizeof(float) * hst_idx->size_of_hessian) );
    CHECK( cudaMalloc(&inv_sqrtVar, sizeof(float) * hst_idx->size_of_hessian) );
    CHECK( cudaMalloc(&gradient, sizeof(float) * hst_idx->input_by_horizon) );
    CHECK( cudaMallocManaged(&eigen_value_vec, sizeof(float) * hst_idx->input_by_horizon) );

    // 行列演算用
    CHECK( cudaMalloc(&orth_matrix, sizeof(float) * hst_idx->size_of_hessian) );


    // SetupIdentityMatrixCMA<<<hst_idx->input_by_horizon, hst_idx->input_by_horizon>>>(Variance, hst_idx->input_by_horizon, hst_idx->input_by_horizon);
    SetupDiagMatrixCMA<<<hst_idx->input_by_horizon, hst_idx->input_by_horizon>>>(Variance, hst_idx->input_by_horizon, hst_idx->input_by_horizon, pow(cma_xi,2));
    CHECK(cudaDeviceSynchronize());
    // SetupIdentityMatrixCMA<<<hst_idx->input_by_horizon, hst_idx->input_by_horizon>>>(sqrtVariance, hst_idx->input_by_horizon, hst_idx->input_by_horizon);
    SetupDiagMatrixCMA<<<hst_idx->input_by_horizon, hst_idx->input_by_horizon>>>(sqrtVariance, hst_idx->input_by_horizon, hst_idx->input_by_horizon, cma_xi);
    CHECK(cudaDeviceSynchronize());
    SetupDiagMatrixCMA<<<hst_idx->input_by_horizon, hst_idx->input_by_horizon>>>(inv_sqrtVar, hst_idx->input_by_horizon, hst_idx->input_by_horizon, 1/cma_xi);
    CHECK(cudaDeviceSynchronize());

    // MPC関連の変数の定義
    CHECK( cudaMallocManaged((void**)&_state, sizeof(float) * hst_idx->dim_of_state) );
    CHECK( cudaMallocManaged((void**)&_ref, sizeof(float) * hst_idx->dim_of_reference) );
    CHECK( cudaMallocManaged((void**)&_param, sizeof(float) * hst_idx->dim_of_parameter) );
    CHECK( cudaMallocManaged((void**)&_cnstrnt, sizeof(float) * hst_idx->dim_of_constraints) );
    CHECK( cudaMallocManaged((void**)&_weight, sizeof(float) * hst_idx->dim_of_weight_matrix) );

    // Set up GPU parameters
    num_random_seed = cma_idx->sample_size_cma * (hst_idx->dim_of_input + 1) * hst_idx->horizon;
    num_blocks = CountBlocks(cma_idx->sample_size_cma, hst_idx->thread_per_block);
    printf("##### Number of Blocks == [%d] ##### \n", num_blocks);
    printf("##### Sample size == [%d] #####\n", cma_idx->sample_size_cma);
    thread_per_block = hst_idx->thread_per_block;
    CHECK(cudaMalloc((void**)&dev_random_seed, num_random_seed * sizeof(curandState)));
    SetRandomSeed<<<cma_idx->sample_size_cma, (hst_idx->dim_of_input + 1) * hst_idx->horizon>>>(dev_random_seed, (int) rand() / (time_object->tm_min + 1));
    CHECK( cudaDeviceSynchronize() );

    // cublas & cusolver setup
    CHECK_CUSOLVER( cusolverDnCreate(&cusolverH), "Failed to Create cusolver handler" );
    CHECK_CUBLAS( cublasCreate(&cublasH), "Failed to create cublas handler" );
    uplo = CUBLAS_FILL_MODE_LOWER;
    jobz = CUSOLVER_EIG_MODE_VECTOR;
    side = CUBLAS_SIDE_LEFT;
    trans = CUBLAS_OP_T;
    trans_no = CUBLAS_OP_N;
    uplo_qr = CUBLAS_FILL_MODE_UPPER;
    cub_diag = CUBLAS_DIAG_NON_UNIT;

    row_v = hst_idx->input_by_horizon;
    column_v = hst_idx->input_by_horizon;
    geqrf_work_size = 0;
    ormqr_work_size = 0;
    hqr_work_size = 0;
    hsvd_work_size = 0;

    nrhs = 1;

    CHECK( cudaMalloc((void**)&qr_tau, sizeof(float) * hst_idx->size_of_hessian) );
    CHECK( cudaMallocManaged((void**)&cu_info, sizeof(int)) );
    
    alpha = 1.0f;
    beta = 0.0f;
    minus = -1.0f;

    c_sample = new SampleInfoCMA[cma_idx->sample_size_cma + 1];
    SetupStructureCMA(c_sample, cma_idx, hst_idx);
 
    thrust::host_vector<int> indices_hst_vec_temp( cma_idx->sample_size_cma );
    indices_d_vec_cma = indices_hst_vec_temp;
    thrust::host_vector<float> sort_key_host_vec_tmep(cma_idx->sample_size_cma, 0.0f);
    sort_key_d_vec_cma = sort_key_host_vec_tmep;
    thrust::host_vector<float> cumsum_weight_hst_vec_temp(cma_idx->elite_sample_cma, 0.0f);
    cumsum_weight_d_vec_cma = cumsum_weight_hst_vec_temp;
    cumsum_weight_h_vec_cma = cumsum_weight_d_vec_cma;
    weight_d_vec_cma = cumsum_weight_hst_vec_temp;
    pow_weight_d_vec_cma = cumsum_weight_hst_vec_temp;
    cumsum_pow_weight_h_vec = weight_d_vec_cma;
    cumsum_pow_weight_d_vec = cumsum_pow_weight_h_vec;

}
// デストラクタ‐
cma_mpc::~cma_mpc()
{
    cmaesFreeArray();
    printf("Called Destroctor of CMA-ES type optimization algorithms !!!\n");
}

void cma_mpc::cmaesFreeArray()
{
    if(cusolverH) cusolverDnDestroy(cusolverH);
    if(cublasH) cublasDestroy(cublasH);
    CHECK( cudaFree(dev_idx) );
    CHECK( cudaFree(dev_cma_idx) );
    CHECK( cudaFree(dev_random_seed) );
    CHECK( cudaFree(cma_es_input_sequences) );
    CHECK( cudaFree(Variance) );
    CHECK( cudaFree(gradient) );
    CHECK( cudaFree(sqrtVariance) );
    CHECK( cudaFree(inv_sqrtVar) );
    CHECK( cudaFree(eigen_value_vec) );
    CHECK( cudaFree(path_zeta) );
    CHECK( cudaFree(path_c) );
    CHECK(cudaFree(_state));
    CHECK(cudaFree(_ref));
    CHECK(cudaFree(_param));
    CHECK(cudaFree(_cnstrnt));
    CHECK(cudaFree(_weight));
}

void cma_mpc::ExecuteMPC(float *current_input)
{
    // clock_t start_t, stop_t;
    start_t = clock();
    // Execute monte carlo simulation
    printf("----- Start %d -th control & optimization loop -----\n", time_steps);
    CovarianceMatrixAdaptationSampling();
    stop_t = clock();
    all_time = stop_t - start_t;
    cost_value = GetCostValue(cma_es_input_sequences, _state, _param, _ref, _cnstrnt, _weight, hst_idx);
    printf("PROCESSING TIME OF CMA-MCMPC [ %f s] \n", all_time / CLOCKS_PER_SEC);

    for(int i = 0; i < hst_idx->dim_of_input; i++)
    {
        current_input[i] = cma_es_input_sequences[i];
    }
    printf("----- Ended %d -th control & optimization loop -----\n", time_steps);
    // ↓は削除しても問題ない
    cumsum_cost += cost_value / hst_idx->horizon;
    time_steps++;
    
}

void cma_mpc::CovarianceMatrixAdaptationSampling()
{
    for(int iter = 0; iter < hst_idx->monte_calro_iteration; iter++)
    {
        ParallelSimulationCMA<<<num_blocks, thread_per_block>>>(c_sample, thrust::raw_pointer_cast(sort_key_d_vec_cma.data()), thrust::raw_pointer_cast(indices_d_vec_cma.data()),
                                                                 _state, _param, _ref, _cnstrnt, _weight, cma_es_input_sequences, sqrtVariance, dev_random_seed, dev_idx, dev_cma_idx);
        CHECK( cudaDeviceSynchronize() );
        
        // 評価値の大小によるThrustを用いたソート
        thrust::sort_by_key(sort_key_d_vec_cma.begin(), sort_key_d_vec_cma.end(), indices_d_vec_cma.begin());
        WeightCalculationCMA<<<cma_idx->elite_sample_cma, 1>>>(c_sample, thrust::raw_pointer_cast(weight_d_vec_cma.data()), thrust::raw_pointer_cast(pow_weight_d_vec_cma.data()),
                                                                dev_cma_idx, dev_idx, thrust::raw_pointer_cast(indices_d_vec_cma.data()));
        CHECK(cudaDeviceSynchronize());
        // 正規化項の計算（W, W^2の２種類を計算）
        thrust::inclusive_scan(weight_d_vec_cma.begin(), weight_d_vec_cma.end(), cumsum_weight_d_vec_cma.begin());
        cumsum_weight_h_vec_cma = cumsum_weight_d_vec_cma;
        denom_weight = cumsum_weight_h_vec_cma[cma_idx->elite_sample_cma - 1];
        thrust::inclusive_scan(pow_weight_d_vec_cma.begin(), pow_weight_d_vec_cma.end(), cumsum_pow_weight_d_vec.begin());
        cumsum_pow_weight_h_vec = cumsum_pow_weight_d_vec;
        denom_pow_weight = cumsum_pow_weight_h_vec[cma_idx->elite_sample_cma - 1];
        GetWeigthedMeanCMA<<<hst_idx->horizon, hst_idx->dim_of_input>>>(cma_es_input_sequences, cma_es_dy, denom_weight, c_sample, thrust::raw_pointer_cast(indices_d_vec_cma.data()), dev_cma_idx, dev_idx);
        CHECK( cudaDeviceSynchronize() );
        printf("estimation %d, %d step == %f\n", time_steps, iter, cma_es_input_sequences[0]);
        // c_{mu}の更新 更新式は、[Hansen et al.,14] eq.(51)に記載
        // dev_cma_idx は廃止を検討し、cudaMallocManagedで確保してもよいかも
        cma_idx->update_rate_mu = (1/denom_pow_weight)*(1/(cma_idx->elite_sample_cma*cma_idx->elite_sample_cma));
        CHECK( cudaMemcpy(dev_cma_idx, cma_idx, sizeof(IndexCMA), cudaMemcpyHostToDevice) );

        if(iter < hst_idx->monte_calro_iteration - 1)
        {
            // 進化パス P_zetaの更新
            alpha_zeta = cma_idx->learning_rate_zeta * (2 - cma_idx->learning_rate_zeta) * (1/denom_pow_weight);
            alpha_zeta = sqrt(alpha_zeta);
            beta_zeta = (1-cma_idx->learning_rate_zeta);
            CHECK_CUBLAS( cublasSgemv(cublasH, trans_no, row_v, column_v, &alpha_zeta, inv_sqrtVar, row_v, cma_es_dy, 1, &beta_zeta, path_zeta, 1), "Failed to update path_{zeta}" );
            // 進化パス P_cの更新
            CHECK( cudaDeviceSynchronize() );
            printf("cma_dy == \n");
            MatrixPrintf(cma_es_dy, 1 ,hst_idx->input_by_horizon);
            printf("path_zeta == \n");
            MatrixPrintf(path_zeta, 1 ,hst_idx->input_by_horizon);
            alpha_c = cma_idx->learning_rate_c * (2 - cma_idx->learning_rate_c) * (1/denom_pow_weight);
            alpha_c = sqrt(alpha_c);
            GetPathCMA<<<hst_idx->input_by_horizon, 1>>>(path_c, 1-cma_idx->learning_rate_c, alpha_c, cma_es_dy, hst_idx->input_by_horizon);
            CHECK(cudaDeviceSynchronize());
            // 分散パラメータzetaの更新
            cma_xi = UpdateVarianceParam();
            // 分散共分散行列の計算
            CovarianceMatrixAdaptation<<<hst_idx->input_by_horizon, hst_idx->input_by_horizon>>>(Variance, c_sample, path_c, dev_cma_idx, dev_idx, thrust::raw_pointer_cast(indices_d_vec_cma.data()));
            CHECK( cudaDeviceSynchronize( ) );
            // MatrixPrintf(Variance, hst_idx->input_by_horizon, hst_idx->input_by_horizon);
            if(time_steps == 0)
            {
                CHECK_CUSOLVER( cusolverDnSgeqrf_bufferSize(cusolverH, row_v, row_v, Variance, row_v, &geqrf_work_size), "Failed to get buffersize of [CovarianceMatrix] <= 1" );
                CHECK_CUSOLVER( cusolverDnSormqr_bufferSize(cusolverH, side, trans, row_v, nrhs, row_v, Variance, row_v, qr_tau, gradient, row_v, &ormqr_work_size), "Failed to get buffersize for QR decom [2]" );
                hqr_work_size = (geqrf_work_size > ormqr_work_size)? geqrf_work_size : ormqr_work_size;
                CHECK( cudaMalloc((void**)&ws_qr_ops, sizeof(float) * hqr_work_size) );
                CHECK_CUSOLVER( cusolverDnSsyevd_bufferSize(cusolverH,jobz, uplo, row_v, Variance, row_v, gradient, &hsvd_work_size), "Faile to get buffersize of [Hessian]<=>1");
                CHECK(cudaMalloc((void**)&ws_svd_ops, sizeof(float) * hsvd_work_size) );
            }
            // 固有値分解(SVD)を実行 (Variance => 固有ベクトルを格納した行列，eigen_value_vec => 固有値を格納したベクトル)
            CHECK_CUSOLVER( cusolverDnSsyevd(cusolverH, jobz, uplo, row_v, Variance, row_v, eigen_value_vec, ws_svd_ops, hsvd_work_size, cu_info), "Failed to decompose singular value of Covariance" );
            // cusolverDnSsyevd(cusolverH, jobz, uplo, row_v, Variance, row_v, eigen_value_vec, ws_svd_ops, hsvd_work_size, cu_info);
            // CHECK( cudaDeviceSynchronize() );
            // printf("Failed to decompose singular value of Covariance due to devInfo = %d\n", cu_info[0]);
            // MatrixPrintf(eigen_value_vec, 1, hst_idx->input_by_horizon);
            // sqrtVariance:=対角行列（各要素は分散共分散行列の固有値を並べたもの）
            SetSqrtEigenValToDiagMatrix<<<hst_idx->input_by_horizon, hst_idx->input_by_horizon>>>(sqrtVariance, eigen_value_vec);
            CHECK( cudaDeviceSynchronize() );
            SetInvSqrtEigenValToDiagMatrix<<<hst_idx->input_by_horizon, hst_idx->input_by_horizon>>>(inv_sqrtVar, eigen_value_vec);
            CHECK( cudaDeviceSynchronize() );
            // orth_matrix = Variance * sqrtVariance (ΛT*Σ)
            CHECK_CUBLAS( cublasSgemm(cublasH, trans_no, trans_no, row_v, row_v, row_v, &alpha, Variance, row_v, sqrtVariance, row_v, &beta, orth_matrix, row_v), "Failed to compute  inverse matrix 1st operation" );
            // sqrtVariance = orth_matrix * Variance^T (ΛT*Σ*Λ)
            CHECK_CUBLAS( cublasSgemm(cublasH, trans_no, trans, row_v, row_v, row_v, &alpha, orth_matrix, row_v, Variance, row_v, &beta, sqrtVariance, row_v), "Failed to compute inv(2.0*Hessian)" );
            MultipliedMatrixCMA<<<hst_idx->input_by_horizon, hst_idx->input_by_horizon>>>(sqrtVariance, cma_xi);
            CHECK( cudaDeviceSynchronize() );
            // 
            CHECK_CUBLAS( cublasSgemm(cublasH, trans_no, trans_no, row_v, row_v, row_v, &alpha, Variance, row_v, inv_sqrtVar, row_v, &beta, orth_matrix, row_v), "Failed to compute  inverse matrix 1st operation" );
            // 
            CHECK_CUBLAS( cublasSgemm(cublasH, trans_no, trans, row_v, row_v, row_v, &alpha, orth_matrix, row_v, Variance, row_v, &beta, inv_sqrtVar, row_v), "Failed to compute inv(2.0*Hessian)" );
        }
        else{
            // 最後のloopで、分散、その他CMAのパラメータを初期化する
            // まずは、分散共分散行列も各予測ステップごとに初期化する
            cma_xi = cma_idx->cma_xi;
            SetupDiagMatrixCMA<<<hst_idx->input_by_horizon, hst_idx->input_by_horizon>>>(sqrtVariance, hst_idx->input_by_horizon, hst_idx->input_by_horizon, cma_idx->cma_xi);
            CHECK(cudaDeviceSynchronize());
            SetupDiagMatrixCMA<<<hst_idx->input_by_horizon, hst_idx->input_by_horizon>>>(Variance, hst_idx->input_by_horizon, hst_idx->input_by_horizon, pow(cma_idx->cma_xi,2));
            CHECK(cudaDeviceSynchronize());
            SetupDiagMatrixCMA<<<hst_idx->input_by_horizon, hst_idx->input_by_horizon>>>(inv_sqrtVar, hst_idx->input_by_horizon, hst_idx->input_by_horizon, 1/cma_idx->cma_xi);
            CHECK(cudaDeviceSynchronize());
            // SetupVector<<<hst_idx->input_by_horizon,1>>>(cma_es_input_sequences, 0.0f);
            // CHECK( cudaDeviceSynchronize() );
            SetupVector<<<hst_idx->input_by_horizon,1>>>(path_zeta, 0.0f);
            CHECK( cudaDeviceSynchronize() );
            SetupVector<<<hst_idx->input_by_horizon,1>>>(path_c, 0.0f);
            CHECK( cudaDeviceSynchronize() );
        }
    }
}

float cma_mpc::UpdateVarianceParam()
{
    float p_zeta_norm;
    float coefficient;
    float temp;
    float ret;
    coefficient = cma_idx->learning_rate_zeta / cma_idx->damping_ratio;
    p_zeta_norm = 0.0f;
    for(int i = 0; i < hst_idx->input_by_horizon; i++)
    {
        p_zeta_norm += path_zeta[i] * path_zeta[i];
    }
    temp = (sqrt(p_zeta_norm) / cma_idx->cma_chi) - 1;
    ret = cma_xi * exp(coefficient * temp);
    printf("Parameter [xi] = %f (%f)\n", ret, p_zeta_norm);
    
    return ret; 
}

void cma_mpc::Set(float *a, ValueType type)
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
                    cma_es_input_sequences[index] = a[i];
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

void cma_mpc::Set(ReferenceType method, ValueType type)
{
    if(type == SET_REFERENCE_TYPE && ref_type != method) ref_type = method;
}

void cma_mpc::WriteDataToFile( )
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

    fprintf(fp_cost, "%f %f %f %f\n", current_time, cost_value, cumsum_cost, all_time / CLOCKS_PER_SEC);

    for(int i = 0; i < hst_idx->dim_of_input; i++)
    {
        if(i == 0)
        {
            if(!(hst_idx->dim_of_input - 1) == 0)
            {
                fprintf(fp_input, "%f %f ", current_time, cma_es_input_sequences[i]);
            }else{
                fprintf(fp_input, "%f %f\n", current_time, cma_es_input_sequences[i]);
            }
        }else if(i == hst_idx->dim_of_input - 1){
            fprintf(fp_input, "%f\n", cma_es_input_sequences[i]);
        }else{
            fprintf(fp_input, "%f ", cma_es_input_sequences[i]);
        }
    }
}

void cma_mpc::WriteDataToFile(float *_input)
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

    fprintf(fp_cost, "%f %f %f %f\n", current_time, cost_value, cumsum_cost, all_time / CLOCKS_PER_SEC);

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


void cma_mpc::ExecuteForwardSimulation(float *state, float *input, IntegralMethod method)
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

void cma_mpc::MatrixPrintf(float *mat, int row, int col)
{
    for(int i = 0; i < row; i++)
    {
        for(int k = 0; k < col; k++)
        {
            if(k < col - 1)
            {
                printf("%f ", mat[i*row + k]);
            }else{
                printf("%f\n", mat[i*row + k]);
            }
        }
    }
}


__global__ void WeightCalculationCMA(SampleInfoCMA *cinfo, float *weight_vec, float *pow_weight_vec, IndexCMA *cidx, IndexStructure *idx, int *indices)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < cidx->elite_sample_cma)
    {
        float lambda, s_cost;
        lambda = idx->lambda_gain * cinfo[indices[cidx->elite_sample_cma - 1]].cost;
        // idに対応する順位のサンプルid(sid)を取得
        unsigned int sid = indices[id];

        s_cost = cinfo[sid].cost / lambda; 
        if(isnan(exp(-s_cost)) || isinf(exp(-s_cost)))
        {
            weight_vec[id] = 0.0f;
            pow_weight_vec[id] = 0.0f;
            cinfo[sid].weight = 0.0f;
        }else{
            weight_vec[id] = exp(-s_cost);
            pow_weight_vec[id] = weight_vec[id] * weight_vec[id];
            cinfo[sid].weight = exp(-s_cost);
        }
    }
}

__global__ void GetWeigthedMeanCMA(float *input_seq, float *dy_seq, float denom, SampleInfoCMA *cinfo, int *indices, IndexCMA *cidx, IndexStructure *idx)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < idx->input_by_horizon)
    {
        input_seq[id] = 0.0f;
        dy_seq[id] = 0.0f;
        for(int i = 0; i < cidx->elite_sample_cma; i++)
        {
            if(isnan(denom) || isinf(denom)) break;
            if(isnan(cinfo[indices[i]].weight) || isinf(cinfo[indices[i]].weight))
            {
                input_seq[id] += 0.0f;
                dy_seq[id] += 0.0f;
            }else{
                input_seq[id] += cinfo[indices[i]].input[id] * cinfo[indices[i]].weight / denom;
                dy_seq[id] += cinfo[indices[i]].dy[id] * cinfo[indices[i]].weight / denom;
            }
        }
    }
}

__global__ void GetPathCMA(float *path, float alpha, float beta, float *dy, int dimention)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < dimention)
    {
        path[id] = alpha * path[id] + beta * dy[id];
    }
}

__global__ void CovarianceMatrixAdaptation(float *Var, SampleInfoCMA *cinfo, float *Pc, IndexCMA *cidx, IndexStructure *idx, int *indices)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    float temp;
    // 行列の要素をtempに一時的に格納
    temp = Var[id];
    float tensort_c;
    tensort_c = cidx->update_rate_top * (Pc[threadIdx.x] * Pc[blockIdx.x] - temp);
    unsigned int sample_id;
    float tensort_y;
    tensort_y = 0.0f;
    for(int i = 0; i < cidx->elite_sample_cma; i++)
    {
        sample_id = indices[i];
        tensort_y += cinfo[sample_id].weight * (cinfo[sample_id].dy[threadIdx.x] * cinfo[sample_id].dy[blockIdx.x] - temp);
    }
    tensort_y = cidx->update_rate_mu * tensort_y;
    Var[id] = temp + tensort_c + tensort_y;

}

__global__ void SetupVector(float *vector, float val)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    vector[id] = val;
}

__global__ void SetupIdentityMatrixCMA(float *matrix, int row, int column)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < row * column)
    {
        if(threadIdx.x == blockIdx.x)
        {
            matrix[id] = 1.0f;
        }else{
            matrix[id] = 0.0f;
        }
    }
}

__global__ void SetupDiagMatrixCMA(float *matrix, int row, int column, float val)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < row * column)
    {
        if(threadIdx.x == blockIdx.x)
        {
            matrix[id] = val;
        }else{
            matrix[id] = 0.0f;
        }
    }
}

__global__ void MultipliedMatrixCMA(float *matrix, float multiplies)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    matrix[id] = multiplies * matrix[id];
}

__global__ void SetSqrtEigenValToDiagMatrix(float *Mat, float *Vec)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadIdx.x == blockIdx.x)
    {
        if(Vec[threadIdx.x] <= 0)
        {
            float temp;
            int vec_id;
            vec_id = 0;
            temp = Vec[threadIdx.x];
            while(temp <= 0)
            {
                vec_id++;
                if(vec_id + threadIdx.x < blockDim.x)
                {
                    temp = Vec[vec_id + threadIdx.x];
                }else{
                    temp = 1.0;
                }
            }
            Mat[id] = sqrt(temp);
        }else{
            Mat[id] = sqrt(Vec[threadIdx.x]);
        }
    }else{
        Mat[id] = 0.0f;
    }
}

__global__ void SetInvSqrtEigenValToDiagMatrix(float *Mat, float *Vec)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadIdx.x == blockIdx.x)
    {
        if(Vec[threadIdx.x] <= 0)
        {
            Vec[threadIdx.x] = 0.001f; 
        }
        Mat[id] = 1 / sqrt(Vec[threadIdx.x]);
    }else{
        Mat[id] = 0.0f;
    }
}
