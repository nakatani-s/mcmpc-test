/*
 * This files is part of MCMPC toolkit
 *
 * MCMPC toolkit -- A toolkit for Model Predictive Control
*/ 
/*
    [file] include/cma-es.cuh
    [author] Shintaro Naktani
    [date] 2022.12.6
*/

#ifndef CMA_ES_CUH
#define CMA_ES_CUH

#include "common_header.cuh"
#include "mcmpc.cuh"
#include "data_structures.cuh"


// ホスト・デバイス関数 for CMA
__global__ void SetupVector(float *vector, float val);
__global__ void SetupIdentityMatrixCMA(float *matrix, int row, int column);
__global__ void SetupDiagMatrixCMA(float *matrix, int row, int column, float val);
__global__ void MultipliedMatrixCMA(float *matrix, float multiplies);
__global__ void WeightCalculationCMA(SampleInfoCMA *cinfo, float *weight_vec, float *pow_weight_vec, IndexCMA *cidx, IndexStructure *idx, int *indices);
__global__ void GetWeigthedMeanCMA(float *input_seq, float *dy_seq, float denom, SampleInfoCMA * cinfo, int *indices, IndexCMA *cidx, IndexStructure *idx);
__global__ void GetPathCMA(float *path, float alpha, float beta, float *dy, int dimention); 
__global__ void CovarianceMatrixAdaptation(float *Var, SampleInfoCMA *cinfo, float *Pc, IndexCMA *cidx, IndexStructure *idx, int *indices);
__global__ void SetSqrtEigenValToDiagMatrix(float *Mat, float *Vec);
__global__ void SetInvSqrtEigenValToDiagMatrix(float *Mat, float *Vec);
                                                                 


class cma_mpc
{
private:
    // GPU parameters
    unsigned int num_blocks, num_random_seed, thread_per_block;
    float denom_weight, denom_pow_weight;
    float alpha_zeta, alpha_c;
    float beta_zeta, beta_c;
    float cma_xi;
    float *Variance, *sqrtVariance, *inv_sqrtVar, *gradient;
    float *eigen_value_vec;
    // float *tensort_y_vector;
    // float *tensort_Pc, *tensort_y;
    float *path_zeta, *path_c;
    float *sqrt_eigen_vector, *eigen_vector;

    IndexStructure *hst_idx, *dev_idx;
    IndexCMA *cma_idx, *dev_cma_idx;

    // variables for cublas & cusolver
    cusolverDnHandle_t cusolverH;
    cublasHandle_t cublasH;
    cublasFillMode_t uplo;
    cublasSideMode_t side;
    cublasOperation_t trans;
    cublasOperation_t trans_no;
    cublasFillMode_t uplo_qr;
    cublasDiagType_t cub_diag;

    cusolverEigMode_t jobz;
    cublasFillMode_t uplo_svd;
    
    int row_v, column_v;
    int geqrf_work_size;
    int ormqr_work_size;
    int hqr_work_size;
    int hsvd_work_size;
    int nrhs;
    int *cu_info;
    float *qr_tau;
    float alpha, beta, minus;
    float *ws_qr_ops;
    float *ws_svd_ops;
    float *orth_matrix; 

public:
    FILE *fp_state, *fp_input, *fp_cost;
    
    clock_t start_t, stop_t;
    float all_time;

    ReferenceType ref_type;

    int time_steps;
    float *cma_es_input_sequences;
    float *cma_es_dy;
    float cost_value, cumsum_cost;
    float *_state, *_ref, *_param, *_cnstrnt, *_weight;
    SampleInfoCMA *c_sample;

    // 乱数の種（サンプル数×ホライズン×入力の次元分）
    curandState *dev_random_seed;

    thrust::device_vector<int> indices_d_vec_cma;
    thrust::device_vector<float> sort_key_d_vec_cma;
    thrust::host_vector<float> cumsum_weight_h_vec_cma;
    thrust::device_vector<float> cumsum_weight_d_vec_cma;
    thrust::device_vector<float> weight_d_vec_cma;
    thrust::device_vector<float> pow_weight_d_vec_cma;
    thrust::host_vector<float> cumsum_pow_weight_h_vec;
    thrust::device_vector<float> cumsum_pow_weight_d_vec;

    cma_mpc(); // Constructor
    ~cma_mpc(); // Destoructor
    void ExecuteMPC(float *current_input);
    void Set(float *a, ValueType type);
    // void Set(CoolingMethod method, ValueType type);
    void Set(ReferenceType method, ValueType type);
    // void Set(StepWidthDecisiveMethod method, ValueType type);
    // void Set(LinearEquationSolver method, ValueType type);
    void WriteDataToFile( );
    void WriteDataToFile(float *_input);
    void CovarianceMatrixAdaptationSampling();
    float UpdateVarianceParam();
    void cmaesFreeArray();
    void ExecuteForwardSimulation(float *state, float *input, IntegralMethod method);
};

#endif