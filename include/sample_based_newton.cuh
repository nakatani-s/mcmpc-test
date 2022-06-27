/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      include/sample_based_newton.cu
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#ifndef SAMPLE_BASED_NEWTON_CUH
#define SAMPLE_BASED_NEWTON_CUH

#include "common_header.cuh"
#include "mcmpc.cuh"
#include "data_structures.cuh"
#include "golden_section_search.cuh"

class sample_based_newton_method : public mcmpc, public golden_section_search
{
private:
    float *hessian, *gradient;

    // For Using cusolver & cublas Library
    cusolverDnHandle_t cusolverH;
    cublasHandle_t cublasH;
    cublasFillMode_t uplo;
    cublasSideMode_t side;
    cublasOperation_t trans;
    cublasOperation_t trans_no;
    cublasFillMode_t uplo_qr;
    cublasDiagType_t cub_diag;

    float alpha, beta, m_alpha;
    float *ws_qr_ops;
    float *ws_hqr_ops;

    int row_x, colunm_x, row_c, row_h;
    int geqrf_work_size;
    int ormqr_work_size;
    int qr_work_size;
    int hqr_work_size;
    int m_r_matrix;
    int nrhs;
    int *cu_info;
    float *qr_tau;
    float *hqr_tau;

    // 行列演算用
    int block_size_qc_regression;
    float *coe_matrix, *b_vector, *tensort_x, *tensort_l;

public:
    sample_based_newton_method(); // Constructor
    ~sample_based_newton_method(); //Destoructor


    float cost_value_newton, cost_value_newton_after_gss;
    float *sbnewton_input_sequences;

    thrust::host_vector<int> newton_h_indices_vec;

    void ExecuteMPC(float *current_input); // override
    void WriteDataToFile( ); // override
    void WriteDataToFile(float *_input); // override
    void FreeAllCudaArrayInSBNewton();
    void SelectOptimalSolution( float *current_input );

};

#endif