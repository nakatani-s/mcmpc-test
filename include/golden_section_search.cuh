/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      include/golden_section_search.cuh
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#ifndef GOLDEN_SECTION_SEARCH_CUH
#define GOLDEN_SECTION_SEARCH_CUH

#include "common_header.cuh"
// #include "managed.cuh"
#include "managed.cuh"
#include "data_structures.cuh"
#include "numerical_integrator.cuh"

struct GoldenSample : public Managed{
    int cv_flag;
    float cost_left, cost_right, cost_limit;
    DynamicArray input_left;
    DynamicArray input_right;
    DynamicArray input_limit;
    DynamicArray dev_state_left;
    DynamicArray dev_state_right;
    DynamicArray dev_input_left;
    DynamicArray dev_input_right;
    DynamicArray dev_dstate_left;
    DynamicArray dev_dstate_right;
    DynamicArray dev_ref_left;
    DynamicArray dev_ref_right;
};

__device__ int CheckBoxConstraintViolation(float *after, float *before, int dim_of_input);

__global__ void InitializeGoldenSearch(GoldenSample *g_info, SampleInfo *info, float *newton_seq, float *mcmpc_seq, float cost_mc, int *indices, IndexStructure *idx);
// __global__ void CopyFirstGuessInfoToWorstSample(SampleInfo *info, float *mcmpc_seq, int *indices, IndexStructure *idx);
// __global__ void CopyLeftBoundaryPoint(GoldenSample *g_sample, float *newton_seq; IndexStructure *idx);
// __global__ void OverwriteInputSequences(float *input_sequences, GoldenSample *g_sample, int *indices, IndexStructure *idx);
__global__ void ParallelGoldenSectionSearch(float *cost_vec, int *indices, GoldenSample *g_sample, SampleInfo *info, float *state, float *param, float *ref, float *cnstrnt, float *weight, int *s_indices, IndexStructure *idx);


class golden_section_search
{
public:
    GoldenSample *g_sample;
    IndexStructure *gss_h_idx, *gss_d_idx;
    thrust::host_vector<int> indices_hst_vec_gss;
    thrust::host_vector<int> gss_indices_h_vec;
    thrust::device_vector<int> gss_indices_d_vec;
    thrust::device_vector<float> gss_sort_key_d_vec;

    int gss_input_id, gss_id_vec_id;
    int *copy_indices;
    float *copy_newton_sequences;
    float *copy_mcmpc_sequences;
    // DEFAULT CONSTRACTOR & DESTRACTOR
    golden_section_search();
    ~golden_section_search();
    golden_section_search(const golden_section_search& old);
    golden_section_search& operator=(const golden_section_search &old);
    
    void ExeGoldenSectionSearch( float &cost_value, float &cost_ref, float *newton_input_seq, float *mcmpc_input_seq, SampleInfo *sample, int *indices, float *_state, float *_param, float *_ref, float *_cnstrnt, float *_weight);
    void SetupGoldenSample( );
};
#endif