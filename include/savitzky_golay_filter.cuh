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

#ifndef SAVITZKY_GOLAY_FILTER_CUH
#define SAVITZKY_GOLAY_FILTER_CUH

#include "common_header.cuh"
#include "data_structures.cuh"

class savitzky_golay_filter
{
private:
    int input_id;
    int window_id;
    int window_cpy_id;
    float *coefficients;
    float *window;
    float normalization;

public:
    IndexStructure *sgf_idx;
    savitzky_golay_filter();
    ~savitzky_golay_filter();
    void Smoothing(float *u, float *seq);
};


#endif