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

#include "../include/savitzky_golay_filter.cuh"

savitzky_golay_filter::savitzky_golay_filter()
{
    sgf_idx = (IndexStructure *)malloc(sizeof(IndexStructure));
    SetupIndicesSampleBasedNewton(sgf_idx);
    coefficients = (float *)malloc(sizeof(float) * 7);
    window = (float *)malloc(sizeof(float) * 7 * sgf_idx->dim_of_input);

    coefficients[0] = -2.0f;
    coefficients[1] = 3.0f;
    coefficients[2] = 6.0f;
    coefficients[3] = 7.0f;
    coefficients[4] = 6.0f;
    coefficients[5] = 3.0f;
    coefficients[6] = -2.0f;

    normalization = 21.0f;
    for(int i = 0; i < 7 * sgf_idx->dim_of_input; i++)
    {
        window[i] = 0.0f;
    }
}

savitzky_golay_filter::~savitzky_golay_filter()
{
    free(coefficients);
    free(window);
}

void savitzky_golay_filter::Smoothing(float *u, float *seq)
{
    input_id = 0;
    window_id = 0;
    for(int i = 3; i < 7; i++)
    {
        for(int k = 0; k < sgf_idx->dim_of_input; k++)
        {
            input_id = (i-3) * sgf_idx->dim_of_input + k;
            window_id = i * sgf_idx->dim_of_input + k;
            window[window_id] = seq[input_id];
        }
    }

    for(int i = 0;i < sgf_idx->dim_of_input; i++)
    {
        for(int k = 0; k < 7; k++)
        {
            input_id = i;
            window_id = k * sgf_idx->dim_of_input + i;
            if(k == 0) u[input_id] = 0.0f;
            u[input_id] += coefficients[k] * window[window_id] / normalization;
        }
    }

    for(int i = 0; i < 3; i++)
    {
        for(int k = 0; k < sgf_idx->dim_of_input; k++)
        {
            window_id = i * sgf_idx->dim_of_input + k;
            window_cpy_id = (i+1) * sgf_idx->dim_of_input + k;
            if(k < 2) window[window_id] = window[window_cpy_id];
            if(k == 2) window[window_id] = u[k];
        }
    }
}