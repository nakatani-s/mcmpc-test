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
    coefficients = (float *)malloc(sizeof(float) * 9);
    window = (float *)malloc(sizeof(float) * 9 * sgf_idx->dim_of_input);

    coefficients[0] = -21.0f;
    coefficients[1] = 14.0f;
    coefficients[2] = 39.0f;
    coefficients[3] = 54.0f;
    coefficients[4] = 59.0f;
    coefficients[5] = 54.0f;
    coefficients[6] = 39.0f;
    coefficients[7] = 14.0f;
    coefficients[8] = -21.0f;

    normalization = 231.0f;
    for(int i = 0; i < 9 * sgf_idx->dim_of_input; i++)
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
    for(int i = 4; i < 9; i++)
    {
        for(int k = 0; k < sgf_idx->dim_of_input; k++)
        {
            input_id = (i-4) * sgf_idx->dim_of_input + k;
            window_id = i * sgf_idx->dim_of_input + k;
            window[window_id] = seq[input_id];
        }
    }

    for(int i = 0;i < sgf_idx->dim_of_input; i++)
    {
        for(int k = 0; k < 9; k++)
        {
            input_id = i;
            window_id = k * sgf_idx->dim_of_input + i;
            if(k == 0) u[input_id] = 0.0f;
            u[input_id] += coefficients[k] * window[window_id] / normalization;
            // u[input_id] += window[window_id] / 9;
        }
    }

    for(int i = 0; i < 4; i++)
    {
        for(int k = 0; k < sgf_idx->dim_of_input; k++)
        {
            window_id = i * sgf_idx->dim_of_input + k;
            window_cpy_id = (i+1) * sgf_idx->dim_of_input + k;
            if(i < 3) window[window_id] = window[window_cpy_id];
            if(i == 3) window[window_id] = seq[k];
        }
    }
}