/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      project/example_inverted_pendulum/main.cu
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/

#include "../../include/mcmpc_toolkit.cuh"

int main(int argc, char **argv)
{
    mcmpc myMPC;
    // sample_based_newton_method myMPC;

    // Optional settings
    myMPC.Set(HYPERBOLIC, SET_COOLING_METHOD);
    myMPC.Set(TIME_INVARIANT , SET_REFERENCE_TYPE);
    // myMPC.Set(GOLDEN_SECTION, SET_STEP_WIDTH_ADJUSTING_METHOD);
    // 
    float p[2] = {0.024f, 0.2f};
    float Jp = p[0] * pow(p[1], 2) / 3.0f;

    float state[OCP_SETTINGS::DIM_OF_STATE] = {0.0f, 0.0f, M_PI + 0.01f, 0.0f};
    float u[OCP_SETTINGS::DIM_OF_INPUT] = {0.0f};
    float param[OCP_SETTINGS::DIM_OF_PARAMETER] = {0.1f, 0.024f, 0.2f, Jp, 1.265f, 1e-7, 9.80665f};
    float constraint[OCP_SETTINGS::DIM_OF_CONSTRAINTS] = {-1.0, 1.0, -0.5, 0.5};
    float weight_matrix[OCP_SETTINGS::DIM_OF_WEIGHT_MATRIX] = {3.0f, 0.04f, 10.0f, 0.04f, 0.5f};
    float reference[OCP_SETTINGS::DIM_OF_REFERENCE] = { };

    myMPC.Set(state, SET_STATE);
    myMPC.Set(u, SET_INPUT);
    myMPC.Set(param, SET_PARAMETER);
    myMPC.Set(constraint, SET_CONSTRAINT);
    myMPC.Set(weight_matrix, SET_WEIGHT_MATRIX);
    myMPC.Set(reference, SET_REFERENCE);

    for(int t = 0; t < OCP_SETTINGS::SIMULATION_STEPS; t++)
    {
        myMPC.ExecuteMPC( u );

        myMPC.ExecuteForwardSimulation(state, u, RUNGE_KUTTA_45);

        myMPC.Set(state, SET_STATE);

        myMPC.WriteDataToFile( );
        // myMPC.WriteDataToFile( u );
    }
   
}