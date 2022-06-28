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
    // mcmpc myMPC;
    sample_based_newton_method myMPC;

    // Optional settings
    myMPC.Set(HYPERBOLIC, SET_COOLING_METHOD);
    myMPC.Set(TIME_VARIANT , SET_REFERENCE_TYPE);
    // 
    float thrust_max = 230.0 * 230.0;

    float state[OCP_SETTINGS::DIM_OF_STATE] = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    float u[OCP_SETTINGS::DIM_OF_INPUT] = {9.8066, 0.0, 0.0, 0.0};
    float param[OCP_SETTINGS::DIM_OF_PARAMETER] = {9.806650, 150.0, 230.0,thrust_max, 5.0, 4.0e-6, 0.0085, 0.008, 0.0165, 1.0, 0.5};
    float constraint[OCP_SETTINGS::DIM_OF_CONSTRAINTS] = {-0.2, 0.2, -20.0, 20.0, 0.0, 25.0};
    float weight_matrix[OCP_SETTINGS::DIM_OF_WEIGHT_MATRIX] = { };
    float reference[OCP_SETTINGS::DIM_OF_REFERENCE] = { 9.8066, 0.0, 0.0, 0.0 };

    // Set weight matrix Q for state
    weight_matrix[0] = 10.0;
    weight_matrix[1] = 1.0;
    weight_matrix[2] = 10.0;
    weight_matrix[3] = 1.0;
    weight_matrix[4] = 50.0;
    weight_matrix[5] = 5.0;
    weight_matrix[6] = 1.0;
    weight_matrix[7] = 1.0;
    weight_matrix[8] = 1.0;
    weight_matrix[9] = 50.0;
    weight_matrix[10] = 50.0;
    weight_matrix[11] = 50.0;
    // Set weight matrix R for input
    weight_matrix[12] = 0.1;
    weight_matrix[13] = 0.1;
    weight_matrix[14] = 0.1;
    weight_matrix[15] = 0.1;

    // 以降は2022.6.28から編集予定
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

        if(300 <= t && t < 310)
        {
            reference[1] += 0.25;
            reference[2] += 0.25;
            reference[3] += 0.25;
            myMPC.Set(reference, SET_REFERENCE);
        }

        myMPC.WriteDataToFile( );
        // myMPC.WriteDataToFile( u );
    }
   
}