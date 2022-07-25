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
    myMPC.Set(TIME_INVARIANT , SET_REFERENCE_TYPE);
    myMPC.Set(GOLDEN_SECTION, SET_STEP_WIDTH_ADJUSTING_METHOD);
    myMPC.Set(EIGEN_VALUE_DECOM, SET_SOLVER);
    // 

    float state[OCP_SETTINGS::DIM_OF_STATE] = {-2.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float u[OCP_SETTINGS::DIM_OF_INPUT] = {0.0f, 0.0f};
    // float plot_u[OCP_SETTINGS::DIM_OF_INPUT] = { };
    float param[OCP_SETTINGS::DIM_OF_PARAMETER] = {1.0f, 9.8067f, 1.0f};
    float constraint[OCP_SETTINGS::DIM_OF_CONSTRAINTS] = {-0.3, 0.3, -2.0, 2.0};
    float weight_matrix[OCP_SETTINGS::DIM_OF_WEIGHT_MATRIX] = {1.0f , 2.0f, 2.0f, 1.0f, 1.0f, 4.0f, 0.05f, 0.05f};
    float reference[OCP_SETTINGS::DIM_OF_REFERENCE] = {2.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f};

    // 以降は2022.6.28から編集予定
    myMPC.Set(state, SET_STATE);
    myMPC.Set(u, SET_INPUT);
    myMPC.Set(param, SET_PARAMETER);
    myMPC.Set(constraint, SET_CONSTRAINT);
    myMPC.Set(weight_matrix, SET_WEIGHT_MATRIX);
    myMPC.Set(reference, SET_REFERENCE);

    float a[7] = { };
    int counter = 0;

    for(int t = 0; t < OCP_SETTINGS::SIMULATION_STEPS; t++)
    {
        myMPC.ExecuteMPC( u );

        myMPC.ExecuteForwardSimulation(state, u, RUNGE_KUTTA_45);
        if(abs(state[0]-reference[0]) < 0.05f && abs(state[2]-reference[2]) < 0.05f) counter += 1;
        
        if(counter > 500){
            reference[0] = -1.0 * reference[0];
            myMPC.Set(reference, SET_REFERENCE);
            counter = 0;
        }

        a[0] = state[2] * sinf(state[4]);
        a[1] = 0.2f;
        a[2] = powf(state[0] + a[0], 2);
        a[3] = a[1] * a[2];
        a[4] = 1.25f;
        a[5] = a[3] + a[4];
        a[6] = state[2] * cosf(state[4]);
        state[6] = a[6] - a[5];
        myMPC.Set(state, SET_STATE);
        // plot_u[0] = param[1] + u[0] - u[2] + u[3];
        // plot_u[1] = param[1] + u[0] + u[2] + u[3];
        // plot_u[2] = param[1] + u[0] + u[1] - u[3];
        // plot_u[3] = param[1] + u[0] - u[1] - u[3];
        // myMPC.WriteDataToFile( );
        myMPC.WriteDataToFile( u );
    }
   
}