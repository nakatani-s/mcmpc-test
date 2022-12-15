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

void setParamDoublePend(float *param);

int main(int argc, char **argv)
{
    time_t time_value;
    struct tm *time_object;
    int before_tm_min;
    time(&time_value);
    time_object = localtime( &time_value );
    before_tm_min = time_object->tm_min;
    
    // mcmpc myMPC;
    // sample_based_newton_method myMPC;
    cma_mpc myMPC;
    // savitzky_golay_filter sgFilter;
    // Optional settings
    // myMPC.Set(HYPERBOLIC, SET_COOLING_METHOD);
    myMPC.Set(TIME_INVARIANT , SET_REFERENCE_TYPE);
    // myMPC.Set(GOLDEN_SECTION, SET_STEP_WIDTH_ADJUSTING_METHOD);
    // myMPC.Set(EIGEN_VALUE_DECOM, SET_SOLVER);
    // 
    // state:: 6-dimention <==> {z, dot{z}, th1, dot{th1}, th2, dot{th2}}
    float state[OCP_SETTINGS::DIM_OF_STATE] = {0.0f, 0.0f, M_PI+0.01, 0.0f, M_PI+0.01, 0.0f};
    float u[OCP_SETTINGS::DIM_OF_INPUT] = {0.0f};
    // float param[OCP_SETTINGS::DIM_OF_PARAMETER] = {0.1f, 0.024f, 0.2f, Jp, 1.265f, 1e-6, 9.80665f, -0.25, 0.1, 0.46}; // <== Success 2022.9.1
    // param:: 7-dimention <==> {}
    float param[OCP_SETTINGS::DIM_OF_PARAMETER] = {};
    setParamDoublePend(param);
    // float constraint[OCP_SETTINGS::DIM_OF_CONSTRAINTS] = {-1.0, 1.0, -0.5, 0.5}; // For utlizing collision
    float constraint[OCP_SETTINGS::DIM_OF_CONSTRAINTS] = {-15.0, 15.0, -0.45, 0.45}; // For predict as constraint 
    float weight_matrix[OCP_SETTINGS::DIM_OF_WEIGHT_MATRIX] = {2.0f, 0.01f, 2.0f, 0.01f, 2.0f, 0.01f, 0.0001f};
    float reference[OCP_SETTINGS::DIM_OF_REFERENCE] = { };

    myMPC.Set(state, SET_STATE);
    myMPC.Set(u, SET_INPUT);
    myMPC.Set(param, SET_PARAMETER);
    myMPC.Set(constraint, SET_CONSTRAINT);
    myMPC.Set(weight_matrix, SET_WEIGHT_MATRIX);
    myMPC.Set(reference, SET_REFERENCE);
    printf("##### Success set up all variables for run MPC!! #####\n");

    for(int t = 0; t < OCP_SETTINGS::SIMULATION_STEPS; t++)
    {
        myMPC.ExecuteMPC( u );

        // sgFilter.Smoothing(u, myMPC.mcmpc_input_sequences);
        // if(t < 10) u[0] = constraint[1];
        myMPC.ExecuteForwardSimulation(state, u, RUNGE_KUTTA_45);

        // if(state[0] >= param[8])
        // {
        //     float collide[3] = {};
        //     float coefficient = param[9];
        //     collide[0] = param[1] * param[2] * cos(state[2]);
        //     collide[1] = Jp + param[1] * powf(param[2], 2);
        //     collide[2] = collide[0] / collide[1];
        //     state[3] = state[3] + (1+ coefficient) * collide[2] * state[1];
        //     state[1] = -coefficient * state[1];
        //     state[0] = param[8];
        // }
        // if(state[0] <= param[7])
        // {
        //     float collide[3] = {};
        //     float coefficient = param[9];
        //     collide[0] = param[1] * param[2] * cos(state[2]);
        //     collide[1] = Jp + param[1] * powf(param[2], 2);
        //     collide[2] = collide[0] / collide[1];
        //     state[3] = state[3] + (1+coefficient) * collide[2] * state[1];
        //     state[1] = -coefficient * state[1];
        //     state[0] = param[7];
        // }

        myMPC.Set(state, SET_STATE);

        myMPC.WriteDataToFile( );
        // myMPC.WriteDataToFile( u );
    }

    // time(&time_value);
    // time_object = localtime( &time_value );
    // while(time_object->tm_min == before_tm_min && time_object->tm_sec < 50)
    // {
    //     sleep(8);
    //     time(&time_value);
    //     time_object = localtime( &time_value );
    // }    
   return 0;
}

void setParamDoublePend(float *param)
{
    float o[14] = { };
    o[0] = 0.18f;       // m1
    o[1] = 0.078f;      // n1
    o[2] = 0.19f;       // l1
    o[3] = 0.000028f;   // Jn1
    o[4] = 0.089f;      // I1
    o[5] = 0.0001f;     // c1
    o[6] = 0.38f;       // L
    o[7] = 0.10f;       // m2
    o[8] = 0.05f;       // n2
    o[9] = 0.115f;      // l2
    o[10] = 0.000002f;  // Jn2
    o[11] = 0.0018f;    // I2
    o[12] = 0.002f;     // c2
    o[13] = 9.80665f;   // g

    param[0] = o[4] + o[3] + (o[7] + o[8]) * powf(o[6],2);  // I1 + Jn1 + (m2 + n2) * L^2
    param[1] = o[11] + o[10];                               // I2 + Jn2
    param[2] = o[7] * o[9] * o[6];                          // m2 * l2 * L
    param[3] = o[0] * o[2] + o[6] * (o[7] + o[8]);          // m1 * l1 + L * (m2 + n2)
    param[4] = o[7] * o[9];                                 // m2 * l2
    param[5] = o[5];                                        // c1
    param[6] = o[12];                                       // c2
    param[7] = o[13];                                       // g
}