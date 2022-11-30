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
    time_t time_value;
    struct tm *time_object;
    int before_tm_min;
    time(&time_value);
    time_object = localtime( &time_value );
    before_tm_min = time_object->tm_min;
    
    mcmpc myMPC;
    // sample_based_newton_method myMPC;
    // savitzky_golay_filter sgFilter;
    // Optional settings
    // myMPC.Set(HYPERBOLIC, SET_COOLING_METHOD);
    myMPC.Set(GEOMETRIC, SET_COOLING_METHOD);
    myMPC.Set(TIME_INVARIANT , SET_REFERENCE_TYPE);
    myMPC.Set(GOLDEN_SECTION, SET_STEP_WIDTH_ADJUSTING_METHOD);
    myMPC.Set(EIGEN_VALUE_DECOM, SET_SOLVER);
    // 
    // float p[2] = {0.024f, 0.2f};
    // float Jp = p[0] * pow(p[1], 2) / 3.0f;
    // float wall_position = atof(argv[1]) / 100;
    // printf("%f th wall position == %f\n", atof(argv[1]),  wall_position);

    float state[OCP_SETTINGS::DIM_OF_STATE] = {2.5, 0.25, 0.5, 3.0};
    float u[OCP_SETTINGS::DIM_OF_INPUT] = {0.0f};
    // float param[OCP_SETTINGS::DIM_OF_PARAMETER] = {0.1f, 0.024f, 0.2f, Jp, 1.265f, 1e-6, 9.80665f, -0.25, 0.1, 0.46}; // <== Success 2022.9.1
    float param[OCP_SETTINGS::DIM_OF_PARAMETER] = {1.0, -1.6658, -11.934, 3.5377e-6, 43.0344, 44.7524, -9.1392e-5, 9.43, -35.3774};
    // float constraint[OCP_SETTINGS::DIM_OF_CONSTRAINTS] = {-1.0, 1.0, -0.5, 0.5}; // For utlizing collision
    float constraint[OCP_SETTINGS::DIM_OF_CONSTRAINTS] = {-1.0, 1.0, -0.04, 0.1}; // For predict as constraint 
    float weight_matrix[OCP_SETTINGS::DIM_OF_WEIGHT_MATRIX] = {1.75f, 1.75f, 0.04f, 0.05f, 1.0f};
    float reference[OCP_SETTINGS::DIM_OF_REFERENCE] = { };

    float optimal[OCP_SETTINGS::NUM_OF_PREDICTION_STEPS] {2.19621, 2.05308, 1.92608, 1.81332, 1.7131, 1.62413, 1.54500, 1.47462, 1.41195, 1.35606, 1.30609,
                                                        1.26123, 1.22068, 1.18368, 1.14946, 1.11723, 1.08617, 1.05538, 1.02391, 0.99069, 0.954534, 0.9140826,
                                                        0.867796, 0.8138970, 0.7503307, 0.674709, 0.584248, 0.475696, 0.345243, 0.1884247};
    
    myMPC.SetOptimal(optimal);                                                        

    myMPC.Set(state, SET_STATE);
    myMPC.Set(u, SET_INPUT);
    myMPC.Set(param, SET_PARAMETER);
    myMPC.Set(constraint, SET_CONSTRAINT);
    myMPC.Set(weight_matrix, SET_WEIGHT_MATRIX);
    myMPC.Set(reference, SET_REFERENCE);

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

    time(&time_value);
    time_object = localtime( &time_value );
    while(time_object->tm_min == before_tm_min && time_object->tm_sec < 58)
    {
        sleep(2);
        time(&time_value);
        time_object = localtime( &time_value );
    }    
   return 0;
}