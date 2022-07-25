/*
 *  This file is part of MCMPC toolkit.
 *  
 *  MCMPC toolkit -- A toolkit for Model Predictive Control
*/
/* 
    [file]      include/mcmpc.cuh
    [author]    Shintaro Nakatani
    [date]      2022.6.15
*/
#ifndef MCMPC_CUH
#define MCMPC_CUH
#include "common_header.cuh"
#include "data_structures.cuh"

class mcmpc{
public:
    int time_steps; // 時刻をplotするのに使用
    unsigned int num_blocks, num_random_seed, thread_per_block;
    float denominator; // 最適化計算時の正規化項
    float cost_value_mcmpc; // MCMPCの評価値を格納（plotに使用）
    float cumsum_cost;
    float all_time; // 処理時間を格納(plotに使用)
    float control_cycle; // 制御周期
    float *_state, *_ref, *_param, *_cnstrnt, *_weight; // 状態、参照軌道、モデルの物理パラメータ、制約、評価関数の重みを管理（ユニファイドメモリ）

    float *mcmpc_input_sequences; // 推定入力時系列を格納（ユニファイドメモリ、サンプリングの平均値として使用）

    curandState *dev_random_seed; // デバイス関数内で乱数生成するための種

    SampleInfo *sample; // ランダムサンプリングの結果を保持する構造体，ユニファイドメモリで管理

    CoolingMethod cooling_method; // 冷却方法を指定する列挙子(幾何 or 双曲線 or なし)

    ReferenceType ref_type; // 参照軌道の形式を決める列挙子（時変 o r固定）

    StepWidthDecisiveMethod line_search; //

    LinearEquationSolver solver_type;

    IndexStructure *hst_idx, *dev_idx; // ホライズン、状態・入力の次元などの情報を管理する構造体(hst_idx:=ホスト関数用, dev_idx:=デバイス関数用)

    thrust::device_vector<int> indices_dev_vec;
    thrust::device_vector<float> sort_key_dev_vec;
    thrust::host_vector<float> cumsum_weight_hst_vec;
    thrust::device_vector<float> cumsum_weight_dev_vec;
    thrust::device_vector<float> weight_dev_vec;

    FILE *fp_state, *fp_input, *fp_cost;

    mcmpc(); // コンストラクタ
    ~mcmpc(); // デストラクタ

    virtual void ExecuteMPC(float *current_input);
    void Set(float *a, ValueType type);
    void Set(CoolingMethod method, ValueType type);
    void Set(ReferenceType method, ValueType type);
    void Set(StepWidthDecisiveMethod method, ValueType type);
    void Set(LinearEquationSolver method, ValueType type);
    void ExecuteForwardSimulation(float *state, float *input, IntegralMethod method);
    void MonteCarloSimulation( );
    virtual void WriteDataToFile( );
    virtual void WriteDataToFile(float *_input);
    void FreeAllCudaArrayInBaseClass();
};
#endif