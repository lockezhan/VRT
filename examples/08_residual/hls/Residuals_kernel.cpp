#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <hls_math.h>
#include <cmath>
#include <cstdlib>
#include "ap_int.h"
#include "hls_vector.h"

#define VEC_SIZE 8    // 根据片上资源调整块大小
typedef hls::vector<double, VEC_SIZE> vec_double;

void read_ddr(vec_double* Input_mem, hls::stream<vec_double,64>& fifo, int N, int num){
    for(int i = 0; i < N-1; i++){
        #pragma HLS PIPELINE II=1
        fifo.write(Input_mem[i]);
    }
    vec_double data;
    data = Input_mem[N-1];
    if (num != 0) {  // 仅当需要填充时才执行
        for(int i = num; i < VEC_SIZE; i++){
            #pragma HLS PIPELINE II=1
            data[i] = 0;
        }
    }
    fifo.write(data);
}

void read_ddr_two(vec_double* Input_mem, hls::stream<vec_double,64>& fifo0, hls::stream<vec_double,64>& fifo1, int N, int num){
    for(int i = 0; i < N-1; i++){
        #pragma HLS PIPELINE II=1
        fifo0.write(Input_mem[i]);
        fifo1.write(Input_mem[i]);
    }
    vec_double data;
    data = Input_mem[N-1];
    if (num != 0) {  // 仅当需要填充时才执行
        for(int i = num; i < VEC_SIZE; i++){
            #pragma HLS PIPELINE II=1
            data[i] = 0;
        }
    }
    fifo0.write(data);
    fifo1.write(data);
}

void write_ddr(vec_double* Output_mem, hls::stream<vec_double,64>& fifo, int N){
    for(int i = 0; i < N; i++){
        #pragma HLS PIPELINE II=1
        Output_mem[i] = fifo.read();
    }
}

void write_ddr1(double* Output_mem, hls::stream<double>& fifo0, hls::stream<double>& fifo1, hls::stream<double>& fifo2, hls::stream<double>& fifo3, hls::stream<double>& fifo4, hls::stream<double>& fifo5){
    Output_mem[0] = fifo0.read();
    double dPrimalFeasibility = fifo1.read();
    dPrimalFeasibility = sqrt(dPrimalFeasibility);
    Output_mem[1] = dPrimalFeasibility;
    double dPrimalObj0 = fifo2.read();
    double dPrimalObj1 = fifo3.read();
    double dPrimalObj2 = fifo4.read();
    dPrimalObj0 = dPrimalObj0 + dPrimalObj1 - dPrimalObj2;
    Output_mem[2] = dPrimalObj0;
    Output_mem[3] = fifo5.read();
}

void cupdlp_dot(
    hls::stream<vec_double,64>& x,
    hls::stream<vec_double,64>& y,
    hls::stream<double>& result,
    int N
){
    // Shift register based accumulation
    const int DELAY = 6;  // Match floating point adder latency
    
    double shift_reg[8][DELAY + 1];
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=0
    
    accum_loop:
    for(int i = 0; i < N; i++){
        #pragma HLS PIPELINE II=1
        
        vec_double d1 = x.read();
        vec_double d2 = y.read();
        
        // Shift and accumulate
        for(int j = 0; j < 8; j++){
            #pragma HLS UNROLL
            double new_val = d1[j] * d2[j] + shift_reg[j][DELAY];
            
            // Shift register
            for(int k = DELAY; k > 0; k--){
                #pragma HLS UNROLL
                shift_reg[j][k] = shift_reg[j][k-1];
            }
            shift_reg[j][0] = new_val;
        }
    }
    
    // Final accumulation from shift registers
    double res = 0.0;
    final_loop:
    for(int i = 0; i < 8; i++){
        for(int j = 0; j <= DELAY; j++){
            res += shift_reg[i][j];
        }
    }
    
    result.write(res);
}

void cupdlp_axpy(
    double alpha,
    hls::stream<vec_double,64>& x,
    hls::stream<vec_double,64>& y_in,
    hls::stream<vec_double,64>& y_out,
    int N)
{
    // 主循环：处理完整的 4 元素块
    for(int i = 0; i < N; i++){
        #pragma HLS PIPELINE II=1
        vec_double d1 = x.read();
        vec_double d2 = y_in.read();
        vec_double dout;
        for(int j = 0; j < VEC_SIZE; j++){
            #pragma HLS UNROLL
            dout[j] = d1[j] * alpha + d2[j];
        }
        y_out.write(dout);
    }
}

void cupdlp_axpy1(
    double alpha,
    hls::stream<vec_double,64>& x,
    hls::stream<vec_double,64>& y_in,
    hls::stream<vec_double,64>& y_out0,
    hls::stream<vec_double,64>& y_out1,
    hls::stream<vec_double,64>& y_out2,
    int N)
{
    // 主循环：处理完整的 4 元素块
    for(int i = 0; i < N; i++){
        #pragma HLS PIPELINE II=1
        vec_double d1 = x.read();
        vec_double d2 = y_in.read();
        vec_double dout;
        for(int j = 0; j < VEC_SIZE; j++){
            #pragma HLS UNROLL
            dout[j] = d1[j] * alpha + d2[j];
        }
        y_out0.write(dout);
        y_out1.write(dout);
        y_out2.write(dout);
    }
}

void cupdlp_axpy2(
    double alpha0,
    double alpha1,
    hls::stream<vec_double,64>& in0,
    hls::stream<vec_double,64>& in1,
    hls::stream<vec_double,64>& in2,
    hls::stream<vec_double,64>& out,
    int N)
{
    // 主循环：处理完整的 4 元素块
    for(int i = 0; i < N; i++){
        #pragma HLS PIPELINE II=1
        vec_double d0 = in0.read();
        vec_double d1 = in1.read();
        vec_double d2 = in2.read();
        vec_double dout;
        for(int j = 0; j < VEC_SIZE; j++){
            #pragma HLS UNROLL
            dout[j] = d0[j] + d1[j] * alpha0 + d2[j] * alpha1;
        }
        out.write(dout);
    }
}

void cupdlp_scaleVector(
    double alpha,
    hls::stream<vec_double,64>& x_in,
    hls::stream<vec_double,64>& x_out,
    int N
){
    // 主循环：处理完整的 4 元素块
    for(int i = 0; i < N; i++){
        #pragma HLS PIPELINE II=1
        vec_double d = x_in.read();
        vec_double dout;
        for(int j = 0; j < VEC_SIZE; j++){
            #pragma HLS UNROLL
            dout[j] = d[j] * alpha;
        }
        x_out.write(dout);
    }
}

void cupdlp_projNeg(
    hls::stream<vec_double,64>& y_in,
    hls::stream<vec_double,64>& y_out,
    int N
){
    // 主循环：处理完整的 4 元素块
    for(int i = 0; i < N; i++){
        #pragma HLS PIPELINE II=1
        vec_double d = y_in.read();
        vec_double dout;
    
        for(int j = 0; j < VEC_SIZE; j++){
            #pragma HLS UNROLL
            if(d[j] < 0.0){
                dout[j] = d[j];
            }else{
                dout[j] = 0.0;
            }
        }
        y_out.write(dout);
    }
}

void cupdlp_projNeg1(
    hls::stream<vec_double,64>& y_in,
    hls::stream<vec_double,64>& y_out,
    int N,
    int nEqs
){
    // 主循环：处理完整的 4 元素块
    for(int i = 0; i < N; i++){
        #pragma HLS PIPELINE II=1
        vec_double d = y_in.read();
        vec_double dout;
        if((i+1)*VEC_SIZE <= nEqs){
            dout = d;
        }
        else if(i*VEC_SIZE < nEqs && (i+1)*VEC_SIZE > nEqs){
            dout = d;
            for(int j = nEqs % VEC_SIZE; j < VEC_SIZE; j++){
                #pragma HLS UNROLL
                if(d[j] < 0.0){
                    dout[j] = d[j];
                }else{
                    dout[j] = 0.0;
                }
            }
        }
        else{
            for(int j = 0; j < VEC_SIZE; j++){
                #pragma HLS UNROLL
                if(d[j] < 0.0){
                    dout[j] = d[j];
                }else{
                    dout[j] = 0.0;
                }
            }
        }
        y_out.write(dout);
    }
}

void cupdlp_projPos(
    hls::stream<vec_double,64>& y_in,
    hls::stream<vec_double,64>& y_out,
    int N
){
    // 主循环：处理完整的 4 元素块
    for(int i = 0; i < N; i++){
        #pragma HLS PIPELINE II=1
        vec_double d = y_in.read();
        vec_double dout;
    
        for(int j = 0; j < VEC_SIZE; j++){
            #pragma HLS UNROLL
            if(d[j] > 0.0){
                dout[j] = d[j];
            }else{
                dout[j] = 0.0;
            }
        }
        y_out.write(dout);
    }
}

void cupdlp_edot(
    hls::stream<vec_double,64>& x,
    hls::stream<vec_double,64>& y_in,
    hls::stream<vec_double,64>& x_out,
    int N)
{
    // 主循环：处理完整的 4 元素块
    for(int i = 0; i < N; i++){
        #pragma HLS PIPELINE II=1
        vec_double d1 = x.read();
        vec_double d2 = y_in.read();
        vec_double dout;
        for(int j = 0; j < VEC_SIZE; j++){
            #pragma HLS UNROLL
            dout[j] = d1[j] * d2[j];
        }
        x_out.write(dout);
    }
}

void cupdlp_edot1(
    hls::stream<vec_double,64>& x,
    hls::stream<vec_double,64>& y_in,
    hls::stream<vec_double,64>& out0,
    hls::stream<vec_double,64>& out1,
    int N)
{
    // 主循环：处理完整的 4 元素块
    for(int i = 0; i < N; i++){
        #pragma HLS PIPELINE II=1
        vec_double d1 = x.read();
        vec_double d2 = y_in.read();
        vec_double dout;
        for(int j = 0; j < VEC_SIZE; j++){
            #pragma HLS UNROLL
            dout[j] = d1[j] * d2[j];
        }
        out0.write(dout);
        out1.write(dout);
    }
}

void cupdlp_twoNorm(
    hls::stream<vec_double,64>& x,
    hls::stream<double>& result,
    int N
){
    // Shift register based accumulation
    const int DELAY = 6;  // Match floating point adder latency
    
    double shift_reg[8][DELAY + 1];
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=0

    accum_loop:
    for(int i = 0; i < N; i++){
        #pragma HLS PIPELINE II=1
        
        vec_double d1 = x.read();
        
        // Shift and accumulate
        for(int j = 0; j < 8; j++){
            #pragma HLS UNROLL
            double new_val = d1[j] * d1[j] + shift_reg[j][DELAY];
            
            // Shift register
            for(int k = DELAY; k > 0; k--){
                #pragma HLS UNROLL
                shift_reg[j][k] = shift_reg[j][k-1];
            }
            shift_reg[j][0] = new_val;
        }
    }
    
    // Final accumulation from shift registers
    double res = 0.0;
    res_loop:
    for(int i = 0; i < 8; i++){
        for(int j = 0; j <= DELAY; j++){
            res += shift_reg[i][j];
        }
    }
    
    result.write(res);
}


void primal_objective(
    vec_double *x, 
    hls::stream<vec_double,64>& cost_stream,
    hls::stream<double>& dPrimalObj,
    int nCols
){
    #pragma HLS DATAFLOW    
    int loop_num = (nCols + VEC_SIZE - 1) / VEC_SIZE;
    int tail_num = nCols % VEC_SIZE;
    double result = 0.0;

    hls::stream<vec_double,64> x_stream;
    read_ddr(x, x_stream, loop_num, tail_num);
    cupdlp_dot(x_stream, cost_stream, dPrimalObj, loop_num);
}

void PrimalFeasibility(
    vec_double* ax,
    hls::stream<vec_double,64>& rhs_stream,
    vec_double* rowScale,
    hls::stream<double>& dPrimalFeasibility,
    int nRows, 
    int nEqs
){
    #pragma HLS DATAFLOW    
    // dPrimalFeasibility is the lhs of "Primal feasibility" in [cupdlp-c]
    int loop_num = (nRows + VEC_SIZE - 1) / VEC_SIZE;
    int tail_num = nRows % VEC_SIZE;
    double result = 0.0;
    
    hls::stream<vec_double,64> primalResidual0, primalResidual1;
    hls::stream<vec_double,64> axpy_to_projNeg, projNeg_to_edot;
    hls::stream<vec_double,64> rowScale_stream;

    read_ddr(ax, primalResidual0, loop_num, tail_num);
    cupdlp_axpy(-1.0, rhs_stream, primalResidual0, axpy_to_projNeg, loop_num);
    cupdlp_projNeg1(axpy_to_projNeg, projNeg_to_edot, loop_num, nEqs);

    read_ddr(rowScale, rowScale_stream, loop_num, tail_num);
    cupdlp_edot(projNeg_to_edot, rowScale_stream, primalResidual1, loop_num);
    cupdlp_twoNorm(primalResidual1, dPrimalFeasibility, loop_num);
}

void dual_objective(
    vec_double *y, 
    hls::stream<vec_double,64>& rhs_stream,
    hls::stream<double>& dPrimalObj0,
    int nRows
){
    #pragma HLS DATAFLOW    
    int loop_num = (nRows + VEC_SIZE - 1) / VEC_SIZE;
    int tail_num = nRows % VEC_SIZE;
    double result = 0.0;

    hls::stream<vec_double,64> y_stream;

    read_ddr(y, y_stream, loop_num, tail_num);

    cupdlp_dot(y_stream, rhs_stream, dPrimalObj0, loop_num);
}

void compute_lambda(
    vec_double *aty, 
    hls::stream<vec_double,64>& cost_stream,
    vec_double* hasLower,
    vec_double* dLowerFiltered,
    vec_double* hasUpper,
    vec_double* dUpperFiltered,
    vec_double* colScale,
    hls::stream<double>& dPrimalObj1,
    hls::stream<double>& dPrimalObj2,
    hls::stream<double>& dDualFeasibility,
    int nCols
){
    #pragma HLS DATAFLOW    
    int loop_num = (nCols + VEC_SIZE - 1) / VEC_SIZE;
    int tail_num = nCols % VEC_SIZE;
    double result = 0.0;

    hls::stream<vec_double,64> hasLower_stream;
    hls::stream<vec_double,64> dLowerFiltered_stream;
    hls::stream<vec_double,64> hasUpper_stream;
    hls::stream<vec_double,64> dUpperFiltered_stream;
    hls::stream<vec_double,64> colScale_stream;

    hls::stream<vec_double,64> dualResidual_stream0, dualResidual_stream1, dualResidual_stream2, dualResidual_stream3, dualResidual_stream4;
    hls::stream<vec_double,64> dSlackPos_stream0, dSlackPos_stream1, dSlackPos_stream2, dSlackPos_stream3;
    hls::stream<vec_double,64> dSlackNeg_stream0, dSlackNeg_stream1, dSlackNeg_stream2, dSlackNeg_stream3, dSlackNeg_stream4;

    read_ddr(aty, dualResidual_stream0, loop_num, tail_num);
    read_ddr(hasLower, hasLower_stream, loop_num, tail_num);
    read_ddr(dLowerFiltered, dLowerFiltered_stream, loop_num, tail_num);
    read_ddr(hasUpper, hasUpper_stream, loop_num, tail_num);
    read_ddr(dUpperFiltered, dUpperFiltered_stream, loop_num, tail_num);
    read_ddr(colScale, colScale_stream, loop_num, tail_num);

    cupdlp_scaleVector(-1.0, dualResidual_stream0, dualResidual_stream1, loop_num);
    cupdlp_axpy1(1.0, cost_stream, dualResidual_stream1, dSlackPos_stream0, dSlackNeg_stream0, dualResidual_stream2, loop_num);

    cupdlp_projPos(dSlackPos_stream0, dSlackPos_stream1, loop_num);
    cupdlp_edot1(dSlackPos_stream1, hasLower_stream, dSlackPos_stream2, dSlackPos_stream3, loop_num);
    cupdlp_dot(dSlackPos_stream2, dLowerFiltered_stream, dPrimalObj1, loop_num);

    cupdlp_projNeg(dSlackNeg_stream0, dSlackNeg_stream1, loop_num);
    cupdlp_scaleVector(-1.0, dSlackNeg_stream1, dSlackNeg_stream2, loop_num);
    cupdlp_edot1(dSlackNeg_stream2, hasUpper_stream, dSlackNeg_stream3, dSlackNeg_stream4, loop_num);
    cupdlp_dot(dSlackNeg_stream3, dUpperFiltered_stream, dPrimalObj2, loop_num);

    cupdlp_axpy2(-1.0, 1.0, dualResidual_stream2, dSlackPos_stream3, dSlackNeg_stream4, dualResidual_stream3, loop_num);
    cupdlp_edot(dualResidual_stream3, colScale_stream, dualResidual_stream4, loop_num);
    cupdlp_twoNorm(dualResidual_stream4, dDualFeasibility, loop_num);
}

void Residuals_kernel(vec_double *x, vec_double* cost, vec_double* ax, vec_double* rhs, vec_double* rowScale, 
    vec_double *aty, vec_double* y,
    vec_double* hasLower, vec_double* dLowerFiltered, 
    vec_double* hasUpper, vec_double* dUpperFiltered, vec_double* colScale,
    double* result,
    int nCols, int nRows, int nEqs
){
    #pragma HLS interface m_axi offset=slave bundle=gmem0 port=x max_read_burst_length=64 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=x
    #pragma HLS interface m_axi offset=slave bundle=gmem1 port=cost max_read_burst_length=64 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=cost
    #pragma HLS interface m_axi offset=slave bundle=gmem2 port=ax max_read_burst_length=64 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=ax
    #pragma HLS interface m_axi offset=slave bundle=gmem3 port=rhs max_read_burst_length=64 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=rhs
    #pragma HLS interface m_axi offset=slave bundle=gmem4 port=rowScale max_read_burst_length=64 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=rowScale
    #pragma HLS interface m_axi offset=slave bundle=gmem5 port=aty max_read_burst_length=64 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=aty
    #pragma HLS interface m_axi offset=slave bundle=gmem6 port=y max_read_burst_length=64 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=y
    #pragma HLS interface m_axi offset=slave bundle=gmem7 port=hasLower max_read_burst_length=64 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=hasLower
    #pragma HLS interface m_axi offset=slave bundle=gmem8 port=dLowerFiltered max_read_burst_length=64 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=dLowerFiltered
    #pragma HLS interface m_axi offset=slave bundle=gmem9 port=hasUpper max_read_burst_length=64 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=hasUpper
    #pragma HLS interface m_axi offset=slave bundle=gmem10 port=dUpperFiltered max_read_burst_length=64 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=dUpperFiltered
    #pragma HLS interface m_axi offset=slave bundle=gmem11 port=colScale max_read_burst_length=64 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=colScale
    #pragma HLS interface m_axi offset=slave bundle=gmem12 port=result max_read_burst_length=64 num_read_outstanding=64
    #pragma HLS interface s_axilite bundle=control port=result
    #pragma HLS INTERFACE s_axilite bundle=control port=nCols
    #pragma HLS INTERFACE s_axilite bundle=control port=nRows
    #pragma HLS INTERFACE s_axilite bundle=control port=nEqs
    #pragma HLS INTERFACE s_axilite bundle=control port=return

    
    #pragma HLS DATAFLOW    
    hls::stream<double> dPrimalObj0, dPrimalObj1, dPrimalObj2, dDualFeasibility;
    hls::stream<double> dPrimalObj, dPrimalFeasibility;
    hls::stream<vec_double,64> cost_stream0, cost_stream1;
    hls::stream<vec_double,64> rhs_stream0, rhs_stream1;
    int loop_num = (nCols + VEC_SIZE - 1) / VEC_SIZE;
    int tail_num = nCols % VEC_SIZE;

    read_ddr_two(cost, cost_stream0, cost_stream1, loop_num, tail_num);
    read_ddr_two(rhs, rhs_stream0, rhs_stream1, loop_num, tail_num);
    primal_objective(
        x,
        cost_stream0,
        dPrimalObj,
        nCols
    );

    PrimalFeasibility(
        ax,
        rhs_stream0,
        rowScale,
        dPrimalFeasibility,
        nRows, 
        nEqs
    );

    dual_objective(
        y, 
        rhs_stream1, 
        dPrimalObj0,
        nRows
    );

    compute_lambda(
        aty, 
        cost_stream1,
        hasLower,
        dLowerFiltered,
        hasUpper,
        dUpperFiltered,
        colScale,
        dPrimalObj1,
        dPrimalObj2,
        dDualFeasibility,
        nCols
    );
    write_ddr1(result, dPrimalObj, dPrimalFeasibility, dPrimalObj0, dPrimalObj1, dPrimalObj2, dDualFeasibility);
}