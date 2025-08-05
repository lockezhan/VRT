#ifndef BASE_KERNEL_HPP
#define BASE_KERNEL_HPP

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

void write_ddr1(double* Output_mem, hls::stream<double>& fifo0, hls::stream<double>& fifo1, hls::stream<double>& fifo2){
    Output_mem[0] = fifo0.read();
    Output_mem[1] = fifo1.read();
    Output_mem[2] = fifo2.read();
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


#endif // BASE_KERNEL_HPP