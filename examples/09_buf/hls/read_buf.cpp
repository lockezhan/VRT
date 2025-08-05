#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <hls_math.h>
#include <cmath>
#include <cstdlib>
#include "ap_int.h"
#include "hls_vector.h"

#define VEC_SIZE 8    
#define BUFFER_DEPTH 128  // 增大缓冲深度以提高性能
typedef hls::vector<double, VEC_SIZE> vec_double;

// 方案1：交替分发 - input[0]->IP0, input[1]->IP1, input[2]->IP0, ...
void read_ddr_alternating(vec_double* Input_mem, 
                          hls::stream<vec_double,BUFFER_DEPTH>& fifo0, 
                          hls::stream<vec_double,BUFFER_DEPTH>& fifo1, 
                          int N, int num) {
    
    // 交替分发数据到两个IP
    for(int i = 0; i < N-1; i++){
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=Input_mem inter false
        
        vec_double data = Input_mem[i];
        
        // 交替分发：偶数索引给IP0，奇数索引给IP1
        if (!(i & 1)) {
            fifo0.write(data);
        } else {
            fifo1.write(data);
        }
    }
    
    // 处理最后一个元素
    if (N > 0) {
        vec_double data = Input_mem[N-1];
        if (num != 0) {  
            for(int i = num; i < VEC_SIZE; i++){
                #pragma HLS UNROLL
                data[i] = 0;
            }
        }
        
        // 最后一个元素也按交替规则分发
        if ((N-1) % 2 == 0) {
            fifo0.write(data);
        } else {
            fifo1.write(data);
        }
    }
}

// 数据分发器 - 将数据从内部缓冲分发到单个输出流
void data_distributor_single(hls::stream<vec_double,BUFFER_DEPTH>& input_fifo,
                              hls::stream<vec_double,64>& output,
                              int loop_num) {
    for(int i = 0; i < loop_num; i++) {
        #pragma HLS PIPELINE II=1
        if (!input_fifo.empty()) {
            vec_double data = input_fifo.read();
            output.write(data);
        }
    }
}

void read_buf(
    vec_double* cost, vec_double *aty,
    vec_double* hasLower, vec_double* dLowerFiltered, 
    vec_double* hasUpper, vec_double* dUpperFiltered, 
    vec_double* colScale,
    
    // 为第一个 compute_lambda IP 提供数据
    hls::stream<vec_double,64>& aty_out0,
    hls::stream<vec_double,64>& cost_out0,
    hls::stream<vec_double,64>& hasLower_out0,
    hls::stream<vec_double,64>& dLowerFiltered_out0,
    hls::stream<vec_double,64>& hasUpper_out0,
    hls::stream<vec_double,64>& dUpperFiltered_out0,
    hls::stream<vec_double,64>& colScale_out0,
    
    // 为第二个 compute_lambda IP 提供数据
    hls::stream<vec_double,64>& aty_out1,
    hls::stream<vec_double,64>& cost_out1,
    hls::stream<vec_double,64>& hasLower_out1,
    hls::stream<vec_double,64>& dLowerFiltered_out1,
    hls::stream<vec_double,64>& hasUpper_out1,
    hls::stream<vec_double,64>& dUpperFiltered_out1,
    hls::stream<vec_double,64>& colScale_out1,
    
    int nCols, int nRows, int nEqs
){
    // HLS 接口配置 - 使用不同的 HBM 端口以最大化带宽
    #pragma HLS interface m_axi offset=slave bundle=gmem0 port=cost max_read_burst_length=256 num_read_outstanding=128
    #pragma HLS interface m_axi offset=slave bundle=gmem1 port=aty max_read_burst_length=256 num_read_outstanding=128
    #pragma HLS interface m_axi offset=slave bundle=gmem2 port=hasLower max_read_burst_length=256 num_read_outstanding=128
    #pragma HLS interface m_axi offset=slave bundle=gmem3 port=dLowerFiltered max_read_burst_length=256 num_read_outstanding=128
    #pragma HLS interface m_axi offset=slave bundle=gmem4 port=hasUpper max_read_burst_length=256 num_read_outstanding=128
    #pragma HLS interface m_axi offset=slave bundle=gmem5 port=dUpperFiltered max_read_burst_length=256 num_read_outstanding=128
    #pragma HLS interface m_axi offset=slave bundle=gmem6 port=colScale max_read_burst_length=256 num_read_outstanding=128
    
    // AXI Stream 接口配置
    #pragma HLS interface axis port=aty_out0
    #pragma HLS interface axis port=cost_out0
    #pragma HLS interface axis port=hasLower_out0
    #pragma HLS interface axis port=dLowerFiltered_out0
    #pragma HLS interface axis port=hasUpper_out0
    #pragma HLS interface axis port=dUpperFiltered_out0
    #pragma HLS interface axis port=colScale_out0
    
    #pragma HLS interface axis port=aty_out1
    #pragma HLS interface axis port=cost_out1
    #pragma HLS interface axis port=hasLower_out1
    #pragma HLS interface axis port=dLowerFiltered_out1
    #pragma HLS interface axis port=hasUpper_out1
    #pragma HLS interface axis port=dUpperFiltered_out1
    #pragma HLS interface axis port=colScale_out1
    
    #pragma HLS interface s_axilite bundle=control port=nCols
    #pragma HLS interface s_axilite bundle=control port=nRows
    #pragma HLS interface s_axilite bundle=control port=nEqs
    #pragma HLS interface s_axilite bundle=control port=return
    
    // 使用 DATAFLOW 实现最大并行度
    #pragma HLS DATAFLOW
    
    int loop_num = (nCols + VEC_SIZE - 1) / VEC_SIZE;
    int tail_num = nCols % VEC_SIZE;
    
    // 内部高带宽缓冲流
    static hls::stream<vec_double,BUFFER_DEPTH> aty_buffer0, aty_buffer1;
    static hls::stream<vec_double,BUFFER_DEPTH> cost_buffer0, cost_buffer1;
    static hls::stream<vec_double,BUFFER_DEPTH> hasLower_buffer0, hasLower_buffer1;
    static hls::stream<vec_double,BUFFER_DEPTH> dLowerFiltered_buffer0, dLowerFiltered_buffer1;
    static hls::stream<vec_double,BUFFER_DEPTH> hasUpper_buffer0, hasUpper_buffer1;
    static hls::stream<vec_double,BUFFER_DEPTH> dUpperFiltered_buffer0, dUpperFiltered_buffer1;
    static hls::stream<vec_double,BUFFER_DEPTH> colScale_buffer0, colScale_buffer1;
    
    #pragma HLS STREAM variable=aty_buffer0 depth=BUFFER_DEPTH
    #pragma HLS STREAM variable=aty_buffer1 depth=BUFFER_DEPTH
    #pragma HLS STREAM variable=cost_buffer0 depth=BUFFER_DEPTH
    #pragma HLS STREAM variable=cost_buffer1 depth=BUFFER_DEPTH
    #pragma HLS STREAM variable=hasLower_buffer0 depth=BUFFER_DEPTH
    #pragma HLS STREAM variable=hasLower_buffer1 depth=BUFFER_DEPTH
    #pragma HLS STREAM variable=dLowerFiltered_buffer0 depth=BUFFER_DEPTH
    #pragma HLS STREAM variable=dLowerFiltered_buffer1 depth=BUFFER_DEPTH
    #pragma HLS STREAM variable=hasUpper_buffer0 depth=BUFFER_DEPTH
    #pragma HLS STREAM variable=hasUpper_buffer1 depth=BUFFER_DEPTH
    #pragma HLS STREAM variable=dUpperFiltered_buffer0 depth=BUFFER_DEPTH
    #pragma HLS STREAM variable=dUpperFiltered_buffer1 depth=BUFFER_DEPTH
    #pragma HLS STREAM variable=colScale_buffer0 depth=BUFFER_DEPTH
    #pragma HLS STREAM variable=colScale_buffer1 depth=BUFFER_DEPTH
    
    //交替分发
    read_ddr_alternating(aty, aty_buffer0, aty_buffer1, loop_num, tail_num);
    read_ddr_alternating(cost, cost_buffer0, cost_buffer1, loop_num, tail_num);
    read_ddr_alternating(hasLower, hasLower_buffer0, hasLower_buffer1, loop_num, tail_num);
    read_ddr_alternating(dLowerFiltered, dLowerFiltered_buffer0, dLowerFiltered_buffer1, loop_num, tail_num);
    read_ddr_alternating(hasUpper, hasUpper_buffer0, hasUpper_buffer1, loop_num, tail_num);
    read_ddr_alternating(dUpperFiltered, dUpperFiltered_buffer0, dUpperFiltered_buffer1, loop_num, tail_num);
    read_ddr_alternating(colScale, colScale_buffer0, colScale_buffer1, loop_num, tail_num);
    
    // 阶段2：从内部缓冲分发到对应的输出流
    data_distributor_single(aty_buffer0, aty_out0, loop_num);
    data_distributor_single(aty_buffer1, aty_out1, loop_num);
    
    data_distributor_single(cost_buffer0, cost_out0, loop_num);
    data_distributor_single(cost_buffer1, cost_out1, loop_num);
    
    data_distributor_single(hasLower_buffer0, hasLower_out0, loop_num);
    data_distributor_single(hasLower_buffer1, hasLower_out1, loop_num);
    
    data_distributor_single(dLowerFiltered_buffer0, dLowerFiltered_out0, loop_num);
    data_distributor_single(dLowerFiltered_buffer1, dLowerFiltered_out1, loop_num);
    
    data_distributor_single(hasUpper_buffer0, hasUpper_out0, loop_num);
    data_distributor_single(hasUpper_buffer1, hasUpper_out1, loop_num);
    
    data_distributor_single(dUpperFiltered_buffer0, dUpperFiltered_out0, loop_num);
    data_distributor_single(dUpperFiltered_buffer1, dUpperFiltered_out1, loop_num);
    
    data_distributor_single(colScale_buffer0, colScale_out0, loop_num);
    data_distributor_single(colScale_buffer1, colScale_out1, loop_num);
}