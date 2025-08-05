#include "config.h"

void read(v8double* x_in, axis_stream_v8double &read_s, int len){

    for(int i = 0; i < len; i++){
        #pragma HLS PIPELINE II=1
        read_s.write(x_in[i]);
    }
    
}

void axpy_core(axis_stream_v8double &x, axis_stream_v8double &aty, 
    axis_stream_v8double &cost, axis_stream_v8double &tempout, double alpha, int len){
    
    // const ap_uint<64> NEG_ZERO = 0x80000000; // IEEE 754 -0.0

    for(int i =0; i < len; i++){
        #pragma HLS PIPELINE II=1
        v8double datain_x = x.read();
        v8double datain_aty = aty.read();
        v8double datain_cost = cost.read();
        v8double dataout;
        for (int j = 0; j < 8; j++) {
             #pragma HLS UNROLL factor=8
            dataout[j] = datain_x[j] - alpha * (datain_cost[j] - datain_aty[j]);
        }
        tempout.write(dataout);
    }

}


void axpy(v8double* x_in, v8double* aty, v8double* cost, axis_stream_v8double &tempout, double alpha, int len) {
    #pragma HLS INTERFACE m_axi port=x_in offset=slave bundle=gmem0 max_read_burst_length=64 num_write_outstanding=64 
    #pragma HLS INTERFACE s_axilite port=x_in bundle=control
    #pragma HLS INTERFACE m_axi port=cost offset=slave bundle=gmem1 max_read_burst_length=64 num_write_outstanding=64 
    #pragma HLS INTERFACE s_axilite port=cost bundle=control
    #pragma HLS INTERFACE m_axi port=aty offset=slave bundle=gmem2 max_read_burst_length=64 num_write_outstanding=64 
    #pragma HLS INTERFACE s_axilite port=aty bundle=control
    #pragma HLS INTERFACE s_axilite port=alpha bundle=control
    #pragma HLS INTERFACE mode=axis port=tempout
    #pragma HLS INTERFACE s_axilite port=len bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    #pragma HLS DATAFLOW

    len = (len + 7) / 8;

    axis_stream_v8double xin;
    axis_stream_v8double aty_s;
    axis_stream_v8double cost_s;

    read(x_in, xin, len);
    read(aty, aty_s, len);
    read(cost, cost_s, len);

    axpy_core(xin, aty_s, cost_s, tempout, alpha, len);

}


