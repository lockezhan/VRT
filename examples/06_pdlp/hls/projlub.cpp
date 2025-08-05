#include "config.h"

void read(v8double* x_in, axis_stream_v8double &read_s, int len){

    for(int i = 0; i < len; i++){
        #pragma HLS PIPELINE II=1
        read_s.write(x_in[i]);
    }
    
}


void projlub_core(axis_stream_v8double &tempin, axis_stream_v8double &lb, axis_stream_v8double &ub, axis_stream_v8double &xupdate, int len){

    for(int i = 0; i < len; i++){
        #pragma HLS PIPELINE II=1
        v8double datain = tempin.read();
        v8double lbin = lb.read();
        v8double ubin = ub.read();

        v8double dataout;
        for(int j = 0; j < 8; j++){
            #pragma HLS UNROLL factor=8
            dataout[j] = (datain[j] > ubin[j]) ? ubin[j] : datain[j];
            dataout[j] = (dataout[j] < lbin[j]) ? lbin[j] : dataout[j];
        }
        xupdate.write(dataout);
    }

}

void write(axis_stream_v8double &write_s, v8double *x_out, int len){
    for(int i = 0; i < len; i++){
        #pragma HLS PIPELINE II=1
        x_out[i] = write_s.read();
    }
}

void projlub(axis_stream_v8double &tempin, v8double* lb, v8double* ub, v8double* xUpdate, int len) {

    #pragma HLS INTERFACE m_axi port=lb offset=slave bundle=gmem0 max_read_burst_length=64 num_write_outstanding=64 
    #pragma HLS INTERFACE s_axilite port=lb bundle=control
    #pragma HLS INTERFACE m_axi port=ub offset=slave bundle=gmem1 max_read_burst_length=64 num_write_outstanding=64 
    #pragma HLS INTERFACE s_axilite port=ub bundle=control
    #pragma HLS INTERFACE m_axi port=xUpdate offset=slave bundle=gmem2 max_read_burst_length=64 num_write_outstanding=64 
    #pragma HLS INTERFACE s_axilite port=xUpdate bundle=control
    #pragma hls interface axis port=tempin

    #pragma HLS INTERFACE s_axilite port=len bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    #pragma HLS DATAFLOW

    len = (len + 7) / 8;

    axis_stream_v8double lb_s;
    axis_stream_v8double ub_s;
    axis_stream_v8double xupdate;

    read(lb, lb_s, len);
    read(ub, ub_s, len);

    projlub_core(tempin, lb_s, ub_s, xupdate, len);
    // hls::task projlub_task(projlub, tempin, lb_s, ub_s, xupdate, len);

    write(xupdate, xUpdate, len);

}

