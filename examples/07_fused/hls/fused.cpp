#include <ap_int.h>
#include <cstring>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <hls_math.h>
#include <hls_vector.h>
#include <cmath>
#include <cstdlib>


typedef double pdlp_float;

typedef hls::stream<ap_uint<512>, 16> axis_stream_512;

typedef hls::vector<double, 8> v8double;
typedef hls::stream<v8double, 64> axis_stream_v8double;

/*
    because the data type of the input and output of the kernel is double,
    and the operations include multiplication and addition,
    so use the hls::stream<v8double, 64> to represent the input and output of the kernel.
*/

void read(v8double* x_in, axis_stream_v8double &read_s, int len){

    for(int i = 0; i < len; i++){
        #pragma HLS PIPELINE II=1
        read_s.write(x_in[i]);
    }
    
}

void axpy(axis_stream_v8double &x, axis_stream_v8double &aty, 
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

void projlub(axis_stream_v8double &tempin, axis_stream_v8double &lb, axis_stream_v8double &ub, axis_stream_v8double &xupdate, int len){

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

void stage1(v8double* x_in, v8double* aty, v8double* cost, axis_stream_v8double &tempout, double alpha, int len) {

    #pragma HLS DATAFLOW
    axis_stream_v8double xin;
    axis_stream_v8double aty_s;
    axis_stream_v8double cost_s;

    read(x_in, xin, len);
    read(aty, aty_s, len);
    read(cost, cost_s, len);

    axpy(xin, aty_s, cost_s, tempout, alpha, len);

}

void stage2(axis_stream_v8double &tempin, v8double* lb, v8double* ub, v8double* xUpdate, int len) {

    #pragma HLS DATAFLOW
    axis_stream_v8double lb_s;
    axis_stream_v8double ub_s;
    axis_stream_v8double xupdate;

    read(lb, lb_s, len);
    read(ub, ub_s, len);

    projlub(tempin, lb_s, ub_s, xupdate, len);

    write(xupdate, xUpdate, len);

}


void fused(v8double* x_in, v8double* xUpdate, v8double* cost, v8double* aty, double alpha, 
    v8double* lb, v8double* ub, int len, int n){
    #pragma HLS INTERFACE m_axi port=x_in offset=slave bundle=gmem0 
    #pragma HLS INTERFACE s_axilite port=x_in bundle=control
    #pragma HLS INTERFACE m_axi port=xUpdate offset=slave bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=xUpdate bundle=control
    #pragma HLS INTERFACE m_axi port=cost offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=cost bundle=control
    #pragma HLS INTERFACE m_axi port=aty offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=aty bundle=control
    #pragma HLS INTERFACE m_axi port=lb offset=slave bundle=gmem4
    #pragma HLS INTERFACE s_axilite port=lb bundle=control
    #pragma HLS INTERFACE m_axi port=ub offset=slave bundle=gmem5
    #pragma HLS INTERFACE s_axilite port=ub bundle=control
    #pragma HLS INTERFACE s_axilite port=alpha bundle=control
    #pragma HLS INTERFACE s_axilite port=len bundle=control
    #pragma HLS interface s_axilite bundle=control port=n
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    #pragma HLS DATAFLOW   
     
    len = (len + 7) / 8;

    axis_stream_v8double temp;
    
    stage1(x_in, aty, cost, temp, alpha, len);

    stage2(temp, lb, ub, xUpdate, len);
}


// This project aims to integrate the four small operators in the PDHG_primalGradientStep function.

/*

void PDHG_primalGradientStep(CUPDLPwork *work, CUPDLPvec *xUpdate,
                             const CUPDLPvec *x, const CUPDLPvec *ATy,
                             cupdlp_float dPrimalStepSize) {
    CUPDLPproblem *problem = work->problem;

    CUPDLP_COPY_VEC(xUpdate->data, x->data, cupdlp_float, problem->nCols);

    // AddToVector(xUpdate, -dPrimalStepSize, problem->cost, problem->nCols);
    // AddToVector(xUpdate, dPrimalStepSize, ATy, problem->nCols);

    cupdlp_float alpha = -dPrimalStepSize;
    cupdlp_axpy(work, problem->nCols, &alpha, problem->cost, xUpdate->data);
    
    alpha = dPrimalStepSize;
    cupdlp_axpy(work, problem->nCols, &alpha, ATy->data, xUpdate->data);

    cupdlp_projub(xUpdate->data, problem->upper, problem->nCols);
    cupdlp_projlb(xUpdate->data, problem->lower, problem->nCols);

}


*/
