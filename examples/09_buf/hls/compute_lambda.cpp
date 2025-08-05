#include "base_kernel.hpp"

void compute_lambda(
    hls::stream<vec_double,64>& aty, 
    hls::stream<vec_double,64>& cost,
    hls::stream<vec_double,64>& hasLower,
    hls::stream<vec_double,64>& dLowerFiltered,
    hls::stream<vec_double,64>& hasUpper,
    hls::stream<vec_double,64>& dUpperFiltered,
    hls::stream<vec_double,64>& colScale,
    double *res,
    int nCols
){
    // HLS 接口配置

    #pragma HLS interface m_axi offset=slave bundle=gmem0 port=res

    #pragma HLS interface axis port=aty
    #pragma HLS interface axis port=cost
    #pragma HLS interface axis port=hasLower
    #pragma HLS interface axis port=dLowerFiltered
    #pragma HLS interface axis port=hasUpper
    #pragma HLS interface axis port=dUpperFiltered
    #pragma HLS interface axis port=colScale
    #pragma HLS interface s_axilite bundle=control port=nCols
    #pragma HLS interface s_axilite bundle=control port=res
    #pragma HLS interface s_axilite bundle=control port=return
    
    #pragma HLS DATAFLOW    
    int loop_num = (nCols + VEC_SIZE - 1) / VEC_SIZE;
    int tail_num = nCols % VEC_SIZE;

    hls::stream<vec_double,64> dualResidual_stream0, dualResidual_stream1, dualResidual_stream2, dualResidual_stream3, dualResidual_stream4;
    hls::stream<vec_double,64> dSlackPos_stream0, dSlackPos_stream1, dSlackPos_stream2, dSlackPos_stream3;
    hls::stream<vec_double,64> dSlackNeg_stream0, dSlackNeg_stream1, dSlackNeg_stream2, dSlackNeg_stream3, dSlackNeg_stream4;

    hls::stream<double> dPrimalObj1;
    hls::stream<double> dPrimalObj2;
    hls::stream<double> dDualFeasibility;

    cupdlp_scaleVector(-1.0, aty, dualResidual_stream1, loop_num);
    cupdlp_axpy1(1.0, cost, dualResidual_stream1, dSlackPos_stream0, dSlackNeg_stream0, dualResidual_stream2, loop_num);

    cupdlp_projPos(dSlackPos_stream0, dSlackPos_stream1, loop_num);
    cupdlp_edot1(dSlackPos_stream1, hasLower, dSlackPos_stream2, dSlackPos_stream3, loop_num);
    cupdlp_dot(dSlackPos_stream2, dLowerFiltered, dPrimalObj1, loop_num);

    cupdlp_projNeg(dSlackNeg_stream0, dSlackNeg_stream1, loop_num);
    cupdlp_scaleVector(-1.0, dSlackNeg_stream1, dSlackNeg_stream2, loop_num);
    cupdlp_edot1(dSlackNeg_stream2, hasUpper, dSlackNeg_stream3, dSlackNeg_stream4, loop_num);
    cupdlp_dot(dSlackNeg_stream3, dUpperFiltered, dPrimalObj2, loop_num);

    cupdlp_axpy2(-1.0, 1.0, dualResidual_stream2, dSlackPos_stream3, dSlackNeg_stream4, dualResidual_stream3, loop_num);
    cupdlp_edot(dualResidual_stream3, colScale, dualResidual_stream4, loop_num);
    cupdlp_twoNorm(dualResidual_stream4, dDualFeasibility, loop_num);

    write_ddr1(res, dPrimalObj1, dPrimalObj2, dDualFeasibility);
}