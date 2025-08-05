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

void cupdlp_twoNorm(
    hls::stream<vec_double,64>& x,
    hls::stream<double>& result,
    int N
){
        // Shift register based accumulation
    const int DELAY = 6;  // Match your floating point adder latency
    
    double shift_reg[8][DELAY + 1];
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=0
    
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

    final_loop:
    for(int i = 0; i < 8; i++){
        final_inner_loop:
        for(int j = 0; j <= DELAY; j++){
            res += shift_reg[i][j];
        }
    }
    
    result.write(res);
}