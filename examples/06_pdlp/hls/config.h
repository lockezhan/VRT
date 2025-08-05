#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "hls_task.h"
#include "hls_vector.h"

#define PLIO_WIDTH 128

typedef ap_axiu<PLIO_WIDTH, 0, 0, 0> aie_axis_data;
typedef hls::stream<aie_axis_data> aie_axis_stream;

typedef hls::stream<ap_uint<128>> axis_stream_128;

typedef hls::stream<float> axis_stream_float;

typedef double pdlp_float;

typedef hls::vector<double, 8> v8double;
typedef hls::stream<v8double, 64> axis_stream_v8double;

typedef union{
    float data_cbuff;
    unsigned int uintval;
} fp_int;

typedef struct{
    ap_uint<8> low0;
    ap_uint<8> low1;
    ap_uint<8> high0;
    ap_uint<8> high1;
 } comb_32;

void Vadd(
    ap_uint<512>* a,
    ap_uint<512>* b,
    ap_uint<512>* c,
    int n
);

#endif