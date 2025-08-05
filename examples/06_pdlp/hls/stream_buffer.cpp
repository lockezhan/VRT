#include "config.h"

void stream_buffer(axis_stream_v8double &stream_in, axis_stream_v8double &stream_out) {
    #pragma HLS INTERFACE axis port=stream_in
    #pragma HLS INTERFACE axis port=stream_out
    #pragma HLS INTERFACE ap_ctrl_none port=return  // 无控制信号，持续运行

    v8double data;
    while (true) {
        #pragma HLS PIPELINE II=1
        if (!stream_in.empty()) {
            data = stream_in.read();
            stream_out.write(data);
        }
    }
}
