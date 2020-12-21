#ifndef MEMORY_ACCESS_H
#define MEMORY_ACCESS_H

#include "../../src_hw/parameters.h"
#include <hls_stream.h>

extern void order2FPGA_3x3(DATA_SW *in_mem, DATA_SW *out_mem, DATA_SW depth, DATA_SW length);
extern void order2FPGA_1x1(DATA_SW *in_mem, DATA_SW *out_mem, DATA_SW depth, DATA_SW length);
extern void order2CPU_map(DATA_SW *in_mem, DATA_SW *out_mem, DATA_SW depth, DATA_SW length, DATA_SW stride);
extern void DRAM_2_STREAM_3x3(DATA_SW *in_mem, hls::stream<DATA_STREAM> &out_mem_A, hls::stream<DATA_STREAM> &out_mem_B,
		hls::stream<DATA_STREAM> &out_mem_C, hls::stream<DATA_STREAM> &out_mem_D,
		DATA_SW depth_out, DATA_SW depth_in, DATA_SW length, DATA_SW type_layer);
extern void DRAM_2_STREAM_1x1(DATA_SW *in_mem, hls::stream<DATA_STREAM> &out_mem_A, hls::stream<DATA_STREAM> &out_mem_B,
		hls::stream<DATA_STREAM> &out_mem_C, hls::stream<DATA_STREAM> &out_mem_D,
		DATA_SW depth_out, DATA_SW depth_in, DATA_SW length, DATA_SW stride, DATA_SW type_layer, DATA_SW expansion);
extern void DRAM_2_STREAM_res(DATA_SW *in_mem, hls::stream<DATA_STREAM> &out_mem_A, hls::stream<DATA_STREAM> &out_mem_B,
		hls::stream<DATA_STREAM> &out_mem_C, hls::stream<DATA_STREAM> &out_mem_D,
		DATA_SW depth, DATA_SW length, DATA_SW residual, DATA_SW store_map);
extern void DRAM_2_STREAM_array(DATA_SW *in_mem, hls::stream<DATA_STREAM> &out_mem_A, hls::stream<DATA_STREAM> &out_mem_B,
		hls::stream<DATA_STREAM> &out_mem_C, hls::stream<DATA_STREAM> &out_mem_D,
		DATA_SW length_out, DATA_SW length_in);
extern void STREAM_2_DRAM_3x3(hls::stream<DATA_STREAM> &in_mem_A, hls::stream<DATA_STREAM> &in_mem_B,
		hls::stream<DATA_STREAM> &in_mem_C, hls::stream<DATA_STREAM> &in_mem_D,
		DATA_SW *out_mem, DATA_SW depth, DATA_SW length, DATA_SW stride, DATA_SW expansion);
extern void STREAM_2_DRAM_1x1(hls::stream<DATA_STREAM> &in_mem_A, hls::stream<DATA_STREAM> &in_mem_B,
		hls::stream<DATA_STREAM> &in_mem_C, hls::stream<DATA_STREAM> &in_mem_D,
		DATA_SW *out_mem, DATA_SW depth, DATA_SW length, DATA_SW expansion);
extern void STREAM_2_DRAM_res(hls::stream<DATA_STREAM> &in_mem_A, hls::stream<DATA_STREAM> &in_mem_B,
		hls::stream<DATA_STREAM> &in_mem_C, hls::stream<DATA_STREAM> &in_mem_D,
		DATA_SW *out_mem, DATA_SW depth, DATA_SW length);
extern void STREAM_2_DRAM_array(hls::stream<DATA_STREAM> &in_mem_A, hls::stream<DATA_STREAM> &in_mem_B,
		hls::stream<DATA_STREAM> &in_mem_C, hls::stream<DATA_STREAM> &in_mem_D,
		DATA_SW *out_mem, DATA_SW depth);
extern void set_tile_info(DATA_SW *tile_3x3, DATA_SW *tile_convs, DATA_SW *tile_avg, DATA_SW *tile_fc,
				   DATA_SW *info_3x3, DATA_SW *info_convs, DATA_SW *info_avg, DATA_SW *info_fc);

#endif
