#ifndef MEMORY_ACCESS_H
#define MEMORY_ACCESS_H

#include "type_data.h"

extern void order2CPU_map(DATA *in_mem, DATA *out_mem, DATA depth, DATA length, DATA stride);
extern DATA DRAM_2_STREAM_3x3(DATA *in_mem, DATA *out_mem,
		DATA depth_out, DATA depth_in, DATA length, DATA type_layer);
extern DATA DRAM_2_STREAM_1x1(DATA *in_mem, DATA *out_mem,
		DATA depth_out, DATA depth_in, DATA length, DATA type_layer, DATA expansion);
extern DATA DRAM_2_STREAM_res(DATA depth, DATA length, DATA residual, DATA store_map);
extern DATA DRAM_2_STREAM_array(DATA *in_mem, DATA *out_mem,
		DATA length_out, DATA length_in);
extern void STREAM_2_DRAM_3x3(DATA *in_mem, DATA *out_mem, DATA depth, DATA length, DATA stride);
extern void STREAM_2_DRAM_1x1(DATA *in_mem, DATA *out_mem, DATA depth, DATA length);
extern void STREAM_2_DRAM_res(DATA *in_mem_A, DATA *in_mem_B, DATA *in_mem_C, DATA *in_mem_D,
		DATA *out_mem, DATA depth, DATA length);
extern void STREAM_2_DRAM_array(DATA *in_mem_A, DATA *in_mem_B, DATA *in_mem_C, DATA *in_mem_D,
		DATA *out_mem, DATA depth);
extern void set_tile_info(DATA *tile_3x3, DATA *tile_convs, DATA *tile_avg, DATA *tile_fc,
				   	   	  DATA *info_3x3, DATA *info_convs, DATA *info_avg, DATA *info_fc);

#endif
