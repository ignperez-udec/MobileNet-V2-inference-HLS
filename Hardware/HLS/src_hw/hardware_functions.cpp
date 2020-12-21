#include <hls_stream.h>
#include "parameters.h"
#include <ap_axi_sdata.h>

// HARDWARE FUNCTIONS
void layer_CONV_3x3(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
					volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
                    DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer);
void layer_expansion_projection(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
								hls::stream<DATA_STREAM> &ext_residual_map_read, hls::stream<DATA_STREAM> &ext_residual_map_write,
								volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
                                DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer);
void layer_depthwise(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
					 volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
					 DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer);
void layer_AVG(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
			   DATA_HW tile_index[3], DATA_HW info[size_info], DATA_HW quant[3], DATA_HW type_layer);
void layer_FC(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
			  volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
			  DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer);
ACT_CONV convolution_1x1(ACT_CONV in_map, W_CONV kernel);
ACT_CONV ReLU6(ACT_CONV in_map, DATA_HW upper);
DATA_HW MIN(DATA_HW x, DATA_HW y);
void read_w_conv(volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
		W_CONV w_conv[w_conv_LEN], I_CONV i_conv[w_conv_LEN], ACT_CONV b_conv[tile_conv_out],
		DATA_HW tile_index[3], DATA_HW info[size_info]);
void read_w_fc(volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
		W_FC w_fc[w_fc_LEN], I_FC i_fc[w_fc_LEN], ACT_FC b_fc[tile_fc_out],
		DATA_HW LEN_W, DATA_HW LEN_B, DATA_HW info[size_info], DATA_HW type_layer);
void generate_quant(DATA_SW layer, DATA_SW inter_layer, DATA_SW type_layer,
					DATA_HW quant[4]);
void generate_info_tile(volatile DATA_SW *ext_tile, volatile DATA_SW *ext_info,
						DATA_HW tile_index[3], DATA_HW info[size_info],
						CALL_DATA call_PE, DATA_HW type_layer);
void generate_tile(volatile DATA_SW *ext_tile,
				   DATA_HW tile_index[3],
				   CALL_DATA call_PE, DATA_HW type_layer);
void generate_info(volatile DATA_SW *ext_info,
				   DATA_HW info[size_info],
				   CALL_DATA call_PE, DATA_HW type_layer);
void PEs(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
		 hls::stream<DATA_STREAM> &ext_residual_map_read, hls::stream<DATA_STREAM> &ext_residual_map_write,
		 volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
		 volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
		 DATA_HW tile_index[3], DATA_SW quant[4], DATA_SW info[size_info], CALL_DATA call_PE, DATA_SW type_layer);
void MobileNet_Stream(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data, hls::stream<DATA_STREAM> &ext_residual_map_write, hls::stream<DATA_STREAM> &ext_residual_map_read,
						volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
						volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
						volatile DATA_SW *ext_tile, volatile DATA_SW *ext_info,
						DATA_HW layer, DATA_SW inter_layer, DATA_SW type_layer);

//
//HARDWARE FUNCTIONS
//

//TILING

void layer_CONV_3x3(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
					volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
                    DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer){

	//FIFOs
	ACT_CONV line_buffer[2][in_map_LEN];
	ACT_CONV window[3][3], reg[3];
	bool stride_x = (info[3] == 1), stride_y = (info[3] == 1);   //strides
	W_CONV kernel[9];   //kernel
	ACT_CONV temp;   //conv
	ACT_CONV bias_value;   //bias
	DATA_HW limit = info[1]*info[1];

	#pragma HLS ARRAY_PARTITION variable=line_buffer dim=1 complete
	#pragma HLS ARRAY_PARTITION variable=windows complete
	#pragma HLS ARRAY_PARTITION variable=reg complete

	ACT_CONV temp_map[in_map_LEN][in_map_LEN];
	DATA_STREAM valIn_data, valOut_data;
	DATA_HW index = 0, len = info[0]*info[0]*tile_index[0];

	#pragma HLS ARRAY_PARTITION variable=kernel complete

	bias_value = (ACT_CONV) ((EXT_DATA) ext_b_conv[info[14]]).range(upper_act_CONV, 0);

	LOOP_depth_3X3: for (DATA_HW i_tile = 0, j_tile = 0; i_tile < tile_index[0]; j_tile++){
		#pragma HLS LOOP_TRIPCOUNT min=24 max=24
		#pragma HLS loop_flatten off
		//Define kernel
		LOOP_kernel_3X3: for (DATA_HW k = 0, pos_kernel = j_tile*9 + i_tile*tile_index[1]*9; k < 9; k++, pos_kernel++){
			#pragma HLS LOOP_TRIPCOUNT min=9 max=9
			#pragma HLS PIPELINE II=1
			kernel[k] = ((EXT_DATA) ext_w_conv[pos_kernel + info[15]]).range(lower_index_CONV, 0);
		}
		//read first values
		DATA_HW read_count = info[1] + 2;
		LOOP_line_buffer_0_3X3: for (DATA_HW l_tile = info[1] - 2; l_tile < info[1]; l_tile++){
			#pragma HLS LOOP_TRIPCOUNT min=30 max=30
			#pragma HLS PIPELINE II=1
			valIn_data = ext_in_data.read();
			temp = (ACT_CONV) (valIn_data.data >> quant[0]);
			line_buffer[0][l_tile] = temp;
		}
		LOOP_line_buffer_1_3X3: for (DATA_HW k_tile = 1; k_tile < 2; k_tile++){
			for (DATA_HW l_tile = 0; l_tile < info[1]; l_tile++){
				#pragma HLS LOOP_TRIPCOUNT min=30 max=30
				#pragma HLS PIPELINE II=1
				valIn_data = ext_in_data.read();
				temp = (ACT_CONV) (valIn_data.data >> quant[0]);
				line_buffer[k_tile][l_tile] = temp;
			}
		}
		//copy values in window
		LOOP_window_3X3: for (DATA_HW x = 1; x < 3; x++){
			for (DATA_HW y = 1; y < 3; y++){
				#pragma HLS PIPELINE II=1
				window[x][y] = line_buffer[x-1][y + info[1] - 3];
			}
		}
		//Process Map_features
		LOOP_map_k_3X3: for (DATA_HW k_tile = 0; k_tile < info[1]; k_tile++){
			#pragma HLS LOOP_TRIPCOUNT min=30 max=30
			#pragma HLS loop_flatten off
			LOOP_map_l_3X3: for (DATA_HW l_tile = 0; l_tile < info[1]; l_tile++){
				#pragma HLS LOOP_TRIPCOUNT min=30 max=30
				#pragma HLS dependence variable=temp_map inter false
				#pragma HLS dependence variable=reg inter false
				#pragma HLS loop_flatten off
				#pragma HLS PIPELINE II=1
				if (k_tile > 0 && k_tile < info[10] && l_tile > 0 && l_tile < info[10] && stride_x && stride_y){
					//convolution
					temp = 0;
					LOOP_conv_x_3X3:for (DATA_HW x = 0, pos_x = k_tile-1; x < 3; x++, pos_x++){
						#pragma HLS LOOP_TRIPCOUNT min=3 max=3
						if ((pos_x > 0 || info[6] == 0) && (pos_x < info[10]  || info[7] == 0)){
							LOOP_conv_y_3X3: for (DATA_HW y = 0, pos_y = l_tile-1, pos_kernel = x*3; y < 3; y++, pos_y++){
								#pragma HLS LOOP_TRIPCOUNT min=3 max=3
								if ((pos_y > 0 || info[8] == 0) && (pos_y < info[10] || info[9] == 0)){
									temp += (window[x][y]) * kernel[y + pos_kernel];
								}
							}
						}
					}
					if (info[2] && j_tile == 0 && type_layer == 0){
						temp_map[k_tile][l_tile] = temp;
					}
					else if (info[4] && j_tile == info[5]){
						//bias
						temp += bias_value + temp_map[k_tile][l_tile];
						//ReLU6
						temp = ReLU6(temp, quant[2]);
						//write out_data
						valOut_data.data = (DATA_HW) temp;
						valOut_data.keep = valIn_data.keep; valOut_data.strb = valIn_data.strb; valOut_data.user = valIn_data.user;
						valOut_data.id = valIn_data.id; valOut_data.dest = valIn_data.dest;
						if (info[16] && index + 1 == len){
							valOut_data.last = 1;
						}
						else{
							valOut_data.last = 0;
						}
						ext_out_data.write(valOut_data);
						index++;
					}
					else{
						temp_map[k_tile][l_tile] += temp;
					}
				}
				//shift line_buffer
				reg[0] = line_buffer[0][l_tile];
				reg[1] = line_buffer[0][l_tile] = line_buffer[1][l_tile];
				//read in_data
				if (read_count < limit){
					valIn_data = ext_in_data.read();
					read_count++;
					temp = (ACT_CONV) (valIn_data.data >> quant[0]);
				}
				reg[2] = line_buffer[1][l_tile] = temp;
				//shift window
				LOOP_shift_window_x_3X3: for (DATA_HW x = 0; x < 3; x++){
					LOOP_shift_window_y_3X3: for (DATA_HW y = 0; y < 2; y++){
						window[x][y] = window[x][y + 1];
					}
					window[x][2] = reg[x];
				}
				//stride x
				if (info[3] == 2){
					stride_x = !stride_x;
				}
			}
			//stride y
			if (info[3] == 2){
				stride_y = !stride_y;
			}
		}
		//New index
		if (j_tile + 1 >= tile_index[1]){
			j_tile = -1;
			i_tile++;
			bias_value = (ACT_CONV) ((EXT_DATA) ext_b_conv[i_tile + info[14]]).range(upper_act_CONV, 0);
		}
	}
}

void layer_expansion_projection(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
								hls::stream<DATA_STREAM> &ext_residual_map_read, hls::stream<DATA_STREAM> &ext_residual_map_write,
								volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
                                DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer){

	static ACT_CONV temp_map[tile_conv_out][out_map_LEN][out_map_LEN];
	ACT_CONV in_map[tile_conv_out][out_map_LEN][out_map_LEN];
	ACT_CONV temp;   //convolution
	W_CONV kernel_value;   //weight value
	ACT_CONV bias_value;   //bias
	ACT_CONV res_map_data;   //residual
	DATA_STREAM valIn_data, valOut_data, valIn_res, valOut_res;   //stream
	DATA_HW len = tile_index[2]*tile_index[2]*tile_index[0], read_res_info = 1, index = 0;
	DATA_HW read_data[tile_conv_in] = {0};   //read data

	#pragma HLS ARRAY_PARTITION variable=temp_map dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=in_map dim=3 complete

	//weights
	W_CONV w_conv[w_conv_LEN];
	ACT_CONV b_conv[tile_conv_out];
	I_CONV i_conv[w_conv_LEN];

	#pragma HLS RESOURCE variable=w_conv core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=b_conv core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=i_conv core=XPM_MEMORY uram

	//read weights
	read_w_conv(ext_w_conv, ext_b_conv, w_conv, i_conv, b_conv, tile_index, info);
	bias_value = b_conv[0];   //bias

	LOOP_depth_expproj: for (DATA_HW i_tile = 0, j_tile = 0, pos_weight = 0, distance = 0; i_tile < tile_index[0]; j_tile ++){   //out_map_depth, in_map_depth, length_map
		#pragma HLS LOOP_TRIPCOUNT min=96 max=1024
		#pragma HLS loop_flatten off
		//Process
		if (distance == i_conv[pos_weight]){
			kernel_value = w_conv[pos_weight];
			if (info[5] && j_tile == info[6]){
				//Loop length map
				LOOP_mapk_p1_expproj: for (DATA_HW k_tile = 0; k_tile < tile_index[2]; k_tile++){   //length_in_map
					#pragma HLS LOOP_TRIPCOUNT min=7 max=28
					LOOP_mapl_p1_expproj: for (DATA_HW l_tile = 0; l_tile < HLS_tile_map; l_tile++){   //length_in_map
						#pragma HLS dependence variable=out_map inter false
						#pragma HLS dependence variable=temp_map inter false
						#pragma HLS PIPELINE II=1
						//CONV
						if (l_tile < tile_index[2]){
							//CONV
							if (read_data[j_tile] == 0){
								//read in data
								valIn_data = ext_in_data.read();
								temp = (ACT_CONV) (valIn_data.data >> quant[0]);
								in_map[j_tile][k_tile][l_tile] = temp;
								temp = temp_map[i_tile][k_tile][l_tile] + convolution_1x1(temp, kernel_value);
							}
							else{
								temp = temp_map[i_tile][k_tile][l_tile] + convolution_1x1(in_map[j_tile][k_tile][l_tile], kernel_value);
							}
							//bias
							temp += bias_value;
							//ReLU6
							if (info[0]){
								temp = ReLU6(temp, quant[2]);
								//write out_data
								valOut_data.data = (DATA_HW) temp;
								valOut_data.keep = valIn_data.keep; valOut_data.strb = valIn_data.strb; valOut_data.user = valIn_data.user;
								valOut_data.id = valIn_data.id; valOut_data.dest = valIn_data.dest;
								if (info[16] && index + 1 == len){
									valOut_data.last = 1;
								}
								else{
									valOut_data.last = 0;
								}
								ext_out_data.write(valOut_data);
							}
							//Residual layer
							else if (info[1]){
								//read res
								valIn_res = ext_residual_map_read.read();
								res_map_data = (ACT_CONV) valIn_res.data;
								temp += res_map_data;
								//write res
								if (info[2]){
									if (quant[3] >= 0){
										valOut_res.data = (DATA_HW) (temp << quant[3]);
									}
									else{
										valOut_res.data = (DATA_HW) (temp >> -quant[3]);
									}
									valOut_res.keep = valIn_res.keep; valOut_res.strb = valIn_res.strb; valOut_res.user = valIn_res.user;
									valOut_res.id = valIn_res.id; valOut_res.dest = valIn_res.dest;
									if (info[16] && index + 1 == len){
										valOut_res.last = 1;
									}
									else{
										valOut_res.last = 0;
									}
									ext_residual_map_write.write(valOut_res);
								}
								//write out_data
								valOut_data.data = (DATA_HW) temp;
								valOut_data.keep = valIn_data.keep; valOut_data.strb = valIn_data.strb; valOut_data.user = valIn_data.user;
								valOut_data.id = valIn_data.id; valOut_data.dest = valIn_data.dest;
								if (info[16] && index + 1 == len){
									valOut_data.last = 1;
								}
								else{
									valOut_data.last = 0;
								}
								ext_out_data.write(valOut_data);
							}
							else{
								//write res
								if (info[2]){
									//read direction to write residual
									if (read_res_info){
										valIn_res = ext_residual_map_read.read();
										read_res_info = 0;
									}
									if (quant[3] >= 0){
										valOut_res.data = (DATA_HW) (temp << quant[3]);
									}
									else{
										valOut_res.data = (DATA_HW) (temp >> -quant[3]);
									}
									valOut_res.keep = valIn_res.keep; valOut_res.strb = valIn_res.strb; valOut_res.user = valIn_res.user;
									valOut_res.id = valIn_res.id; valOut_res.dest = valIn_res.dest;
									if (info[16] && index + 1 == len){
										valOut_res.last = 1;
									}
									else{
										valOut_res.last = 0;
									}
									ext_residual_map_write.write(valOut_res);
								}
								//write out_data
								valOut_data.data = (DATA_HW) temp;
								valOut_data.keep = valIn_data.keep; valOut_data.strb = valIn_data.strb; valOut_data.user = valIn_data.user;
								valOut_data.id = valIn_data.id; valOut_data.dest = valIn_data.dest;
								if (info[16] && index + 1 == len){
									valOut_data.last = 1;
								}
								else{
									valOut_data.last = 0;
								}
								ext_out_data.write(valOut_data);
							}
							index++;
						}
					}
				}
				if (read_data[j_tile] == 0){
					read_data[j_tile] = 1;
				}
			}
			else{
				if (read_data[j_tile] == 0){
					//Loop length map
					LOOP_mapk_w1_expproj: for (DATA_HW k_tile = 0; k_tile < tile_index[2]; k_tile++){   //length_in_map
						#pragma HLS LOOP_TRIPCOUNT min=7 max=28
						LOOP_mapl_w1_expproj: for (DATA_HW l_tile = 0; l_tile < HLS_tile_map; l_tile++){   //length_in_map
							#pragma HLS dependence variable=out_map inter false
							#pragma HLS dependence variable=temp_map inter false
							#pragma HLS PIPELINE II=1
							//CONV
							if (l_tile < tile_index[2]){
								//read in data
								valIn_data = ext_in_data.read();
								temp = (ACT_CONV) (valIn_data.data >> quant[0]);
								in_map[j_tile][k_tile][l_tile] = temp;
								//Process
								if (info[4] && j_tile == 0){
									temp_map[i_tile][k_tile][l_tile] = convolution_1x1(temp, kernel_value);
								}
								else{
									temp_map[i_tile][k_tile][l_tile] += convolution_1x1(temp, kernel_value);
								}
							}
						}
					}
					read_data[j_tile] = 1;
				}
				else{
					//Loop length map
					LOOP_mapk_w0_expproj: for (DATA_HW k_tile = 0; k_tile < tile_index[2]; k_tile++){   //length_in_map
						#pragma HLS LOOP_TRIPCOUNT min=7 max=28
						#pragma HLS PIPELINE II=1
						LOOP_mapl_w0_expproj: for (DATA_HW l_tile = 0; l_tile < HLS_tile_map; l_tile++){   //length_in_map
							#pragma HLS dependence variable=out_map inter false
							#pragma HLS dependence variable=temp_map inter false
							//CONV
							if (l_tile < tile_index[2]){
								if (info[4] && j_tile == 0){
									temp_map[i_tile][k_tile][l_tile] = convolution_1x1(in_map[j_tile][k_tile][l_tile], kernel_value);
								}
								else{
									temp_map[i_tile][k_tile][l_tile] += convolution_1x1(in_map[j_tile][k_tile][l_tile], kernel_value);
								}
							}
						}
					}
				}
			}
			distance = 0;
			pos_weight++;
		}
		//Not enter in Loop length map
		else{
			distance ++;
			if (read_data[j_tile] == 0){
				//Loop length map
				LOOP_mapk_p0_expproj: for (DATA_HW k_tile = 0; k_tile < tile_index[2]; k_tile++){   //length_in_map
					#pragma HLS LOOP_TRIPCOUNT min=7 max=28
					LOOP_mapl_p0_expproj: for (DATA_HW l_tile = 0; l_tile < HLS_tile_map; l_tile++){   //length_in_map
						#pragma HLS PIPELINE II=1
						//CONV
						if (l_tile < tile_index[2]){
							//read in data
							valIn_data = ext_in_data.read();
							in_map[j_tile][k_tile][l_tile] = (ACT_CONV) (valIn_data.data >> quant[0]);
						}
					}
				}
				read_data[j_tile] = 1;
			}
		}
		//New index
		if (j_tile+1 == tile_index[1]){
			j_tile = -1;
			distance = 0;
			i_tile++;
			bias_value = b_conv[i_tile];
		}
	}
}


void layer_depthwise(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
					 volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
					 DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer){

	//FIFOs
	ACT_CONV line_buffer[2][in_map_LEN];
	ACT_CONV window[3][3], reg[3];
	bool stride_x = (info[3] == 1), stride_y = (info[3] == 1);   //strides
	W_CONV kernel[9];   //kernel
	ACT_CONV temp;   //conv
	ACT_CONV bias_value;   //bias
	DATA_STREAM valIn_data, valOut_data;
	DATA_HW index = 0, len = info[0]*info[0]*tile_index[0], limit = info[1]*info[1];

	#pragma HLS ARRAY_PARTITION variable=line_buffer dim=1 complete
	#pragma HLS ARRAY_PARTITION variable=windows complete
	#pragma HLS ARRAY_PARTITION variable=reg complete
	#pragma HLS ARRAY_PARTITION variable=kernel complete

	//Process
	LOOP_depth_depthwise: for (DATA_HW i_tile = 0, pos_kernel_base = 0; i_tile < tile_index[0]; i_tile++, pos_kernel_base += 9){
		#pragma HLS LOOP_TRIPCOUNT min=4 max=32
		#pragma HLS loop_flatten off
		//Define kernel
		LOOP_kernel_depthwise: for (DATA_HW k = 0, pos_kernel = pos_kernel_base; k < 9; k++, pos_kernel++){
			#pragma HLS LOOP_TRIPCOUNT min=9 max=9
			#pragma HLS PIPELINE II=1
			kernel[k] = ((EXT_DATA) ext_w_conv[pos_kernel + info[15]]).range(lower_index_CONV, 0);
			if (k == 0){
				bias_value = (ACT_CONV) ((EXT_DATA) ext_b_conv[i_tile + info[14]]).range(upper_act_CONV, 0);
			}
		}
		//read first values
		DATA_HW read_count = info[1] + 2;
		LOOP_line_buffer_0_depthwise: for (DATA_HW k_tile = info[1] - 2; k_tile < info[1]; k_tile++){
			#pragma HLS LOOP_TRIPCOUNT min=9 max=30
			#pragma HLS PIPELINE II=1
			valIn_data = ext_in_data.read();
			line_buffer[0][k_tile] = (ACT_CONV) (valIn_data.data >> quant[0]);
		}
		LOOP_line_buffer_1_depthwise: for (DATA_HW j_tile = 1; j_tile < 2; j_tile++){
			LOOP_line_buffer_2_depthwise: for (DATA_HW k_tile = 0; k_tile < info[1]; k_tile++){
				#pragma HLS LOOP_TRIPCOUNT min=9 max=30
				#pragma HLS PIPELINE II=1
				valIn_data = ext_in_data.read();
				line_buffer[j_tile][k_tile] = (ACT_CONV) (valIn_data.data >> quant[0]);
			}
		}
		//copy values in window
		LOOP_window_x_depthwise: for (DATA_HW x = 1; x < 3; x++){
			LOOP_window_y_depthwise: for (DATA_HW y = 1; y < 3; y++){
				#pragma HLS PIPELINE II=1
				window[x][y] = line_buffer[x-1][y + info[1] - 3];
			}
		}
		//Process Map_features
		LOOP_maps_depthwise: for (DATA_HW j_tile = 0, k_tile = 0; j_tile < info[1]; k_tile++){
			#pragma HLS LOOP_TRIPCOUNT min=81 max=900 //30
			#pragma HLS loop_flatten off
			#pragma HLS PIPELINE II=1
			if (j_tile > 0 && j_tile < info[10] && k_tile > 0 && k_tile < info[10] && stride_x && stride_y){
				//convolution
				temp = 0;
				LOOP_conv_x_depthwise: for (DATA_HW x = 0, pos_x = j_tile-1; x < 3; x++, pos_x++){
					#pragma HLS LOOP_TRIPCOUNT min=3 max=3
					if ((pos_x > 0 || info[6] == 0) && (pos_x < info[10]  || info[7] == 0)){
						LOOP_conv_y_depthwise: for (DATA_HW y = 0, pos_y = k_tile-1, pos_kernel = x*3; y < 3; y++, pos_y++){
							#pragma HLS LOOP_TRIPCOUNT min=3 max=3
							if ((pos_y > 0 || info[8] == 0) && (pos_y < info[10] || info[9] == 0)){
								temp += (window[x][y]) * kernel[y + pos_kernel];
							}
						}
					}
				}
				//bias
				temp += bias_value;
				//ReLU6
				temp = ReLU6(temp, quant[2]);
				//write out_data
				valOut_data.data = (DATA_HW) temp;
				valOut_data.keep = valIn_data.keep; valOut_data.strb = valIn_data.strb; valOut_data.user = valIn_data.user;
				valOut_data.id = valIn_data.id; valOut_data.dest = valIn_data.dest;
				if (info[16] && index + 1 == len){
					valOut_data.last = 1;
				}
				else{
					valOut_data.last = 0;
				}
				ext_out_data.write(valOut_data);
				index++;
			}
			//shift line_buffer
			reg[0] = line_buffer[0][k_tile];
			reg[1] = line_buffer[0][k_tile] = line_buffer[1][k_tile];
			//read in_data
			if (read_count < limit){
				valIn_data = ext_in_data.read();
				read_count++;
			}
			reg[2] = line_buffer[1][k_tile] = (ACT_CONV) (valIn_data.data >> quant[0]);
			//shift window
			LOOP_shift_window_x_depthwise: for (DATA_HW x = 0; x < 3; x++){
				LOOP_shift_window_y_depthwise: for (DATA_HW y = 0; y < 2; y++){
					window[x][y] = window[x][y + 1];
				}
				window[x][2] = reg[x];
			}
			//stride x
			if (info[3] == 2){
				stride_x = !stride_x;
			}
			//New index
			if (k_tile + 1 == info[1]){
				k_tile = -1;
				j_tile++;
				//stride y
				if (info[3] == 2){
					stride_y = !stride_y;
				}
			}
		}
	}
}

void layer_AVG(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
			   DATA_HW tile_index[3], DATA_HW info[size_info], DATA_HW quant[3], DATA_HW type_layer){

	#pragma HLS INLINE off

	ACT_FC temp = 0;
	ACT_CONV in_map_data;
	DATA_STREAM valIn_data, valOut_data;

	LOOP_avg: for (DATA_HW i_tile = 0, j_tile = 0, k_tile = 0; i_tile < tile_index[0]; k_tile ++){   //out_map_depth
		#pragma HLS LOOP_TRIPCOUNT min=7*7*32 max=7*7*32
		#pragma HLS PIPELINE
		//read in_data
		valIn_data = ext_in_data.read();
		in_map_data = (ACT_CONV) (valIn_data.data >> quant[0]);
		//Process
		temp += in_map_data;
		//New index
		if (k_tile+1 == tile_index[1]){
			k_tile = -1;
			j_tile++;
			if (j_tile == tile_index[1]){
				j_tile = 0;
				//write out_data
				valOut_data.data = (DATA_SW) temp/49;;
				valOut_data.keep = valIn_data.keep; valOut_data.strb = valIn_data.strb; valOut_data.user = valIn_data.user;
				valOut_data.id = valIn_data.id; valOut_data.dest = valIn_data.dest;
				if (info[9] && i_tile + 1 == tile_index[0]){
					valOut_data.last = 1;
				}
				else{
					valOut_data.last = 0;
				}
				ext_out_data.write(valOut_data);
				temp = 0;
				i_tile++;
			}
		}
	}
}

void layer_FC(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
			  volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
			  DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer){

	static ACT_FC temp_array[tile_fc_out];
	ACT_FC in_array[tile_fc_in];
	ACT_FC temp;   //FC
	DATA_STREAM valIn_data, valOut_data;

	//weights
	W_FC w_fc[w_fc_LEN];
	ACT_FC b_fc[tile_fc_out];
	I_FC i_fc[w_fc_LEN];

	#pragma HLS RESOURCE variable=w_fc core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=b_fc core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=i_fc core=XPM_MEMORY uram

	//read weights
	read_w_fc(ext_w_fc, ext_b_fc, w_fc, i_fc, b_fc, tile_index[0]*info[3], tile_index[0], info, type_layer);

	LOOP_fc: for (DATA_HW i_tile = 0, j_tile = 0, distance = 0, init_signal = 1, pos_weight = 0; i_tile < tile_index[0]; j_tile++){
		#pragma HLS LOOP_TRIPCOUNT min=32*64 max=32*64//HLS_tile_fc_out
		#pragma HLS PIPELINE II=1
		//read in_data
		if (i_tile == 0){
			valIn_data = ext_in_data.read();
			in_array[j_tile] = (ACT_FC) (valIn_data.data >> quant[0]);
		}
		//Process
		if (distance == i_fc[pos_weight]){
			//weight = w_fc[pos_weight];
			if (info[2] && init_signal){
				temp = (in_array[j_tile]) * w_fc[pos_weight];
				init_signal = 0;
			}
			else if (info[2] == 0 && init_signal){
				temp = temp_array[i_tile] + (in_array[j_tile]) * w_fc[pos_weight];
				init_signal = 0;
			}
			else{
				temp += (in_array[j_tile]) * w_fc[pos_weight];
			}
			pos_weight++;
			distance = 0;
		}
		else{
			distance++;
		}
		//New index
		if (j_tile+1 == tile_index[1]){
			j_tile = -1;
			distance = 0;
			init_signal = 1;
			//bias
			if (info[1] >= info[0]){
				//write out_data
				valOut_data.data = (DATA_SW) temp + b_fc[i_tile];
				valOut_data.keep = valIn_data.keep; valOut_data.strb = valIn_data.strb; valOut_data.user = valIn_data.user;
				valOut_data.id = valIn_data.id; valOut_data.dest = valIn_data.dest;
				if (info[9] && i_tile + 1 == tile_index[0]){
					valOut_data.last = 1;
				}
				else{
					valOut_data.last = 0;
				}
				ext_out_data.write(valOut_data);
				//out_array[i_tile] = temp + b_fc[i_tile];
			}
			else{
				//out_array[i_tile] = temp;
				temp_array[i_tile] = temp;
			}
			i_tile++;
		}
	}
}

//MATH

ACT_CONV convolution_1x1(ACT_CONV in_map, W_CONV kernel){
	#pragma HLS INLINE

	return in_map * kernel;
}

ACT_CONV ReLU6(ACT_CONV in_map, DATA_HW upper){
	#pragma HLS INLINE

    if(in_map <= 0){
        return 0;
    }
    else if(in_map >= upper){
        return upper;
    }
    else{
        return in_map;
    }
}

DATA_HW MIN(DATA_HW x, DATA_HW y){
	#pragma HLS INLINE

    if (x < y){
        return x;
    }
    else{
        return y;
    }
}


//
//Read AXI master
//

void read_w_conv(volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
		W_CONV w_conv[w_conv_LEN], I_CONV i_conv[w_conv_LEN], ACT_CONV b_conv[tile_conv_out],
		DATA_HW tile_index[3], DATA_HW info[size_info]){

	if (info[18]){
		DATA_HW temp;
		DATA_HW LEN_W = tile_index[0]*info[3];
		DATA_HW LEN_B = tile_index[0];
		//read W_CONV
		for (DATA_HW i = 0; i < LEN_W; i++){
			#pragma HLS LOOP_TRIPCOUNT min=96 max=1024
			#pragma HLS PIPELINE II=1
			temp = ext_w_conv[i + info[15]];
			w_conv[i] = ((EXT_DATA) temp).range(lower_index_CONV, 0);
			i_conv[i] = ((EXT_DATA) temp).range(upper_index_CONV, bit_w_CONV);
			if (i < LEN_B && info[17]){
				b_conv[i] = (ACT_CONV) ((EXT_DATA) ext_b_conv[i + info[14]]).range(upper_act_CONV, 0);
			}
		}
	}
}

void read_w_fc(volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
		W_FC w_fc[w_fc_LEN], I_FC i_fc[w_fc_LEN], ACT_FC b_fc[tile_fc_out],
		DATA_HW LEN_W, DATA_HW LEN_B, DATA_HW info[size_info], DATA_HW type_layer){

	DATA_HW temp;

	//read W_FC
	for (DATA_HW i = 0; i < LEN_W; i++){
		#pragma HLS LOOP_TRIPCOUNT max=32*64
		#pragma HLS PIPELINE II=1
		temp = ext_w_fc[i + info[6]];
		w_fc[i] = ((EXT_DATA) temp).range(lower_index_FC, 0);
		i_fc[i] = ((EXT_DATA) temp).range(upper_index_FC, bit_w_FC);
		if (i < LEN_B){
			b_fc[i] = (ACT_FC) ((EXT_DATA) ext_b_fc[i + info[5]]).range(upper_act_CONV, 0);
		}
	}
}

DATA_STREAM read_in_map(hls::stream<DATA_STREAM> &ext_in_data,
				 	 ACT_CONV in_map[tile_conv_out][in_map_LEN][in_map_LEN],
					 DATA_HW tile_index[3], DATA_HW info[size_info], DATA_HW quant, DATA_HW type_layer){

	DATA_STREAM valIn;
	DATA_HW len, limit;

	if (type_layer == 0){
		len = info[1]*info[1]*tile_index[1];
		limit = info[1];
	}
	else{
		len = info[1]*info[1]*tile_index[0];
		limit=info[1];
	}

	//read IN_MAP
	for (DATA_HW index = 0, i = 0, j = 0, k = 0; index <len; index++, k++){
		#pragma HLS LOOP_TRIPCOUNT min=3*HLS_tile_map*HLS_tile_map max=HLS_tile_conv_out*HLS_tile_map*HLS_tile_map
		#pragma HLS PIPELINE II=1
		if (k == limit){
			k = 0;
			j++;
			if (j == limit){
				j = 0;
				i++;
			}
		}
		valIn = ext_in_data.read();
		in_map[i][j][k] = (ACT_CONV) (valIn.data >> quant);
	}

	return valIn;
}

void generate_quant(DATA_SW layer, DATA_SW inter_layer, DATA_SW type_layer,
					DATA_HW quant[4]){

	if (type_layer < 3){
		for (DATA_HW i = 0; i < 4; i++){
			#pragma HLS PIPELINE II=1
			quant[i] = CONV_quant[layer][inter_layer][i];
		}
	}
	else if (type_layer == 3){
		quant[0] = AVG_quant;
	}
	else{
		for (DATA_HW i = 0; i < 2; i++){
			#pragma HLS PIPELINE II=1
			quant[i] = FC_quant[i];
		}
	}
}

void generate_info_tile(volatile DATA_SW *ext_tile, volatile DATA_SW *ext_info,
						DATA_HW tile_index[3], DATA_HW info[size_info],
						CALL_DATA call_PE, DATA_HW type_layer){

	if (call_PE < MAX_CALL[type_layer]){
		//tile
		generate_tile(ext_tile, tile_index, call_PE, type_layer);
		//info
		generate_info(ext_info, info, call_PE, type_layer);
	}
}

void generate_tile(volatile DATA_SW *ext_tile,
				   DATA_HW tile_index[3],
				   CALL_DATA call_PE, DATA_HW type_layer){
	DATA_HW offset = call_PE*3;

	for (DATA_HW i = 0; i < 3; i++){
		#pragma HLS PIPELINE
		tile_index[i] = (DATA_HW) ext_tile[offset + i];
	}
}

void generate_info(volatile DATA_SW *ext_info,
				   DATA_HW info[size_info],
				   CALL_DATA call_PE, DATA_HW type_layer){
	DATA_HW offset = call_PE*size_info;

	for (DATA_HW i = 0; i < size_info; i++){
		#pragma HLS PIPELINE
		info[i] = (DATA_HW) ext_info[offset + i];
	}
}


//
//PE function
//

void PEs(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
		 hls::stream<DATA_STREAM> &ext_residual_map_read, hls::stream<DATA_STREAM> &ext_residual_map_write,
		 volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
		 volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
		 DATA_HW tile_index[3], DATA_SW quant[4], DATA_SW info[size_info], CALL_DATA call_PE, DATA_SW type_layer){

	if (call_PE < MAX_CALL[type_layer]){
		if (type_layer == 0){
			layer_CONV_3x3(ext_in_data, ext_out_data, /*in_map, out_map,*/ ext_w_conv, ext_b_conv, tile_index, quant, info, type_layer);
		}
		else if (type_layer == 1){
			layer_expansion_projection(ext_in_data, ext_out_data, ext_residual_map_read, ext_residual_map_write, /*in_map, out_map, res_map,*/ ext_w_conv, ext_b_conv, tile_index, quant, info, type_layer);
		}
		else if (type_layer == 2){
			layer_depthwise(ext_in_data, ext_out_data, ext_w_conv, ext_b_conv, tile_index, quant, info, type_layer);
		}
		else if (type_layer == 3){
			layer_AVG(ext_in_data, ext_out_data, /*in_map, out_array,*/ tile_index, info, quant, type_layer);
		}
		else if (type_layer == 4){
			layer_FC(ext_in_data, ext_out_data, /*in_array, out_array,*/ ext_w_fc, ext_b_fc, tile_index, quant, info, type_layer);
		}
	}
}

//
//Top function
//

void MobileNet_Stream(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data, hls::stream<DATA_STREAM> &ext_residual_map_write, hls::stream<DATA_STREAM> &ext_residual_map_read,
		volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
		volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
		volatile DATA_SW *ext_tile, volatile DATA_SW *ext_info,
        DATA_HW layer, DATA_SW inter_layer, DATA_SW type_layer){

	#pragma HLS INTERFACE s_axilite port=layer bundle=CTRL_BUS
	#pragma HLS INTERFACE s_axilite port=inter_layer bundle=CTRL_BUS
	#pragma HLS INTERFACE s_axilite port=type_layer bundle=CTRL_BUS
	#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS
	#pragma HLS INTERFACE axis port=ext_in_data bundle=IN_MAP
	#pragma HLS INTERFACE axis port=ext_out_data bundle=OUT_MAP
	#pragma HLS INTERFACE axis port=ext_residual_map_read bundle=RESIDUAL_MAP_READ
	#pragma HLS INTERFACE axis port=ext_residual_map_write bundle=RESIDUAL_MAP_WRITE
	#pragma HLS INTERFACE m_axi port=ext_w_conv depth=w_conv_layer offset=slave bundle=W_CONV
	#pragma HLS INTERFACE m_axi port=ext_b_conv depth=b_conv_layer offset=slave bundle=B_CONV
	#pragma HLS INTERFACE m_axi port=ext_w_fc depth=fc_layer offset=slave bundle=W_FC
	#pragma HLS INTERFACE m_axi port=ext_b_fc depth=b_fc_layer offset=slave bundle=B_FC
	#pragma HLS INTERFACE m_axi port=ext_tile depth=HLS_tile_size offset=slave bundle=TILE
	#pragma HLS INTERFACE m_axi port=ext_info depth=HLS_info_size offset=slave bundle=INFO

	/*
	 * UNROLL
	 */

	//Register
	DATA_HW quant[4];

	#pragma HLS ARRAY_PARTITION variable=quant complete

	//PE 0
	//Register
	DATA_HW info_0[size_info], tile_index_0[3];

	#pragma HLS ARRAY_PARTITION variable=info_0 complete
	#pragma HLS ARRAY_PARTITION variable=tile_index_0 complete

	//PE 1
	//Register
	DATA_HW info_1[size_info], tile_index_1[3];

	#pragma HLS ARRAY_PARTITION variable=info_1 complete
	#pragma HLS ARRAY_PARTITION variable=tile_index_1 complete

	//Processing
	//quant
	generate_quant(layer, inter_layer, type_layer, quant);
	for (CALL_DATA i = 0; i < MAX_FC; i += stage_pipeline){
		#pragma HLS UNROLL

		//Info and Tile
		generate_info_tile(ext_tile, ext_info,
						   tile_index_0, info_0,
						   i, type_layer);
		//Info and Tile
		generate_info_tile(ext_tile, ext_info,
						   tile_index_1, info_1,
						   i+1, type_layer);
		//PEs
		PEs(ext_in_data, ext_out_data,
			ext_residual_map_read, ext_residual_map_write,
			ext_w_conv, ext_b_conv,
			ext_w_fc, ext_b_fc,
			tile_index_0, quant, info_0, i, type_layer);
		//PEs
		PEs(ext_in_data, ext_out_data,
			ext_residual_map_read, ext_residual_map_write,
			ext_w_conv, ext_b_conv,
			ext_w_fc, ext_b_fc,
			tile_index_1, quant, info_1, i+1, type_layer);
	}
}






