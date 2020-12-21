#include "memory_access.h"
#include "../../src_hw/parameters.h"
#include "tile.h"
#include "info.h"
#include <stdio.h>
#include <hls_stream.h>

void order2FPGA_3x3(DATA_SW *in_mem, DATA_SW *out_mem, DATA_SW depth, DATA_SW length);
void order2FPGA_1x1(DATA_SW *in_mem, DATA_SW *out_mem, DATA_SW depth, DATA_SW length);
void order2CPU_map(DATA_SW *in_mem, DATA_SW *out_mem, DATA_SW depth, DATA_SW length, DATA_SW stride);
void DRAM_2_STREAM_3x3(DATA_SW *in_mem, hls::stream<DATA_STREAM> &out_mem_A, hls::stream<DATA_STREAM> &out_mem_B,
		hls::stream<DATA_STREAM> &out_mem_C, hls::stream<DATA_STREAM> &out_mem_D,
		DATA_SW depth_out, DATA_SW depth_in, DATA_SW length, DATA_SW type_layer);
void DRAM_2_STREAM_1x1(DATA_SW *in_mem, hls::stream<DATA_STREAM> &out_mem_A, hls::stream<DATA_STREAM> &out_mem_B,
		hls::stream<DATA_STREAM> &out_mem_C, hls::stream<DATA_STREAM> &out_mem_D,
		DATA_SW depth_out, DATA_SW depth_in, DATA_SW length, DATA_SW stride, DATA_SW type_layer, DATA_SW expansion);
void DRAM_2_STREAM_res(DATA_SW *in_mem, hls::stream<DATA_STREAM> &out_mem_A, hls::stream<DATA_STREAM> &out_mem_B,
		hls::stream<DATA_STREAM> &out_mem_C, hls::stream<DATA_STREAM> &out_mem_D,
		DATA_SW depth, DATA_SW length, DATA_SW residual, DATA_SW store_map);
void DRAM_2_STREAM_array(DATA_SW *in_mem, hls::stream<DATA_STREAM> &out_mem_A, hls::stream<DATA_STREAM> &out_mem_B,
		hls::stream<DATA_STREAM> &out_mem_C, hls::stream<DATA_STREAM> &out_mem_D,
		DATA_SW length_out, DATA_SW length_in);
void STREAM_2_DRAM_3x3(hls::stream<DATA_STREAM> &in_mem_A, hls::stream<DATA_STREAM> &in_mem_B,
		hls::stream<DATA_STREAM> &in_mem_C, hls::stream<DATA_STREAM> &in_mem_D,
		DATA_SW *out_mem, DATA_SW depth, DATA_SW length, DATA_SW stride, DATA_SW expansion);
void STREAM_2_DRAM_1x1(hls::stream<DATA_STREAM> &in_mem_A, hls::stream<DATA_STREAM> &in_mem_B,
		hls::stream<DATA_STREAM> &in_mem_C, hls::stream<DATA_STREAM> &in_mem_D,
		DATA_SW *out_mem, DATA_SW depth, DATA_SW length, DATA_SW expansion);
void STREAM_2_DRAM_res(hls::stream<DATA_STREAM> &in_mem_A, hls::stream<DATA_STREAM> &in_mem_B,
		hls::stream<DATA_STREAM> &in_mem_C, hls::stream<DATA_STREAM> &in_mem_D,
		DATA_SW *out_mem, DATA_SW depth, DATA_SW length);
void STREAM_2_DRAM_array(hls::stream<DATA_STREAM> &in_mem_A, hls::stream<DATA_STREAM> &in_mem_B,
		hls::stream<DATA_STREAM> &in_mem_C, hls::stream<DATA_STREAM> &in_mem_D,H,
		DATA_SW *out_mem, DATA_SW depth);
void set_tile_info(DATA_SW *tile_3x3, DATA_SW *tile_convs, DATA_SW *tile_avg, DATA_SW *tile_fc,
				   DATA_SW *info_3x3, DATA_SW *info_convs, DATA_SW *info_avg, DATA_SW *info_fc);
int min(int x, int y);

void order2FPGA_3x3(DATA_SW *in_mem, DATA_SW *out_mem, DATA_SW depth, DATA_SW length){
	DATA_SW multi_max_in = length*length;
	DATA_SW limit_map = length - 1;
	DATA_SW limit_i, limit_j, limit_k;
	DATA_SW min_j, min_k, size_x;
    int pos_x, pos_y;
    int counter = 0;

	for (int i = 0; i < depth; i += tile_conv_out){   //out_map_depth
		limit_i = min(depth, i + tile_conv_out);
		for (int j = 0; j < length; j += tile_map){   //length_in_map
			limit_j = min(length, j + tile_map) + 1;
			min_j = j - 1;
			size_x = limit_j - j + 1;
			for (int k = 0; k < length; k += tile_map){   //length_in_map
				limit_k = min(length, k + tile_map) + 1;
				min_k = k - 1;
				//Transfer
			    for (int l = i; l < limit_i; l++){
			        pos_x = l*multi_max_in;
			        for (int m = min_j; m < limit_j; m++){
			            if (m >= 0 && m <= limit_map){
			                pos_y = m*length + pos_x;
			                for (int o = min_k; o < limit_k; o++){
			                    if (o >= 0 && o <= limit_map){
			                    	out_mem[counter] = in_mem[o + pos_y];
			                    }
			                    counter++;
			                }
			            }
			            else{
			            	counter += size_x;
			            }
			        }
			    }
			}
		}
	}
}

void order2FPGA_1x1(DATA_SW *in_mem, DATA_SW *out_mem, DATA_SW depth, DATA_SW length){
	DATA_SW multi_max_in = length*length;
	DATA_SW min_i, min_j, min_k;
	DATA_SW limit_i, limit_j, limit_k;
    int counter = 0;
    int pos_x, pos_y;

	for (int i = 0; i < depth; i += tile_conv_in){   //in_map_depth
		min_i = min(depth, i + tile_conv_in);
		limit_i = min_i - i;
		for (int j = 0; j < length; j += tile_map){   //length_in_map
			min_j = min(length, j + tile_map);
			limit_j = min_j - j;
			for (int k = 0; k < length; k += tile_map){   //length_in_map
				min_k = min(length, k + tile_map);
				//Transfer
				for (int l = i; l < min_i; l++){
					pos_x = l*multi_max_in;
					for (int m = j; m < min_j; m++){
						pos_y = m*length + pos_x;
						for (int o = k; o < min_k; o++){
							out_mem[counter] = in_mem[o + pos_y];
							counter++;
						}
					}
				}
			}
		}
	}
}

void order2CPU_map(DATA_SW *in_mem, DATA_SW *out_mem, DATA_SW depth, DATA_SW length, DATA_SW stride){
	DATA_SW length_out = length/stride;
	DATA_SW multi_max_in = length_out*length_out;
	DATA_SW min_i, min_j, min_k;
	DATA_SW limit_i, limit_j, limit_k;
    int counter = 0;
    int pos_x, pos_y;

	for (int i = 0; i < depth; i += tile_conv_out){   //in_map_depth
		min_i = min(depth, i + tile_conv_out);
		limit_i = min_i - i;
		for (int j = 0; j < length; j += tile_map){   //length_in_map
			min_j = min(length, j + tile_map);
			limit_j = min_j - j;
			for (int k = 0; k < length; k += tile_map){   //length_in_map
				min_k = min(length, k + tile_map);
				//Transfer
			    for (int l = i; l < min_i; l++){
			        pos_x = l*multi_max_in;
			        for (int m = j/stride; m < min_j/stride; m++){
			            pos_y = m*length_out + pos_x;
			            for (int o = k/stride; o < min_k/stride; o++){
			                out_mem[o + pos_y] = in_mem[counter];
			                counter++;
			            }
			        }
			    }
			}
		}
	}
}

void DRAM_2_STREAM_3x3(DATA_SW *in_mem, hls::stream<DATA_STREAM> &out_mem_A, hls::stream<DATA_STREAM> &out_mem_B,
		hls::stream<DATA_STREAM> &out_mem_C, hls::stream<DATA_STREAM> &out_mem_D,
		DATA_SW depth_out, DATA_SW depth_in, DATA_SW length, DATA_SW type_layer){

	DATA_SW multi_max_in = length*length;
	DATA_SW limit_map = length - 1;
	DATA_SW limit_j, limit_k, limit_l;
	DATA_SW min_k, min_l, size_x;
    int pos_x, pos_y;
    DATA_STREAM valIn;

    if (type_layer == 0){
		//WARNING: in depthwise dont need repetition, but is tile_conv_out is less than 32, error in CONV_3x3
		for (int j = 0; j < depth_in; j += tile_conv_out){   //in_map_depth
			limit_j = min(depth_in, j + tile_conv_out);
			for (int k = 0; k < length; k += tile_map){   //length_in_map
				limit_k = min(length, k + tile_map) + 1;
				min_k = k - 1;
				size_x = limit_k - k + 1;
				for (int l = 0; l < length; l += tile_map){   //length_in_map
					limit_l = min(length, l + tile_map) + 1;
					min_l = l - 1;
					for (int i = 0; i < min(depth_out, tile_conv_out)/number_PE; i++){   //repeat for each out channel
						//Transfer
						for (int m = j; m < limit_j; m++){
							pos_x = m*multi_max_in;
							for (int n = min_k; n < limit_k; n++){
								if (n >= 0 && n <= limit_map){
									pos_y = n*length + pos_x;
									for (int o = min_l; o < limit_l; o++){
										if (o >= 0 && o <= limit_map){
											valIn.data = in_mem[o + pos_y];
											valIn.keep = 1; valIn.strb = 1; valIn.user = 1; valIn.id = 0; valIn.dest = 0;
											out_mem_A << valIn;
											out_mem_B << valIn;
											out_mem_C << valIn;
											out_mem_D << valIn;
										}
										else{
											valIn.data = 0;
											valIn.keep = 1; valIn.strb = 1; valIn.user = 1; valIn.id = 0; valIn.dest = 0;
											out_mem_A << valIn;
											out_mem_B << valIn;
											out_mem_C << valIn;
											out_mem_D << valIn;
										}
									}
								}
								else{
									for (int n = 0; n < size_x; n++){
										valIn.data = 0;
										valIn.keep = 1; valIn.strb = 1; valIn.user = 1; valIn.id = 0; valIn.dest = 0;
										out_mem_A << valIn;
										out_mem_B << valIn;
										out_mem_C << valIn;
										out_mem_D << valIn;
									}
								}
							}
						}
					}
				}
			}
		}
    }
    else if (type_layer == 2){
		//WARNING: in depthwise dont need repetition, but is tile_conv_out is less than 32, error in CONV_3x3
		for (int PE = 0; PE < number_PE; PE++){
	    	for (int j = PE*depth_in/number_PE; j < (PE + 1)*depth_in/number_PE; j += tile_conv_out){   //in_map_depth
				limit_j = min((PE + 1)*depth_in/number_PE, j + tile_conv_out);
				for (int k = 0; k < length; k += tile_map){   //length_in_map
					limit_k = min(length, k + tile_map) + 1;
					min_k = k - 1;
					size_x = limit_k - k + 1;
					for (int l = 0; l < length; l += tile_map){   //length_in_map
						limit_l = min(length, l + tile_map) + 1;
						min_l = l - 1;
						//Transfer
						for (int m = j; m < limit_j; m++){
							pos_x = m*multi_max_in;
							for (int n = min_k; n < limit_k; n++){
								if (n >= 0 && n <= limit_map){
									pos_y = n*length + pos_x;
									for (int o = min_l; o < limit_l; o++){
										if (o >= 0 && o <= limit_map){
											valIn.data = in_mem[o + pos_y];
											valIn.keep = 1; valIn.strb = 1; valIn.user = 1; valIn.id = 0; valIn.dest = 0;
											if (PE == 0){
												out_mem_A << valIn;
											}
											else if (PE == 1){
												out_mem_B << valIn;
											}
											else if (PE == 2){
												out_mem_C << valIn;
											}
											else if (PE == 3){
												out_mem_D << valIn;
											}
										}
										else{
											valIn.data = 0;
											valIn.keep = 1; valIn.strb = 1; valIn.user = 1; valIn.id = 0; valIn.dest = 0;
											if (PE == 0){
												out_mem_A << valIn;
											}
											else if (PE == 1){
												out_mem_B << valIn;
											}
											else if (PE == 2){
												out_mem_C << valIn;
											}
											else if (PE == 3){
												out_mem_D << valIn;
											}
										}
									}
								}
								else{
									for (int n = 0; n < size_x; n++){
										valIn.data = 0;
										valIn.keep = 1; valIn.strb = 1; valIn.user = 1; valIn.id = 0; valIn.dest = 0;
										if (PE == 0){
											out_mem_A << valIn;
										}
										else if (PE == 1){
											out_mem_B << valIn;
										}
										else if (PE == 2){
											out_mem_C << valIn;
										}
										else if (PE == 3){
											out_mem_D << valIn;
										}
									}
								}
							}
						}
					}
				}
			}
		}
    }

}

void DRAM_2_STREAM_1x1(DATA_SW *in_mem, hls::stream<DATA_STREAM> &out_mem_A, hls::stream<DATA_STREAM> &out_mem_B,
		hls::stream<DATA_STREAM> &out_mem_C, hls::stream<DATA_STREAM> &out_mem_D,
		DATA_SW depth_out, DATA_SW depth_in, DATA_SW length, DATA_SW stride, DATA_SW type_layer, DATA_SW expansion){

	DATA_SW multi_max_in = length*length;
	DATA_SW min_j, min_k, min_l;
	DATA_SW limit_j, limit_k, limit_l;
    int pos_x, pos_y, reg = 0;
    DATA_STREAM valIn;

    if (type_layer == 1){
    	if (expansion){
    		for (int k = 0; k < length; k += tile_map){   //length_in_map
    			min_k = min(length, k + tile_map);
    			limit_k = min_k - k;
    			for (int l = 0; l < length; l += tile_map){   //length_in_map
    				min_l = min(length, l + tile_map);
    				for (int i = 0; i < depth_out/number_PE; i += tile_conv_out){   //out_map_depth (repetition stream)
    					for (int j = 0; j < depth_in; j += tile_conv_in){   //in_map_depth
    						min_j = min(depth_in, j + tile_conv_in);
    						limit_j = min_j - j;
    						//Transfer
    						for (int m = j; m < min_j; m++){   //depth
    							pos_x = m*multi_max_in;
    							for (int n = k; n < min_k; n++){   //length
    								pos_y = n*length + pos_x;
    								for (int o = l; o < min_l; o++){   //length
    									valIn.data = in_mem[o + pos_y];
    									valIn.keep = 1; valIn.strb = 1; valIn.user = 1; valIn.id = 0; valIn.dest = 0;
    									out_mem_A << valIn;
    									out_mem_B << valIn;
    									out_mem_C << valIn;
    									out_mem_D << valIn;
    								}
    							}
    						}
    					}
    				}
    			}
    		}
    	}
    	else{
        	for (int k = 0, k_pos = 0; k < length; k += tile_map){   //length_in_map
        		min_k = min(length, k + tile_map);
        		limit_k = min_k - k;
        		for (int l = 0, l_pos = 0; l < length; l += tile_map){   //length_in_map
        			min_l = min(length, l + tile_map);
        			limit_l = min_l - l;
        			for (int i = 0; i < depth_out/number_PE; i += tile_conv_out){   //out_map_depth (repetition stream)
        			    for (int PE = 0; PE < number_PE; PE++){
        			    	for (int j = PE*depth_in/number_PE, j_pos = 0; j < (PE + 1)*depth_in/number_PE; j += tile_conv_in){   //in_map_depth
								min_j = min((PE + 1)*depth_in/number_PE, j + tile_conv_in);
								limit_j = min_j - j;
								reg = l_pos*limit_j + k_pos*limit_j + j_pos + PE*(length/stride)*(length/stride)*depth_in/number_PE;
								//Transfer
								for (int x = reg; x < reg + limit_l*limit_k*limit_j/(stride*stride); x++){
									valIn.data = in_mem[x];
									valIn.keep = 1; valIn.strb = 1; valIn.user = 1; valIn.id = 0; valIn.dest = 0;
									out_mem_A << valIn;
									out_mem_B << valIn;
									out_mem_C << valIn;
									out_mem_D << valIn;
								}
								j_pos += (length/stride)*(length/stride)*limit_j;
        			    	}
        			    }
        			}
        			l_pos += limit_l*limit_k/(stride*stride);
        		}
        		k_pos += (length/stride)*limit_k/stride;
        	}
    	}
    }
    else if (type_layer == 3){
    	for (int PE = 0; PE < number_PE; PE++){
    		for (int k = 0; k < length; k += tile_map){   //length_in_map
    			min_k = min(length, k + tile_map);
    			limit_k = min_k - k;
    			for (int l = 0; l < length; l += tile_map){   //length_in_map
    				min_l = min(length, l + tile_map);
					for (int j = PE*depth_in/number_PE; j < (PE + 1)*depth_in/number_PE; j += tile_conv_in){   //in_map_depth
						min_j = min((PE + 1)*depth_in/number_PE, j + tile_conv_in);
						limit_j = min_j - j;
						//Transfer
						for (int m = j; m < min_j; m++){   //depth
							pos_x = m*multi_max_in;
							for (int n = k; n < min_k; n++){   //length
								pos_y = n*length + pos_x;
								for (int o = l; o < min_l; o++){   //length
									valIn.data = in_mem[o + pos_y];
									valIn.keep = 1; valIn.strb = 1; valIn.user = 1; valIn.id = 0; valIn.dest = 0;
									if (PE == 0){
										out_mem_A << valIn;
									}
									else if (PE == 1){
										out_mem_B << valIn;
									}
									else if (PE == 2){
										out_mem_C << valIn;
									}
									else if (PE == 3){
										out_mem_D << valIn;
									}
								}
							}
						}
					}
    			}
    		}
    	}
    }
}

void DRAM_2_STREAM_res(DATA_SW *in_mem, hls::stream<DATA_STREAM> &out_mem_A, hls::stream<DATA_STREAM> &out_mem_B,
		hls::stream<DATA_STREAM> &out_mem_C, hls::stream<DATA_STREAM> &out_mem_D,
		DATA_SW depth, DATA_SW length, DATA_SW residual, DATA_SW store_map){

	DATA_STREAM valIn;
	DATA_SW min_i, min_j, min_k;
	int counter = 0;

	if (residual){
		for (int PE = 0; PE < number_PE; PE++){
			for (int i = PE*depth/number_PE*length*length; i < ((PE + 1)*depth/number_PE)*length*length; i++){
				valIn.data = in_mem[counter];
				valIn.keep = 1; valIn.strb = 1; valIn.user = 1; valIn.id = 0; valIn.dest = 0;
				if (PE == 0){
					out_mem_A << valIn;
				}
				else if (PE == 1){
					out_mem_B << valIn;
				}
				else if (PE == 2){
					out_mem_C << valIn;
				}
				else if (PE == 3){
					out_mem_D << valIn;
				}
				counter++;
			}
		}
	}

	if (store_map && residual == 0){
		for (int PE = 0; PE < number_PE; PE++){
		    for (int i = PE*depth/number_PE; i < (PE + 1)*depth/number_PE; i += tile_conv_out){   //depth
				for (int j = 0; j < length; j += tile_map){   //length
					for (int k = 0; k < length; k += tile_map){   //length
						valIn.data = 0;
						valIn.keep = 1; valIn.strb = 1; valIn.user = 1; valIn.id = 0; valIn.dest = 0;
						if (PE == 0){
							out_mem_A << valIn;
						}
						else if (PE == 1){
							out_mem_B << valIn;
						}
						else if (PE == 2){
							out_mem_C << valIn;
						}
						else if (PE == 3){
							out_mem_D << valIn;
						}
					}
				}
		    }
		}
	}
}

void DRAM_2_STREAM_array(DATA_SW *in_mem, hls::stream<DATA_STREAM> &out_mem_A, hls::stream<DATA_STREAM> &out_mem_B,
		hls::stream<DATA_STREAM> &out_mem_C, hls::stream<DATA_STREAM> &out_mem_D,
		DATA_SW length_out, DATA_SW length_in){

	DATA_STREAM valIn;

	for (int i = 0; i < length_out/number_PE; i += tile_fc_out){   //out length
		for (int j = 0; j < length_in; j += tile_fc_in){   //in length
			//Transfer
			for (int k = j; k < j + tile_fc_in; k++){
				valIn.data = in_mem[k];
				valIn.keep = 1; valIn.strb = 1; valIn.user = 1; valIn.id = 0; valIn.dest = 0;
				out_mem_A << valIn;
				out_mem_B << valIn;
				out_mem_C << valIn;
				out_mem_D << valIn;
			}
		}
	}
}

void STREAM_2_DRAM_3x3(hls::stream<DATA_STREAM> &in_mem_A, hls::stream<DATA_STREAM> &in_mem_B,
		hls::stream<DATA_STREAM> &in_mem_C, hls::stream<DATA_STREAM> &in_mem_D,
		DATA_SW *out_mem, DATA_SW depth, DATA_SW length, DATA_SW stride, DATA_SW expansion){

	DATA_SW length_out = length/stride;
	DATA_SW multi_max_in = length_out*length_out;
	DATA_SW min_i, min_j, min_k;
	DATA_SW limit_i, limit_j, limit_k;
	DATA_STREAM valOut;
    int counter = 0;
    int pos_x, pos_y;

    if (expansion){
        for (int PE = 0; PE < number_PE; PE++){
        	for (int i = PE*depth/number_PE; i < (PE + 1)*depth/number_PE; i += tile_conv_out){   //in_map_depth
        		min_i = min((PE + 1)*depth/number_PE, i + tile_conv_out);
        		limit_i = min_i - i;
        		for (int j = 0; j < length; j += tile_map){   //length_in_map
        			min_j = min(length, j + tile_map);
        			limit_j = min_j - j;
        			for (int k = 0; k < length; k += tile_map){   //length_in_map
        				min_k = min(length, k + tile_map);
        				//Transfer
        			    for (int l = i; l < min_i; l++){
        			        pos_x = l*multi_max_in;
        			        for (int m = j/stride; m < min_j/stride; m++){
        			            pos_y = m*length_out + pos_x;
        			            for (int o = k/stride; o < min_k/stride; o++){
        			            	if (PE == 0){
            			        		in_mem_A.read(valOut);
            			        		out_mem[o + pos_y] = valOut.data;
        			            	}
        			            	else if (PE == 1){
            			        		in_mem_B.read(valOut);
            			        		out_mem[o + pos_y] = valOut.data;
        			            	}
        			            	else if (PE == 2){
            			        		in_mem_C.read(valOut);
            			        		out_mem[o + pos_y] = valOut.data;
        			            	}
        			            	else if (PE == 3){
            			        		in_mem_D.read(valOut);
            			        		out_mem[o + pos_y] = valOut.data;
        			            	}
        			        		counter++;
        			        		if (counter == ((PE + 1)*depth/number_PE)*length_out*length_out){   //Last data
        			        			if (valOut.last == 0 && valOut.keep == 1 && valOut.strb == 1 && valOut.user == 1 && valOut.id == 0 && valOut.dest == 0){
        			        				printf("ERROR");
        			        			}
        			        		}
        			        		else{
        			        			if (valOut.last == 1 && valOut.keep == 1 && valOut.strb == 1 && valOut.user == 1 && valOut.id == 0 && valOut.dest == 0){
        			        				printf("ERROR");
        			        			}
        			        		}
        			            }
        			        }
        			    }
        			}
        		}
        	}
        }
    }
    else{
    	for (DATA_SW i = 0; i < depth*length*length/(stride*stride); i++){
    		in_mem_A.read(valOut);
    		out_mem[i] = valOut.data;
    	}
    }
}

void STREAM_2_DRAM_1x1(hls::stream<DATA_STREAM> &in_mem_A, hls::stream<DATA_STREAM> &in_mem_B,
		hls::stream<DATA_STREAM> &in_mem_C, hls::stream<DATA_STREAM> &in_mem_D,
		DATA_SW *out_mem, DATA_SW depth, DATA_SW length, DATA_SW expansion){

	DATA_SW multi_max_in = length*length;
	DATA_SW min_i, min_j, min_k;
	DATA_SW limit_i, limit_j, limit_k;
	DATA_STREAM valOut;
    int counter = 0;
    int pos_x, pos_y;

    if (expansion){
        for (int PE = 0; PE < number_PE; PE++){
        	for (int j = 0; j < length; j += tile_map){   //length_in_map
        		min_j = min(length, j + tile_map);
        		limit_j = min_j - j;
        		for (int k = 0; k < length; k += tile_map){   //length_in_map
        			min_k = min(length, k + tile_map);
        			for (int i = PE*depth/number_PE; i < (PE + 1)*depth/number_PE; i += tile_conv_out){   //out_map_depth
        				min_i = min((PE + 1)*depth/number_PE, i + tile_conv_out);
        				limit_i = min_i - i;
        				//Transfer
        			    for (int l = i; l < min_i; l++){   //depth
        			        pos_x = l*multi_max_in;
        			        for (int m = j; m < min_j; m++){   //length
        			            pos_y = m*length + pos_x;
        			            for (int o = k; o < min_k; o++){   //length
        			            	if (PE == 0){
            			        		in_mem_A.read(valOut);
            			        		out_mem[o + pos_y] = valOut.data;
        			            	}
        			            	else if (PE == 1){
            			        		in_mem_B.read(valOut);
            			        		out_mem[o + pos_y] = valOut.data;
        			            	}
        			            	else if (PE == 2){
            			        		in_mem_C.read(valOut);
            			        		out_mem[o + pos_y] = valOut.data;
        			            	}
        			            	else if (PE == 3){
            			        		in_mem_D.read(valOut);
            			        		out_mem[o + pos_y] = valOut.data;
        			            	}
        			        		counter++;
        			        		if (counter == ((PE + 1)*depth/number_PE)*length*length){   //Last data
        			        			if (valOut.last == 0 && valOut.keep == 1 && valOut.strb == 1 && valOut.user == 1 && valOut.id == 0 && valOut.dest == 0){
        			        				printf("ERROR");
        			        			}
        			        		}
        			        		else{
        			        			if (valOut.last == 1 && valOut.keep == 1 && valOut.strb == 1 && valOut.user == 1 && valOut.id == 0 && valOut.dest == 0){
        			        				printf("ERROR");
        			        			}
        			        		}
        			            }
        			        }
        			    }
        			}
        		}
        	}
        }
    }
    else{
    	for (DATA_SW i = 0; i < depth*length*length; i++){
    		in_mem_A.read(valOut);
    		out_mem[i] = valOut.data;
    	}
    }
}

void STREAM_2_DRAM_res(hls::stream<DATA_STREAM> &in_mem_A, hls::stream<DATA_STREAM> &in_mem_B,
		hls::stream<DATA_STREAM> &in_mem_C, hls::stream<DATA_STREAM> &in_mem_D,
		DATA_SW *out_mem, DATA_SW depth, DATA_SW length){

	DATA_STREAM valOut;

	for (int PE = 0; PE < number_PE; PE++){
		for (int i = PE*depth/number_PE*length*length; i < ((PE + 1)*depth/number_PE)*length*length; i++){
			if (PE == 0){
				in_mem_A.read(valOut);
				out_mem[i] = valOut.data;
			}
			else if (PE == 1){
				in_mem_A.read(valOut);
				out_mem[i] = valOut.data;
			}
			else if (PE == 2){
				in_mem_A.read(valOut);
				out_mem[i] = valOut.data;
			}
			else if (PE == 3){
				in_mem_A.read(valOut);
				out_mem[i] = valOut.data;
			}
			if (i + 1 == ((PE + 1)*depth/number_PE)*length*length){   //Last data
				if (valOut.last == 0 && valOut.keep == 1 && valOut.strb == 1 && valOut.user == 1 && valOut.id == 0 && valOut.dest == 0){
					printf("ERROR");
				}
			}
			else{
				if (valOut.last == 1 && valOut.keep == 1 && valOut.strb == 1 && valOut.user == 1 && valOut.id == 0 && valOut.dest == 0){
					printf("ERROR");
				}
			}
		}
	}
}

void STREAM_2_DRAM_array(hls::stream<DATA_STREAM> &in_mem_A, hls::stream<DATA_STREAM> &in_mem_B,
		hls::stream<DATA_STREAM> &in_mem_C, hls::stream<DATA_STREAM> &in_mem_D,
		DATA_SW *out_mem, DATA_SW depth){

	DATA_STREAM valOut;

	for (int PE = 0; PE < number_PE; PE++){
		for (int i = PE*depth/number_PE; i < (PE + 1)*depth/number_PE; i++){
			if (PE == 0){
				in_mem_A.read(valOut);
				out_mem[i] = valOut.data;
			}
			else if (PE == 1){
				in_mem_B.read(valOut);
				out_mem[i] = valOut.data;
			}
			else if (PE == 2){
				in_mem_C.read(valOut);
				out_mem[i] = valOut.data;
			}
			else if (PE == 3){
				in_mem_D.read(valOut);
				out_mem[i] = valOut.data;
			}
			if (i + 1 == (PE + 1)*depth/number_PE){   //Last data
				if (valOut.last == 0 && valOut.keep == 1 && valOut.strb == 1 && valOut.user == 1 && valOut.id == 0 && valOut.dest == 0){
					printf("ERROR");
				}
			}
			else{
				if (valOut.last == 1 && valOut.keep == 1 && valOut.strb == 1 && valOut.user == 1 && valOut.id == 0 && valOut.dest == 0){
					printf("ERROR");
				}
			}
		}
	}
}

void set_tile_info(DATA_SW *tile_3x3, DATA_SW *tile_convs, DATA_SW *tile_avg, DATA_SW *tile_fc,
				   DATA_SW *info_3x3, DATA_SW *info_convs, DATA_SW *info_avg, DATA_SW *info_fc){

	//PEs
	for (int PE = 0; PE < number_PE; PE++){
	    //TILE
	    //3x3
		for (int i = 0; i < MAX_CONV_3X3; i++){
			for (int j = 0; j < 3; j++){
				tile_3x3[j + i*3 + PE*3*MAX_CONV_3X3] = CONV_3X3_TILE[PE][i][j];
			}
		}
		//convs
		for (int i = 0; i < 18; i++){
			for (int j = 0; j < 3; j++){
				for (int k = 0; k < MAX_CONVS; k++){
					for (int l = 0; l < 3; l++){
						tile_convs[l + k*3 + j*3*MAX_CONVS + i*3*MAX_CONVS*3 + PE*3*MAX_CONVS*3*18] = CONVS_TILE[i][j][PE][k][l];
					}
				}
			}
		}
	    //avg
		for (int i = 0; i < MAX_AVG; i++){
			for (int j = 0; j < 3; j++){
				tile_avg[j + i*3 + PE*3*MAX_AVG] = AVG_TILE[PE][i][j];
			}
		}
	    //fc
		for (int i = 0; i < MAX_FC; i++){
			for (int j = 0; j < 3; j++){
				tile_fc[j + i*3 + PE*3*MAX_FC] = FC_TILE[PE][i][j];
			}
		}

	    //INFO
	    //3x3
		for (int i = 0; i < MAX_CONV_3X3; i++){
			for (int j = 0; j < size_info; j++){
				info_3x3[j + i*size_info + PE*size_info*MAX_CONV_3X3] = CONV_3X3_INFO[PE][i][j];;
			}
		}
		//convs
		for (int i = 0; i < 18; i++){
			for (int j = 0; j < 3; j++){
				for (int k = 0; k < MAX_CONVS; k++){
					for (int l = 0; l < size_info; l++){
						info_convs[l + k*size_info + j*size_info*MAX_CONVS + i*size_info*MAX_CONVS*3 + PE*size_info*MAX_CONVS*3*18] = CONVS_INFO[i][j][PE][k][l];
					}
				}
			}
		}
	    //avg
		for (int i = 0; i < MAX_AVG; i++){
			for (int j = 0; j < size_info; j++){
				info_avg[j + i*size_info + PE*size_info*MAX_AVG] = AVG_INFO[PE][i][j];
			}
		}
	    //fc
		for (int i = 0; i < MAX_FC; i++){
			for (int j = 0; j < size_info; j++){
				info_fc[j + i*size_info + PE*size_info*MAX_FC] = FC_INFO[PE][i][j];
			}
		}
	}
}


int min(int x, int y){
    if (x < y){
        return x;
    }
    else{
        return y;
    }
}
