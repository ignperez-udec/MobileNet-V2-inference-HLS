#include "memory_access.h"
#include "type_data.h"
#include "parameters.h"
#include "tile.h"
#include "info.h"
#include <stdio.h>

void order2CPU_map(DATA *in_mem, DATA *out_mem, DATA depth, DATA length, DATA stride);
DATA DRAM_2_STREAM_3x3(DATA *in_mem, DATA *out_mem,
		DATA depth_out, DATA depth_in, DATA length, DATA type_layer);
DATA DRAM_2_STREAM_1x1(DATA *in_mem, DATA *out_mem,
		DATA depth_out, DATA depth_in, DATA length, DATA type_layer, DATA expansion);
DATA DRAM_2_STREAM_res(DATA depth, DATA length, DATA residual, DATA store_map);
DATA DRAM_2_STREAM_array(DATA *in_mem, DATA *out_mem,
		DATA length_out, DATA length_in);
void STREAM_2_DRAM_3x3(DATA *in_mem, DATA *out_mem, DATA depth, DATA length, DATA stride);
void STREAM_2_DRAM_1x1(DATA *in_mem, DATA *out_mem, DATA depth, DATA length);
void set_tile_info(DATA *tile_3x3, DATA *tile_convs, DATA *tile_avg, DATA *tile_fc,
				   DATA *info_3x3, DATA *info_convs, DATA *info_avg, DATA *info_fc);
int min(int x, int y);

void order2CPU_map(DATA *in_mem, DATA *out_mem, DATA depth, DATA length, DATA stride){
	DATA length_out = length/stride;
	DATA multi_max_in = length_out*length_out;
	DATA min_i, min_j, min_k;
	DATA limit_i, limit_j, limit_k;
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

DATA DRAM_2_STREAM_3x3(DATA *in_mem, DATA *out_mem,
		DATA depth_out, DATA depth_in, DATA length, DATA type_layer){
	DATA multi_max_in = length*length;
	DATA limit_map = length - 1;
	DATA limit_j, limit_k, limit_l;
	DATA min_k, min_l, size_x;
	DATA depth_in_PE = depth_in/number_PE;
    int pos_x, pos_y;
    DATA counter = 0, len_out = 0;

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
    				for (int i = 0; i < min(depth_out, tile_conv_out)/number_PE; i++){
						//Transfer
						for (int m = j; m < limit_j; m++){
							pos_x = m*multi_max_in;
							for (int n = min_k; n < limit_k; n++){
								if (n >= 0 && n <= limit_map){
									pos_y = n*length + pos_x;
									for (int o = min_l; o < limit_l; o++){
										if (o >= 0 && o <= limit_map){
											out_mem[len_out] = in_mem[o + pos_y];
											len_out++;
										}
										else{
											out_mem[len_out] = 0;
											len_out++;
										}
									}
								}
								else{
									for (int n = 0; n < size_x; n++){
										out_mem[len_out] = 0;
										len_out++;
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
    	for (int PE = 0; PE < number_PE; PE++){
        	for (int j = PE*depth_in_PE; j < (PE + 1)*depth_in_PE; j += tile_conv_out){   //in_map_depth
        		limit_j = min((PE + 1)*depth_in_PE, j + tile_conv_out);
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
        									out_mem[counter] = in_mem[o + pos_y];
        									counter++;
        								}
        								else{
        									out_mem[counter] = 0;
        									counter++;
        								}
        							}
        						}
        						else{
        							for (int n = 0; n < size_x; n++){
    									out_mem[counter] = 0;
        								counter++;
        							}
        						}
        					}
        				}
        			}
        		}
        	}
        	if (PE == 0){
        		len_out = counter;
        	}
    	}
    }

	return len_out;
}

DATA DRAM_2_STREAM_1x1(DATA *in_mem, DATA *out_mem,
		DATA depth_out, DATA depth_in, DATA length, DATA type_layer, DATA stride){
	DATA multi_max_in = length*length;
	DATA data_for_PE = length*length*depth_in/number_PE;
	DATA add_lk, mul_lk;
	DATA min_j, min_k, min_l;
	DATA limit_j, limit_k, limit_l;
	DATA depth_out_PE = depth_out/number_PE;
	DATA depth_in_PE = depth_in/number_PE;
    int pos_x, pos_y;
    DATA counter = 0, counter_in = 0, aux;

    if (type_layer == 1){
    	if (stride == 1){
        	for (int k = 0; k < length; k += tile_map){   //length_in_map
        		min_k = min(length, k + tile_map);
        		for (int l = 0; l < length; l += tile_map){   //length_in_map
        			min_l = min(length, l + tile_map);
        			for (int i = 0; i < depth_out_PE; i += tile_conv_out){   //out_map_depth (repetition stream)
        				for (int j = 0; j < depth_in; j += tile_conv_in){   //in_map_depth
        					min_j = min(depth_in, j + tile_conv_in);
        					//Transfer
        					for (int m = j; m < min_j; m++){   //depth
        						pos_x = m*multi_max_in;
        						for (int n = k; n < min_k; n++){   //length
        							pos_y = n*length + pos_x;
        							for (int o = l; o < min_l; o++){   //length
        								out_mem[counter] = in_mem[o + pos_y];
        								counter++;
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
        			add_lk = l_pos +  k_pos;
        			mul_lk = limit_l*limit_k;
        			for (int i = 0; i < depth_out_PE; i += tile_conv_out){   //out_map_depth (repetition stream)
        			    for (int PE = 0; PE < number_PE; PE++){
        			    	aux = PE*data_for_PE;
        			    	for (int j = PE*depth_in_PE, j_pos = 0; j < (PE + 1)*depth_in_PE; j += tile_conv_in){   //in_map_depth
								min_j = min((PE + 1)*depth_in_PE, j + tile_conv_in);
								limit_j = min_j - j;
								counter_in = add_lk*limit_j + j_pos + aux;
								//Transfer
								memcpy(out_mem + counter, in_mem + counter_in, mul_lk*limit_j*sizeof(DATA));
								counter += mul_lk*limit_j;
								j_pos += multi_max_in*limit_j;
        			    	}
        			    }
        			}
        			l_pos += mul_lk;
        		}
        		k_pos += length*limit_k;
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
        			for (int j = PE*depth_in_PE; j < (PE + 1)*depth_in_PE; j += tile_conv_in){   //in_map_depth
        				min_j = min((PE + 1)*depth_in_PE, j + tile_conv_in);
						limit_j = min_j - j;
						//Transfer
						for (int m = j; m < min_j; m++){   //depth
							pos_x = m*multi_max_in;
							for (int n = k; n < min_k; n++){   //length
								pos_y = n*length + pos_x;
								for (int o = l; o < min_l; o++){   //length
									out_mem[counter_in] = in_mem[o + pos_y];
									counter_in++;
								}
							}
						}
					}
        		}
        	}
        	if (PE == 0){
        		counter = counter_in;
        	}
    	}
    }

    return counter;
}

DATA DRAM_2_STREAM_res(DATA depth, DATA length, DATA residual, DATA store_map){
	DATA counter_out = 0;

	if (residual){
		counter_out = depth/number_PE*length*length;
	}

	if (store_map && residual == 0){
		counter_out = ((depth/number_PE)/tile_conv_out + (((depth/number_PE)%tile_conv_out)&&1))*(length/tile_map + ((length%tile_map)&&1))*(length/tile_map + ((length%tile_map)&&1));
	}

	return counter_out;
}

DATA DRAM_2_STREAM_array(DATA *in_mem, DATA *out_mem, DATA length_out, DATA length_in){
	DATA counter = 0;

	for (int i = 0; i < length_out/number_PE; i += tile_fc_out){
		memcpy(out_mem + counter, in_mem, length_in*sizeof(DATA));
		counter += length_in;
	}

	return counter;
}

void STREAM_2_DRAM_3x3(DATA *in_mem, DATA *out_mem, DATA depth, DATA length, DATA stride){
	DATA length_out = length/stride;
	DATA multi_max_in = length_out*length_out;
	DATA min_i, min_j, min_k, j_stride, k_stride;
	DATA depth_PE = depth/number_PE;
    int counter = 0;
    int pos_x, pos_y;

    for (int PE = 0; PE < number_PE; PE++){
    	for (int i = PE*depth_PE; i < (PE + 1)*depth_PE; i += tile_conv_out){   //in_map_depth
    		min_i = min((PE + 1)*depth_PE, i + tile_conv_out);
			for (int j = 0; j < length; j += tile_map){   //length_in_map
				min_j = min(length, j + tile_map)/stride;
				j_stride = j/stride;
				for (int k = 0; k < length; k += tile_map){   //length_in_map
					min_k = min(length, k + tile_map)/stride;
					k_stride = k/stride;
					//Transfer
					for (int l = i; l < min_i; l++){
						pos_x = l*multi_max_in;
						for (int m = j_stride; m < min_j; m++){
							pos_y = m*length_out + pos_x;
							for (int o = k_stride; o < min_k; o++){
								out_mem[o + pos_y] = in_mem[counter];
								counter++;
							}
						}
					}
				}
			}
		}
    }
}

void STREAM_2_DRAM_1x1(DATA *in_mem, DATA *out_mem, DATA depth, DATA length){
	DATA multi_max_in = length*length;
	DATA min_i, min_j, min_k;
	DATA limit_i, limit_j, limit_k;
	DATA depth_PE = depth/number_PE;
    int counter = 0;
    int pos_x, pos_y;

    for (int PE = 0; PE < number_PE; PE++){
		for (int j = 0; j < length; j += tile_map){   //length_in_map
			min_j = min(length, j + tile_map);
			limit_j = min_j - j;
			for (int k = 0; k < length; k += tile_map){   //length_in_map
				min_k = min(length, k + tile_map);
    			for (int i = PE*depth_PE; i < (PE + 1)*depth_PE; i += tile_conv_out){   //out_map_depth
    				min_i = min((PE + 1)*depth_PE, i + tile_conv_out);
					limit_i = min_i - i;
					//Transfer
					for (int l = i; l < min_i; l++){   //depth
						pos_x = l*multi_max_in;
						for (int m = j; m < min_j; m++){   //length
							pos_y = m*length + pos_x;
							for (int o = k; o < min_k; o++){   //length
								out_mem[o + pos_y] = in_mem[counter];
								counter++;
							}
						}
					}
				}
			}
		}
    }
}

void set_tile_info(DATA *tile_3x3, DATA *tile_convs, DATA *tile_avg, DATA *tile_fc,
				   DATA *info_3x3, DATA *info_convs, DATA *info_avg, DATA *info_fc){

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
