#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <windows.h>
#include <hls_stream.h>
#include <time.h>
#include "layer.h"
#include "memory_access.h"
#include "../../src_hw/quant.h"
#include "../../src_hw/parameters.h"
#include "tile.h"
#include "info.h"
#include "quant.h"

//GENERAL PARAMETERS
#define IMAGE_SIZE 224
#define MAX_SIZE_KERNEL 3

//VARIABLES
DATA_SW weights_CONV[w_conv_layer];
DATA_SW bias_CONV[b_conv_layer];
DATA_SW weights_FC[fc_layer];
DATA_SW bias_FC[b_fc_layer];
char **class_ImageNet;
DATA_SW cpu_map[map_size];
DATA_SW cpu_array[1280];
DATA_SW res_map_A[save_input_LEN];
DATA_SW res_map_B[save_input_LEN];
//PE 0
hls::stream<DATA_STREAM> in_stream_0;
hls::stream<DATA_STREAM> out_stream_0;
hls::stream<DATA_STREAM> res_read_stream_0;
hls::stream<DATA_STREAM> res_write_stream_0;
//PE 1
hls::stream<DATA_STREAM> in_stream_1;
hls::stream<DATA_STREAM> out_stream_1;
hls::stream<DATA_STREAM> res_read_stream_1;
hls::stream<DATA_STREAM> res_write_stream_1;
//PE 2
hls::stream<DATA_STREAM> in_stream_2;
hls::stream<DATA_STREAM> out_stream_2;
hls::stream<DATA_STREAM> res_read_stream_2;
hls::stream<DATA_STREAM> res_write_stream_2;
//PE 3
hls::stream<DATA_STREAM> in_stream_3;
hls::stream<DATA_STREAM> out_stream_3;
hls::stream<DATA_STREAM> res_read_stream_3;
hls::stream<DATA_STREAM> res_write_stream_3;
//softmax
float array_softmax[1000];
//info and tile
DATA_SW tile_3x3[number_PE*3*(MAX_CONV_3X3)];
DATA_SW tile_convs[18*3*(number_PE*3*MAX_CONVS)];
DATA_SW tile_avg[number_PE*3*(MAX_AVG)];
DATA_SW tile_fc[number_PE*3*(MAX_FC)];
DATA_SW info_3x3[number_PE*size_info*(MAX_CONV_3X3)];
DATA_SW info_convs[18*3*(number_PE*size_info*MAX_CONVS)];
DATA_SW info_avg[number_PE*size_info*(MAX_AVG)];
DATA_SW info_fc[number_PE*size_info*(MAX_FC)];

// FUNCTIONS
void image_read(char *filename, DATA_SW *vector);
void weights_read(char *filename, DATA_SW *w_conv, DATA_SW *w_fc);
void bias_read(char *filename, DATA_SW *b_conv, DATA_SW *b_fc);
void class_read(char *filename, char **i_class);
void CONV_BATCH_RELU(hls::stream<DATA_STREAM> &in_map, hls::stream<DATA_STREAM> &out_map, hls::stream<DATA_STREAM> &res_map_write, hls::stream<DATA_STREAM> &res_map_read, DATA_SW *w_conv, DATA_SW *b_conv, int layer, int inter_layer, int type_layer, int PE);
void InvertedResidual(DATA_SW *res_map_write, DATA_SW *res_map_read, DATA_SW *w_conv, DATA_SW *b_conv, int layer, int length_in_map);
void AVG(hls::stream<DATA_STREAM> &in_map, hls::stream<DATA_STREAM> &out_array, hls::stream<DATA_STREAM> &res_map_write, hls::stream<DATA_STREAM> &res_map_read, int PE);
void Fully_Connected_layer(hls::stream<DATA_STREAM> &in_array, hls::stream<DATA_STREAM> &out_array, hls::stream<DATA_STREAM> &res_map_write, hls::stream<DATA_STREAM> &res_map_read, DATA_SW *w_fc, DATA_SW *b_fc, int PE);
void Softmax_layer(DATA_SW *in_array, float *out_array, int length);
int find_max_int(DATA_SW *in_array, int length);
float find_max_float(float *in_array, int length);
void model();

//HW FUNCTIONS

void MobileNet_Stream(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data, hls::stream<DATA_STREAM> &ext_residual_map_write, hls::stream<DATA_STREAM> &ext_residual_map_read,
						volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
						volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
						volatile DATA_SW *ext_tile, volatile DATA_SW *ext_info,
						DATA_HW layer, DATA_SW inter_layer, DATA_SW type_layer);

const DATA_HW CONV_size[19][3][8] ={   //layer, CONV_for_layer, CONV
    {{32, 3, 3, 2, w_layer_0, b_layer_0, 224, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}},  //CONV = out_map, in_map, kernel, stride, offset_kernel, offset_index, offset_bias, length_in_map, length_in_map^2
    {{32, 1, 3, 1, w_layer_1, b_layer_1, 112, 0}, {16, 32, 1, 1, w_layer_2, b_layer_2, 112, 12544}, {0, 0, 0, 0, 0, 0, 0}},
    {{96, 16, 1, 1, w_layer_3, b_layer_3, 112, 12544}, {96, 1, 3, 2, w_layer_4, b_layer_4, 112, 0}, {24, 96, 1, 1, w_layer_5, b_layer_5, 56, 3136}},
    {{144, 24, 1, 1, w_layer_6, b_layer_6, 56, 3136}, {144, 1, 3, 1, w_layer_7, b_layer_7, 56, 0}, {24, 144, 1, 1, w_layer_8, b_layer_8, 56, 3136}},
    {{144, 24, 1, 1, w_layer_9, b_layer_9, 56, 3136}, {144, 1, 3, 2, w_layer_10, b_layer_10, 56, 0}, {32, 144, 1, 1, w_layer_11, b_layer_11, 28, 784}},
    {{192, 32, 1, 1, w_layer_12, b_layer_12, 28, 784}, {192, 1, 3, 1, w_layer_13, b_layer_13, 28, 0}, {32, 192, 1, 1, w_layer_14, b_layer_14, 28, 784}},
    {{192, 32, 1, 1, w_layer_15, b_layer_15, 28, 784}, {192, 1, 3, 1, w_layer_16, b_layer_16, 28, 0}, {32, 192, 1, 1, w_layer_17, b_layer_17, 28, 784}},
    {{192, 32, 1, 1, w_layer_18, b_layer_18, 28, 784}, {192, 1, 3, 2, w_layer_19, b_layer_19, 28, 0}, {64, 192, 1, 1, w_layer_20, b_layer_20, 14, 196}},
    {{384, 64, 1, 1, w_layer_21, b_layer_21, 14, 196}, {384, 1, 3, 1, w_layer_22, b_layer_22, 14, 0}, {64, 384, 1, 1, w_layer_23, b_layer_23, 14, 196}},
    {{384, 64, 1, 1, w_layer_24, b_layer_24, 14, 196}, {384, 1, 3, 1, w_layer_25, b_layer_25, 14, 0}, {64, 384, 1, 1, w_layer_26, b_layer_26, 14, 196}},
    {{384, 64, 1, 1, w_layer_27, b_layer_27, 14, 196}, {384, 1, 3, 1, w_layer_28, b_layer_28, 14, 0}, {64, 384, 1, 1, w_layer_29, b_layer_29, 14, 196}},
    {{384, 64, 1, 1, w_layer_30, b_layer_30, 14, 196}, {384, 1, 3, 1, w_layer_31, b_layer_31, 14, 0}, {96, 384, 1, 1, w_layer_32, b_layer_32, 14, 196}},
    {{576, 96, 1, 1, w_layer_33, b_layer_33, 14, 196}, {576, 1, 3, 1, w_layer_34, b_layer_34, 14, 0}, {96, 576, 1, 1, w_layer_35, b_layer_35, 14, 196}},
    {{576, 96, 1, 1, w_layer_36, b_layer_36, 14, 196}, {576, 1, 3, 1, w_layer_37, b_layer_37, 14, 0}, {96, 576, 1, 1, w_layer_38, b_layer_38, 14, 196}},
    {{576, 96, 1, 1, w_layer_39, b_layer_39, 14, 196}, {576, 1, 3, 2, w_layer_40, b_layer_40, 14, 0}, {160, 576, 1, 1, w_layer_41, b_layer_41, 7, 49}},
    {{960, 160, 1, 1, w_layer_42, b_layer_42, 7, 49}, {960, 1, 3, 1, w_layer_43, b_layer_43, 7, 0}, {160, 960, 1, 1, w_layer_44, b_layer_44, 7, 49}},
    {{960, 160, 1, 1, w_layer_45, b_layer_45, 7, 49}, {960, 1, 3, 1, w_layer_46, b_layer_46, 7, 0}, {160, 960, 1, 1, w_layer_47, b_layer_47, 7, 49}},
    {{960, 160, 1, 1, w_layer_48, b_layer_48, 7, 49}, {960, 1, 3, 1, w_layer_49, b_layer_49, 7, 0}, {320, 960, 1, 1, w_layer_50, b_layer_50, 7, 49}},
    {{1280, 320, 1, 1, w_layer_51, b_layer_51, 7, 49}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0}},
};
const DATA_HW FC_size[2] = {1000, 1280};  //out_map, in_map

const DATA_HW CONV_res[19][3][2] ={   //store_map, residual
	{{0, 0}, {0, 0}, {0, 0}},
	{{0, 0}, {0, 0}, {0, 0}},
	{{0, 0}, {0, 0}, {1, 0}},
	{{0, 0}, {0, 0}, {0, 1}},
	{{0, 0}, {0, 0}, {1, 0}},
	{{0, 0}, {0, 0}, {1, 1}},
	{{0, 0}, {0, 0}, {0, 1}},
	{{0, 0}, {0, 0}, {1, 0}},
	{{0, 0}, {0, 0}, {1, 1}},
	{{0, 0}, {0, 0}, {1, 1}},
	{{0, 0}, {0, 0}, {0, 1}},
	{{0, 0}, {0, 0}, {1, 0}},
	{{0, 0}, {0, 0}, {1, 1}},
	{{0, 0}, {0, 0}, {0, 1}},
	{{0, 0}, {0, 0}, {1, 0}},
	{{0, 0}, {0, 0}, {1, 1}},
	{{0, 0}, {0, 0}, {0, 1}},
	{{0, 0}, {0, 0}, {0, 0}},
	{{0, 0}, {0, 0}, {0, 0}},
};

//
//PREPROCESSING
//

void image_read(char *filename, DATA_SW *vector){
    FILE* file = fopen (filename, "r");
    DATA_SW temp = 0;
    char read;
    int i = 0, sign = 1;

    while(1){
        read = fgetc(file);
        if (read != 32){
            if (read ==  '-'){
                sign = -1;
            }
            else{
                temp = temp*10 + sign*(read - 48);
            }
        }
        else{
            vector[i] = temp;
            temp = 0;
            sign = 1;
            i++;
        }
        if (i > IMAGE_SIZE*IMAGE_SIZE*3 - 1){
            break;
        }
    }

    fclose(file);
}

void weights_read(char *filename, DATA_SW *w_conv, DATA_SW *w_fc){
    FILE* file = fopen(filename, "r");
    char read;
    DATA_SW temp_CONV = 0;
    DATA_SW temp_FC = 0;
    int counter = 0, layer = 0, i = 0, sign = 1;

    //CONVs (weights)
    while(i <= w_conv_layer - 1){
        read = fgetc(file);
        if (read != 32){
            if (read ==  '-'){
                sign = -1;
            }
            else{
                temp_CONV = temp_CONV*10 + sign*(read - 48);
            }
        }
        else{
            w_conv[i] = temp_CONV;
            temp_CONV = 0;
            sign = 1;
            if (layer == 1){
                if (i == counter + CONV_size[layer][1][4]){
                    counter += CONV_size[layer][1][4];
                    layer++;
                }
            }
            else{
                if (i == counter + CONV_size[layer][2][4]){
                    counter += CONV_size[layer][2][4];
                    layer++;
                }
            }
            i++;
        }
    }
    //FC (weights)
    layer++;
    i = 0, sign = 1;
    while(i <= fc_layer - 1){
        read = fgetc(file);
        if (read != 32){
            if (read ==  '-'){
                sign = -1;
            }
            else{
                temp_FC = temp_FC*10 + sign*(read - 48);
            }
        }
        else{
            w_fc[i] = temp_FC;
            temp_FC = 0;
            sign = 1;
            i++;
        }
    }

    fclose(file);
}

void bias_read(char *filename, DATA_SW *b_conv, DATA_SW *b_fc){
    FILE* file = fopen(filename, "r");
    char read;
    DATA_SW temp_CONV = 0;
    DATA_SW temp_FC = 0;
    int counter = 0, layer = 0, i = 0, sign = 1;

    //CONVs (bias)
    i = 0, temp_CONV = 0, sign = 1;
    while(i <= b_conv_layer - 1){
        read = fgetc(file);
        if (read != 32){
            if (read ==  '-'){
                sign = -1;
            }
            else{
                temp_CONV = temp_CONV*10 + sign*(read - 48);
            }
        }
        else{
            b_conv[i] = temp_CONV;
            temp_CONV = 0;
            sign = 1;
            i++;
        }
    }
    //FC (bias)
    i = 0, temp_FC = 0, sign = 1;
    while(i <= FC_size[0] - 1){
        read = fgetc(file);
        if (read != 32){
            if (read ==  '-'){
                sign = -1;
            }
            else{
                temp_FC = temp_FC*10 + sign*(read - 48);
            }
        }
        else{
            b_fc[i] = temp_FC;
            temp_FC = 0;
            sign = 1;
            i++;
        }
    }

    fclose(file);
}

void class_read(char *filename, char **i_class){
    FILE* file = fopen (filename, "r");
    char read[1000];
    int i=0, j=0, k=0;
    int active=0;

    while(fgets(read, 1000, (FILE*) file)){
        while (read[i]!=0){
            if (read[i]==':'){   //compare with :
                i+=3;
                active=1;
            }
            else if (active){    //if char is after : and before '0', write in vector
                if (read[i+3]==0){
                    i_class[j][k]=0;
                    break;
                }
                i_class[j][k]=read[i];
                i++;
                k++;
            }
            else{   //else, add one to iterator
                i++;
            }
        }
        i=0;
        active=0;
        k=0;
        j++;
    }

    fclose(file);
}

//
//PROCESSING
//

//LAYERS
void CONV_BATCH_RELU(hls::stream<DATA_STREAM> &in_map, hls::stream<DATA_STREAM> &out_map, hls::stream<DATA_STREAM> &res_map_write, hls::stream<DATA_STREAM> &res_map_read, DATA_SW *w_conv, DATA_SW *b_conv, int layer, int inter_layer, int type_layer, int PE){
    //CONV layer 3x3
    if (type_layer == 0){
        MobileNet_Stream(in_map, out_map, res_map_write, res_map_read, w_conv, b_conv, weights_FC, bias_FC,
        		tile_3x3 + PE*3*MAX_CONV_3X3, info_3x3  + PE*size_info*MAX_CONV_3X3,
        		layer, inter_layer, 0);
    }
    //CONV layer 1x1
    else if (type_layer == 1){
        MobileNet_Stream(in_map, out_map, res_map_write, res_map_read, w_conv, b_conv, weights_FC, bias_FC,
        		tile_convs + inter_layer*3*MAX_CONVS + (layer-1)*3*MAX_CONVS*3 + PE*3*MAX_CONVS*3*18, info_convs + inter_layer*size_info*MAX_CONVS + (layer-1)*size_info*MAX_CONVS*3 + PE*size_info*MAX_CONVS*3*18,
        		layer, inter_layer, 1);
    }
    //depthwise layer
    else{
        MobileNet_Stream(in_map, out_map, res_map_write, res_map_read, w_conv, b_conv, weights_FC, bias_FC,
        		tile_convs + inter_layer*3*MAX_CONVS + (layer-1)*3*MAX_CONVS*3 + PE*3*MAX_CONVS*3*18, info_convs + inter_layer*size_info*MAX_CONVS + (layer-1)*size_info*MAX_CONVS*3 + PE*size_info*MAX_CONVS*3*18,
            layer, inter_layer, 2);
    }
}

void InvertedResidual(DATA_SW *res_map_write, DATA_SW *res_map_read, DATA_SW *w_conv, DATA_SW *b_conv, int layer, int length_in_map){
    int inter_layer, relu, type_layer, store_map = 0, expansion = 1, length = length_in_map/CONV_size[layer][1][3];
    if (CONV_size[layer][1][3] == 1){
    	expansion = 0;
    	length = length_in_map;
    }
    //GROUP 1
    inter_layer = 0;
    type_layer = 1;
    DRAM_2_STREAM_1x1(cpu_map, in_stream_0, in_stream_1, in_stream_2, in_stream_3,
    		CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][1], length_in_map, 1, type_layer, 0);
    //PE 0
    CONV_BATCH_RELU(in_stream_0, out_stream_0, res_write_stream_0, res_read_stream_0, w_conv, b_conv, layer, inter_layer, type_layer, 0);
    //PE 1
    CONV_BATCH_RELU(in_stream_1, out_stream_1, res_write_stream_0, res_read_stream_1, w_conv, b_conv, layer, inter_layer, type_layer, 1);
    //PE 2
    CONV_BATCH_RELU(in_stream_2, out_stream_2, res_write_stream_0, res_read_stream_2, w_conv, b_conv, layer, inter_layer, type_layer, 2);
    //PE 3
    CONV_BATCH_RELU(in_stream_3, out_stream_3, res_write_stream_0, res_read_stream_3, w_conv, b_conv, layer, inter_layer, type_layer, 3);
    STREAM_2_DRAM_1x1(out_stream_0, out_stream_1, out_stream_2, out_stream_3,
    		cpu_map, CONV_size[layer][inter_layer][0], length_in_map, 1);
    //GROUP 2
	inter_layer = 1;
	type_layer = 2;
	DRAM_2_STREAM_3x3(cpu_map, in_stream_0, in_stream_1, in_stream_2, in_stream_3,
			CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][0], length_in_map, type_layer);
	//PE 0
	CONV_BATCH_RELU(in_stream_0, out_stream_0, res_write_stream_0, res_read_stream_0, w_conv, b_conv, layer, inter_layer, type_layer, 0);
	//PE 1
	CONV_BATCH_RELU(in_stream_1, out_stream_0, res_write_stream_0, res_read_stream_1, w_conv, b_conv, layer, inter_layer, type_layer, 1);
	//PE 2
	CONV_BATCH_RELU(in_stream_2, out_stream_0, res_write_stream_0, res_read_stream_2, w_conv, b_conv, layer, inter_layer, type_layer, 2);
	//PE 3
	CONV_BATCH_RELU(in_stream_3, out_stream_0, res_write_stream_0, res_read_stream_3, w_conv, b_conv, layer, inter_layer, type_layer, 3);
	STREAM_2_DRAM_3x3(out_stream_0, out_stream_0, out_stream_0, out_stream_0,
			cpu_map, CONV_size[layer][inter_layer][0], length_in_map, CONV_size[layer][inter_layer][3], expansion);
	//GROUP 3
	inter_layer = 2;
	type_layer = 1;
	DRAM_2_STREAM_1x1(cpu_map, in_stream_0, in_stream_1, in_stream_2, in_stream_3,
			CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][1], length, CONV_size[layer][1][3], type_layer, expansion);
	DRAM_2_STREAM_res(res_map_read, res_read_stream_0, res_read_stream_1, res_read_stream_2, res_read_stream_3,
			CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][6], CONV_res[layer][inter_layer][1], CONV_res[layer][inter_layer][0]);
	//PE 0
	CONV_BATCH_RELU(in_stream_0, out_stream_0, res_write_stream_0, res_read_stream_0, w_conv, b_conv, layer, inter_layer, type_layer, 0);
	//PE 1
	CONV_BATCH_RELU(in_stream_1, out_stream_0, res_write_stream_0, res_read_stream_1, w_conv, b_conv, layer, inter_layer, type_layer, 1);
	//PE 2
	CONV_BATCH_RELU(in_stream_2, out_stream_0, res_write_stream_0, res_read_stream_2, w_conv, b_conv, layer, inter_layer, type_layer, 2);
	//PE 3
	CONV_BATCH_RELU(in_stream_3, out_stream_0, res_write_stream_0, res_read_stream_3, w_conv, b_conv, layer, inter_layer, type_layer, 3);
	if (CONV_res[layer][inter_layer][0]){   //store residual
		STREAM_2_DRAM_res(res_write_stream_0, res_write_stream_1, res_write_stream_2, res_write_stream_3,
				res_map_write, CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][6]);
	}
	STREAM_2_DRAM_1x1(out_stream_0, out_stream_1, out_stream_2, out_stream_3,
			cpu_map, CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][6], 0);
}

void AVG(hls::stream<DATA_STREAM> &in_map, hls::stream<DATA_STREAM> &out_array, hls::stream<DATA_STREAM> &res_map_write, hls::stream<DATA_STREAM> &res_map_read, int PE){
    MobileNet_Stream(in_map, out_array, res_map_write, res_map_read, weights_CONV, bias_CONV, weights_FC, bias_FC,
    		tile_avg + PE*3*MAX_AVG, info_avg + PE*size_info*MAX_AVG,
    	0, 0, 3);
}

void Fully_Connected_layer(hls::stream<DATA_STREAM> &in_array, hls::stream<DATA_STREAM> &out_array, hls::stream<DATA_STREAM> &res_map_write, hls::stream<DATA_STREAM> &res_map_read, DATA_SW *w_fc, DATA_SW *b_fc, int PE){
    MobileNet_Stream(in_array, out_array, res_map_write, res_map_read, weights_CONV, bias_CONV, w_fc, b_fc,
    		tile_fc + PE*3*MAX_FC, info_fc + PE*size_info*MAX_FC,
        0, 0, 4);
}

void Softmax_layer(DATA_SW *in_array, float *out_array, int length){
    int max_position, value_max, quant = softmax_quant;
    float int2float, exp_result, sum = 0;

    max_position = find_max_int(in_array, length);
    value_max = in_array[max_position];

    for (int i = 0; i < length; i++){
        int2float = (in_array[i] - value_max) >> quant;
        exp_result = exp(int2float);
        sum += exp_result;
        out_array[i] = exp_result;
    }

    for (int i=0; i<length; i++){
        out_array[i] /= sum;
    }
}

//MATH

int find_max_int(DATA_SW *in_array, int length){
    int iter = 0;
    for (int i = 0; i < length; i++){
        if(in_array[i] > in_array[iter]){
            iter = i;
        }
    }
    return iter;
}

float find_max_float(float *in_array, int length){
    int iter = 0;
    for (int i = 0; i < length; i++){
        if(in_array[i] > in_array[iter]){
            iter = i;
        }
    }
    return iter;
}

//
//MODEL
//

void model(){
    int layer, inter_layer, length, relu, type_layer, residual;

    ///CONVBNReLU 0
    printf("\nCONVBNReLU 0 ...");
    layer = 0;
    length = 224;
    inter_layer = 0;
    relu = 1;
    type_layer = 0;
    residual = 0;
    DRAM_2_STREAM_3x3(cpu_map, in_stream_0, in_stream_1, in_stream_2, in_stream_3,
    		CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][1], length, type_layer);
    //PE 0
    CONV_BATCH_RELU(in_stream_0, out_stream_0, res_write_stream_0, res_read_stream_0, weights_CONV, bias_CONV, 0, 0, 0, 0);
    //PE 1
    CONV_BATCH_RELU(in_stream_1, out_stream_1, res_write_stream_1, res_read_stream_1, weights_CONV, bias_CONV, 0, 0, 0, 1);
    //PE 2
    CONV_BATCH_RELU(in_stream_2, out_stream_2, res_write_stream_2, res_read_stream_2, weights_CONV, bias_CONV, 0, 0, 0, 2);
    //PE 3
    CONV_BATCH_RELU(in_stream_3, out_stream_3, res_write_stream_3, res_read_stream_3, weights_CONV, bias_CONV, 0, 0, 0, 3);
    STREAM_2_DRAM_3x3(out_stream_0, out_stream_1, out_stream_2, out_stream_3,
    		cpu_map, CONV_size[layer][inter_layer][0], length, CONV_size[layer][inter_layer][3], 1);

    ///InvertedResidual 1
    printf("\nInvertedResidual 1 ...");
    layer = 1;
    length = 112;
    //Group 1
    inter_layer = 0;
    relu = 1;
    type_layer = 2;
    residual = 0;
    DRAM_2_STREAM_3x3(cpu_map, in_stream_0, in_stream_1, in_stream_2, in_stream_3,
			CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][0], length, type_layer);
    //PE 0
    CONV_BATCH_RELU(in_stream_0, out_stream_0, res_write_stream_0, res_read_stream_0, weights_CONV, bias_CONV, 1, 0, 2, 0);
    //PE 1
    CONV_BATCH_RELU(in_stream_1, out_stream_0, res_write_stream_1, res_read_stream_1, weights_CONV, bias_CONV, 1, 0, 2, 1);
    //PE 2
    CONV_BATCH_RELU(in_stream_2, out_stream_0, res_write_stream_2, res_read_stream_2, weights_CONV, bias_CONV, 1, 0, 2, 2);
    //PE 3
    CONV_BATCH_RELU(in_stream_3, out_stream_0, res_write_stream_3, res_read_stream_3, weights_CONV, bias_CONV, 1, 0, 2, 3);
    STREAM_2_DRAM_3x3(out_stream_0, out_stream_0, out_stream_0, out_stream_0,
    		cpu_map, CONV_size[layer][inter_layer][0], length, CONV_size[layer][inter_layer][3], 0);
    //Group 2
    inter_layer = 1;
    relu = 0;
    type_layer = 1;
    DRAM_2_STREAM_1x1(cpu_map, in_stream_0, in_stream_1, in_stream_2, in_stream_3,
    		CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][1], length, 1, type_layer, 0);
    //PE 0
    CONV_BATCH_RELU(in_stream_0, out_stream_0, res_write_stream_0, res_read_stream_0, weights_CONV, bias_CONV, 1, 1, 1, 0);
    //PE 1
    CONV_BATCH_RELU(in_stream_1, out_stream_0, res_write_stream_1, res_read_stream_1, weights_CONV, bias_CONV, 1, 1, 1, 1);
    //PE 2
    CONV_BATCH_RELU(in_stream_2, out_stream_0, res_write_stream_2, res_read_stream_2, weights_CONV, bias_CONV, 1, 1, 1, 2);
    //PE 3
    CONV_BATCH_RELU(in_stream_3, out_stream_0, res_write_stream_3, res_read_stream_3, weights_CONV, bias_CONV, 1, 1, 1, 3);
    STREAM_2_DRAM_1x1(out_stream_0, out_stream_1, out_stream_2, out_stream_3,
    		cpu_map, CONV_size[layer][inter_layer][0], length, 0);

    ///InvertedResidual 2
    printf("\nInvertedResidual 2 ...");
    layer = 2;
    length = 112;
    residual = 0;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 2, 112);

    ///InvertedResidual 3
    printf("\nInvertedResidual 3 ...");
    layer = 3;
    length = 56;
    residual = 1;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 3, 56);

    ///InvertedResidual 4
    printf("\nInvertedResidual 4 ...");
    layer = 4;
    length = 56;
    residual = 0;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 4, 56);

    ///InvertedResidual 5
    printf("\nInvertedResidual 5 ...");
    layer = 5;
    length = 28;
    residual = 1;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 5, 28);

    ///InvertedResidual 6
    printf("\nInvertedResidual 6 ...");
    layer = 6;
    length = 28;
    residual = 1;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 6, 28);

    ///InvertedResidual 7
    printf("\nInvertedResidual 7 ...");
    layer = 7;
    length = 28;
    residual = 0;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 7, 28);

    ///InvertedResidual 8
    printf("\nInvertedResidual 8 ...");
    layer = 8;
    length = 14;
    residual = 1;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 8, 14);

    ///InvertedResidual 9
    printf("\nInvertedResidual 9 ...");
    layer = 9;
    length = 14;
    residual = 1;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 9, 14);

    ///InvertedResidual 10
    printf("\nInvertedResidual 10 ...");
    layer = 10;
    length = 14;
    residual = 1;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 10, 14);

    ///InvertedResidual 11
    printf("\nInvertedResidual 11 ...");
    layer = 11;
    length = 14;
    residual = 0;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 11, 14);

    ///InvertedResidual 12
    printf("\nInvertedResidual 12 ...");
    layer = 12;
    length = 14;
    residual = 1;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 12, 14);

    ///InvertedResidual 13
    printf("\nInvertedResidual 13 ...");
    layer = 13;
    length = 14;
    residual = 1;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 13, 14);

    ///InvertedResidual 14
    printf("\nInvertedResidual 14 ...");
    layer = 14;
    length = 14;
    residual = 0;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 14, 14);

    ///InvertedResidual 15
    printf("\nInvertedResidual 15 ...");
    layer = 15;
    length = 7;
    residual = 1;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 15, 7);

    ///InvertedResidual 16
    printf("\nInvertedResidual 16 ...");
    layer = 16;
    length = 7;
    residual = 1;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 16, 7);

    ///InvertedResidual 17
    printf("\nInvertedResidual 17 ...");
    layer = 17;
    length = 7;
    residual = 0;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 17, 7);

    ///CONVBNReLU 18
    printf("\nCONVBNReLU 18 ...");
    layer = 18;
    length = 7;
    inter_layer = 0;
    relu = 1;
    type_layer = 1;
    residual = 0;
    DRAM_2_STREAM_1x1(cpu_map, in_stream_0, in_stream_1, in_stream_2, in_stream_3,
			CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][1], length, 1, type_layer, 0);
    //PE 0
    CONV_BATCH_RELU(in_stream_0, out_stream_0, res_write_stream_0, res_read_stream_0, weights_CONV, bias_CONV, 18, 0, 1, 0);
    //PE 1
    CONV_BATCH_RELU(in_stream_1, out_stream_1, res_write_stream_1, res_read_stream_1, weights_CONV, bias_CONV, 18, 0, 1, 1);
    //PE 2
    CONV_BATCH_RELU(in_stream_2, out_stream_2, res_write_stream_2, res_read_stream_2, weights_CONV, bias_CONV, 18, 0, 1, 2);
    //PE 3
    CONV_BATCH_RELU(in_stream_3, out_stream_3, res_write_stream_3, res_read_stream_3, weights_CONV, bias_CONV, 18, 0, 1, 3);
    STREAM_2_DRAM_1x1(out_stream_0, out_stream_1, out_stream_2, out_stream_3,
			cpu_map, CONV_size[layer][inter_layer][0], length, 1);

    ///Flatten 19
    printf("\nGlobal AVG 19 ...");
    DRAM_2_STREAM_1x1(cpu_map, in_stream_0, in_stream_1, in_stream_2, in_stream_3,
    		1, CONV_size[18][0][0], 7, 1, 3, 1);
    //PE 0
    AVG(in_stream_0, out_stream_0, res_write_stream_0, res_read_stream_0, 0);
    //PE 1
    AVG(in_stream_1, out_stream_1, res_write_stream_1, res_read_stream_1, 1);
    //PE 2
    AVG(in_stream_2, out_stream_2, res_write_stream_2, res_read_stream_2, 2);
    //PE 3
    AVG(in_stream_3, out_stream_3, res_write_stream_3, res_read_stream_3, 3);
    STREAM_2_DRAM_array(out_stream_0, out_stream_1, out_stream_2, out_stream_3,
    		cpu_array, FC_size[1]);

    ///FC 20
    printf("\nFC 19 ...");
    write_txt (cpu_array, res_map_A, res_map_B, 18, 0, 4, 1);
    DRAM_2_STREAM_array(cpu_array, in_stream_0, in_stream_1, in_stream_2, in_stream_3,
    		FC_size[0], FC_size[1]);
    //PE 0
    Fully_Connected_layer(in_stream_0, out_stream_0, res_write_stream_0, res_read_stream_0, weights_FC, bias_FC, 0);
    //PE 1
    Fully_Connected_layer(in_stream_1, out_stream_1, res_write_stream_1, res_read_stream_1, weights_FC, bias_FC, 1);
    //PE 2
    Fully_Connected_layer(in_stream_2, out_stream_2, res_write_stream_2, res_read_stream_2, weights_FC, bias_FC, 2);
    //PE 3
    Fully_Connected_layer(in_stream_3, out_stream_3, res_write_stream_3, res_read_stream_3, weights_FC, bias_FC, 3);
    STREAM_2_DRAM_array(out_stream_0, out_stream_1, out_stream_2, out_stream_3,
    		cpu_array, FC_size[0]);

    ///Softmax
    printf("\nSoftmax ...");
    Softmax_layer(cpu_array, array_softmax, FC_size[0]);
}

//
//MAIN
//

int main(){
//Path_files
    char *image_path;
    char *weights_path, *bias_path;
    char *imagenet_path;
    clock_t t_ini, t_fin;
    double secs;

    image_path = (char*) "image_int.dat";
    imagenet_path = (char*) "imagenet_class.txt";
    //PC UNIVERSIDAD
	weights_path = (char*) "PATH\\weights.dat";
	bias_path = (char*) "PATH\\bias.dat";

    //read image
    image_read(image_path, cpu_map);

    //read weights MobileNet
    t_ini = clock();
    weights_read(weights_path, weights_CONV, weights_FC);
    bias_read(bias_path, bias_CONV, bias_FC);
    t_fin = clock();

	//set info and tile
	set_tile_info(tile_3x3, tile_convs, tile_avg, tile_fc,
				  info_3x3, info_convs, info_avg, info_fc);

    secs = (double)(t_fin - t_ini) / CLOCKS_PER_SEC;
    printf("\nTime for read: ");
    printf("%.16g seconds\n", secs);

    //read class ImageNet
    class_ImageNet = (char **) malloc (1000 * sizeof(char*));
    for (int i = 0; i < 1000; i++){
        class_ImageNet[i] = (char*) malloc(1000 * sizeof(char));
    }
    class_read(imagenet_path, class_ImageNet);

    //model
    t_ini = clock();
    model();
    t_fin = clock();

    secs = (double)(t_fin - t_ini) / CLOCKS_PER_SEC;
    printf("\n\nTime for inference: ");
    printf("%.16g seconds\n", secs);

    //Search TOP-5 results
    int max_position, index = 0, top = 5;

    printf("\n\nTop-%d:\n\n", top);
    for (int j = 0; j < top; j++){
        index = 0;
        max_position = find_max_float(array_softmax, FC_size[0]);

        while(class_ImageNet[max_position][index]!=0){
            printf("%c", class_ImageNet[max_position][index]);
            index++;
        }
        printf(": %.2f %c\n", array_softmax[max_position] * 100, 37);
        array_softmax[max_position] = 0;
    }

    //test (erase)
    //weights
    /*for (int i = 0; i < 9; i++){
        printf("%d %d %d\n", weights_CONV[i]>>12, (weights_CONV[i]<<(32-12))>>(32-12), weights_CONV[i]);
    }*/
    //image
    /*for (int k = 0; k < 3; k ++){
        for (int i = 0; i < 224; i++){
            for (int j = 0; j < 224; j++){
                printf("%d ", image[j + i*224 + k*224*224]);
            }
            printf("\n");
        }
        printf("\n");
    }*/
    //feature
    /*int layer = 0;
    int inter_layer = 0;
    int length = CONV_size[layer][inter_layer][6]/CONV_size[layer][inter_layer][3];
    //DATA_STREAM val;
    //rder2CPU_map(res_map_2, res_map_1, CONV_size[layer][inter_layer][0], length, CONV_size[layer][inter_layer][3]);
    for (int i = 0; i < length; i++){
        for (int j = 0; j < length; j++){
        	//out_stream_0.read(val);
        	//printf("%d ", val.data);
            printf("%d ", cpu_map[j + i*length + 31*length*length]);
        }
        printf("\n");
    }*/
    //in map order 3x3
    /*for (int lim = 16*16*1; lim < (224/14)*(224/14)*16*16*3; lim += 16*16*3){
		for (int i = 0; i < 16; i++){
			for (int j = 0; j < 16; j++){
				printf("%d ", out_fea_2[j + i*16 + lim]);
			}
			printf("\n");
		}
		printf("\n");
    }*/
    //in map order 1x1 or out
    /*for (int lim = 14*14*0; lim < (112/14)*(112/14)*14*14*32; lim += 14*14*32){
		for (int i = 0; i < 14; i++){
			for (int j = 0; j < 14; j++){
				printf("%d ", out_fea_1[j + i*14 + lim]);
			}
			printf("\n");
		}
		printf("\n");
    }*/
    //FC
    /*for (int i = 0; i < 1280; i++){
        printf("%d:  %d\n", i, cpu_array[i]);
    }*/
}
