#include "parameters.h"
#include "layer.h"

//SIZE WEIGHTS
int w_conv_layer = 1525656;
int fc_layer = 360000;
int b_conv_layer = 17056;
int b_fc_layer = 1000;

//Pruning DATA
float sparsity_CONV = 0.3;
float sparsity_FC = 0.7;

int number_inter_layer[19] ={1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1}; //CONV and BATCH
int feature_extraction_outs[3]={IMAGE_SIZE, IMAGE_SIZE, 1280};   //width, length, depth
int residual_CONV[19] = {0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0};   //residual layers

//MAP and ARRAY size
int map_size = 1572864;   //(112/14)*(112/14)*16*16*96
int array_size = 1280;

//Multi_PE conv 1x1 and FC
int multi_PE_CONV = tile_conv_in;
int multi_PE_FC = tile_fc_in;

//Tiling factor
int tiling_CONV[4] = {tile_map, tile_map, tile_conv_out, tile_conv_in}; //size_map, size_map, out_map, in_map (in_map tiling same of multi_PE pruning structured in conv 1x1)
int tiling_FC[2] = {tile_fc_out, tile_fc_in}; //out_map, in_map

//Info layers
int CONV_size[19][3][8] ={   //layer, CONV_for_layer, CONV
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

int FC_size[2] = {1000, 1280};

int CONV_res[19][3][2] ={   //store_map, residual
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
