#ifndef PARAMETERS_H
#define PARAMETERS_H

//ADDRESS BRAM
//AXI Stream
//PE 0
#define IN_STREAM_ADDR_0		0x01000000
#define OUT_STREAM_ADDR_0		0x02000000
#define RES_READ_STREAM_ADDR_0	0x41000000
#define RES_WRITE_STREAM_ADDR_0	0x42000000
//PE 1
#define IN_STREAM_ADDR_1		0x03000000
#define OUT_STREAM_ADDR_1		0x04000000
#define RES_READ_STREAM_ADDR_1	0x43000000
#define RES_WRITE_STREAM_ADDR_1	0x44000000
//PE 2
#define IN_STREAM_ADDR_2		0x05000000
#define OUT_STREAM_ADDR_2		0x06000000
#define RES_READ_STREAM_ADDR_2	0x45000000
#define RES_WRITE_STREAM_ADDR_2	0x46000000
//PE 3
#define IN_STREAM_ADDR_3		0x07000000
#define OUT_STREAM_ADDR_3		0x08000000
#define RES_READ_STREAM_ADDR_3	0x47000000
#define RES_WRITE_STREAM_ADDR_3	0x48000000
//Weights Masters
#define W_CONV_ADDR				0x21000000
#define B_CONV_ADDR				0x22000000
#define W_FC_ADDR				0x23000000
#define B_FC_ADDR				0x24000000
//Info Master
#define TILE_3X3_ADDR			0x25000000
#define TILE_CONVS_ADDR			0x26000000
#define TILE_AVG_ADDR			0x27000000
#define TILE_FC_ADDR			0x28000000
#define INFO_3X3_ADDR			0x29000000
#define INFO_CONVS_ADDR			0x2A000000
#define INFO_AVG_ADDR			0x2B000000
#define INFO_FC_ADDR			0x2C000000
#define CONV_QUANT_ADDR			0x2D000000
#define AVG_QUANT_ADDR			0x2E000000
#define FC_QUANT_ADDR			0x2F000000

//GENERAL PARAMETERS
#define IMAGE_SIZE 				224
#define MAX_SIZE_KERNEL 		3
#define BASEADDR_MEM			0x60000000
#define UART_BASEADDR			XPAR_XUARTPS_0_BASEADDR

//Tiling factor
#define tile_map            	28
#define tile_conv_in        	32
#define tile_conv_out       	32
#define tile_fc_in          	32
#define tile_fc_out         	64

//Length BRAM
#define in_map_LEN				(tile_map + 2)*(tile_map + 2)*tile_conv_out
#define out_map_LEN				tile_map*tile_map*tile_conv_out
#define save_input_LEN 			86400   //60*60*24 (56*56*24 bigger save input but axi_data use 3.5=4 reg more (56+4=60))
#define w_conv_LEN				tile_conv_in*tile_conv_out*3
#define w_fc_LEN				tile_fc_in*tile_fc_out

//Max call of PE
#define MAX_CONV_3X3 			64
#define MAX_CONVS				100
#define MAX_AVG					10
#define MAX_FC					160

//size info
#define size_info				19

//number PEs
#define number_PE				4

//Softmax quant
#define softmax_quant 			9

//SIZE WEIGHTS
extern int w_conv_layer;
extern int fc_layer;
extern int b_conv_layer;
extern int b_fc_layer;

//Pruning DATA
extern float sparsity_CONV;
extern float sparsity_FC;

extern int number_inter_layer[19];
extern int feature_extraction_outs[3];
extern int residual_CONV[19];

//MAP size
extern int map_size;
extern int array_size;

//Multi_PE conv 1x1 and FC
extern int multi_PE_CONV;
extern int multi_PE_FC;

//Tiling factor
extern int tiling_CONV[4];
extern int tiling_FC[2];

//Registers
extern int CONV_size[19][3][8];
extern int FC_size[2];
extern int CONV_res[19][3][2];

#endif
