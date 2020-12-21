#include <stdio.h>
#include <stdlib.h>
#include "platform.h"
#include "xil_printf.h"
#include "xuartps.h"
#include "xparameters.h"
#include "ff.h"
#include <math.h>
#include "xil_cache.h"
#include "xtime_l.h"
#include "xparameters.h"
#include "xmobilenet_stream.h"
#include "xaxidma.h"
#include "memory_access.h"
#include "parameters.h"
#include "read_write_functions.h"
#include "type_data.h"

//VECTORS
W_CONV *weights_CONV;
W_CONV *bias_CONV;
W_FC *weights_FC;
W_FC *bias_FC;
char **class_ImageNet;
DATA *axi_stream_in;
DATA *axi_stream_out;
DATA *axi_stream_res_read;
DATA *axi_stream_res_write;
//CPU
DATA *cpu_map;
DATA *cpu_array;
DATA *res_map_A;
DATA *res_map_B;
float *array_softmax;
uint8_t *uart_send;
DATA *tile_3x3;
DATA *tile_convs;
DATA *tile_avg;
DATA *tile_fc;
DATA *info_3x3;
DATA *info_convs;
DATA *info_avg;
DATA *info_fc;
DATA *conv_quant;
DATA *avg_quant;
DATA *fc_quant;

// FUNCTIONS
void init_memory();
void free_memory();
void CONV_BATCH_RELU(DATA *in_map_0, DATA *out_map_0, DATA *res_map_write_0, DATA *res_map_read_0,
					 W_CONV *w_conv, W_CONV *b_conv,
					 int layer, int inter_layer, int type_layer, int len_in, int len_out, int len_res_write, int len_res_read);
void InvertedResidual(DATA *res_map_write, DATA *res_map_read, W_CONV *w_conv, W_CONV *b_conv, int layer, int length_in_map);
void AVG(DATA *in_map_0, DATA *out_array_0,
		DATA len_in, DATA len_out);
void Fully_Connected_layer(DATA *in_array_0, DATA *out_array_0,
		W_FC *w_fc, W_FC *b_fc, DATA len_in, DATA len_out);
void Softmax_layer(ACT_FC *in_array, float *out_array, int length);
int find_max_int(ACT_FC *in_array, int length);
float find_max_float(float *in_array, int length);
int MIN(int x, int y);
void model(DATA first_len_in);

//HARDWARE

//PE 0
XMobilenet_stream doMobilenet_stream_0;
XMobilenet_stream_Config *doMobilenet_stream_cfg_0;
XAxiDma axiDMA_data_0;
XAxiDma_Config *axiDMA_cfg_data_0;
XAxiDma axiDMA_res_0;
XAxiDma_Config *axiDMA_cfg_res_0;
//PE 1
XMobilenet_stream doMobilenet_stream_1;
XMobilenet_stream_Config *doMobilenet_stream_cfg_1;
XAxiDma axiDMA_data_1;
XAxiDma_Config *axiDMA_cfg_data_1;
XAxiDma axiDMA_res_1;
XAxiDma_Config *axiDMA_cfg_res_1;
//PE 2
XMobilenet_stream doMobilenet_stream_2;
XMobilenet_stream_Config *doMobilenet_stream_cfg_2;
XAxiDma axiDMA_data_2;
XAxiDma_Config *axiDMA_cfg_data_2;
XAxiDma axiDMA_res_2;
XAxiDma_Config *axiDMA_cfg_res_2;
//PE 3
XMobilenet_stream doMobilenet_stream_3;
XMobilenet_stream_Config *doMobilenet_stream_cfg_3;
XAxiDma axiDMA_data_3;
XAxiDma_Config *axiDMA_cfg_data_3;
XAxiDma axiDMA_res_3;
XAxiDma_Config *axiDMA_cfg_res_3;

void init_PL(){
	int status = 0;

	//PE 0
	//MobileNet_stream
	doMobilenet_stream_cfg_0 = XMobilenet_stream_LookupConfig(XPAR_MOBILENET_STREAM_0_DEVICE_ID);

	if (doMobilenet_stream_cfg_0){
		status = XMobilenet_stream_CfgInitialize(&doMobilenet_stream_0, doMobilenet_stream_cfg_0);
	}

	//axiDMA_data
	axiDMA_cfg_data_0 = XAxiDma_LookupConfig(XPAR_AXI_DMA_DATA_0_DEVICE_ID);
	if (axiDMA_cfg_data_0){
		status = XAxiDma_CfgInitialize(&axiDMA_data_0, axiDMA_cfg_data_0);
	}
	//Disable interrupts
	XAxiDma_IntrDisable(&axiDMA_data_0, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&axiDMA_data_0, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

	//AXIDMA_res
	axiDMA_cfg_res_0 = XAxiDma_LookupConfig(XPAR_AXI_DMA_RES_0_DEVICE_ID);
	if (axiDMA_cfg_res_0){
		status = XAxiDma_CfgInitialize(&axiDMA_res_0, axiDMA_cfg_res_0);
	}
	//Disable interrupts
	XAxiDma_IntrDisable(&axiDMA_res_0, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&axiDMA_res_0, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

	//PE 1
	//MobileNet_stream
	doMobilenet_stream_cfg_1 = XMobilenet_stream_LookupConfig(XPAR_MOBILENET_STREAM_1_DEVICE_ID);

	if (doMobilenet_stream_cfg_1){
		status = XMobilenet_stream_CfgInitialize(&doMobilenet_stream_1, doMobilenet_stream_cfg_1);
	}

	//axiDMA_data
	axiDMA_cfg_data_1 = XAxiDma_LookupConfig(XPAR_AXI_DMA_DATA_1_DEVICE_ID);
	if (axiDMA_cfg_data_1){
		status = XAxiDma_CfgInitialize(&axiDMA_data_1, axiDMA_cfg_data_1);
	}
	//Disable interrupts
	XAxiDma_IntrDisable(&axiDMA_data_1, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&axiDMA_data_1, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

	//AXIDMA_res
	axiDMA_cfg_res_1 = XAxiDma_LookupConfig(XPAR_AXI_DMA_RES_1_DEVICE_ID);
	if (axiDMA_cfg_res_1){
		status = XAxiDma_CfgInitialize(&axiDMA_res_1, axiDMA_cfg_res_1);
	}
	//Disable interrupts
	XAxiDma_IntrDisable(&axiDMA_res_1, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&axiDMA_res_1, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

	//PE 2
	//MobileNet_stream
	doMobilenet_stream_cfg_2 = XMobilenet_stream_LookupConfig(XPAR_MOBILENET_STREAM_2_DEVICE_ID);

	if (doMobilenet_stream_cfg_2){
		status = XMobilenet_stream_CfgInitialize(&doMobilenet_stream_2, doMobilenet_stream_cfg_2);
	}

	//axiDMA_data
	axiDMA_cfg_data_2 = XAxiDma_LookupConfig(XPAR_AXI_DMA_DATA_2_DEVICE_ID);
	if (axiDMA_cfg_data_2){
		status = XAxiDma_CfgInitialize(&axiDMA_data_2, axiDMA_cfg_data_2);
	}
	//Disable interrupts
	XAxiDma_IntrDisable(&axiDMA_data_2, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&axiDMA_data_2, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

	//AXIDMA_res
	axiDMA_cfg_res_2 = XAxiDma_LookupConfig(XPAR_AXI_DMA_RES_2_DEVICE_ID);
	if (axiDMA_cfg_res_2){
		status = XAxiDma_CfgInitialize(&axiDMA_res_2, axiDMA_cfg_res_2);
	}
	//Disable interrupts
	XAxiDma_IntrDisable(&axiDMA_res_2, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&axiDMA_res_2, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

	//PE 3
	//MobileNet_stream
	doMobilenet_stream_cfg_3 = XMobilenet_stream_LookupConfig(XPAR_MOBILENET_STREAM_3_DEVICE_ID);

	if (doMobilenet_stream_cfg_3){
		status = XMobilenet_stream_CfgInitialize(&doMobilenet_stream_3, doMobilenet_stream_cfg_3);
	}

	//axiDMA_data
	axiDMA_cfg_data_3 = XAxiDma_LookupConfig(XPAR_AXI_DMA_DATA_3_DEVICE_ID);
	if (axiDMA_cfg_data_3){
		status = XAxiDma_CfgInitialize(&axiDMA_data_3, axiDMA_cfg_data_3);
	}
	//Disable interrupts
	XAxiDma_IntrDisable(&axiDMA_data_3, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&axiDMA_data_3, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

	//AXIDMA_res
	axiDMA_cfg_res_3 = XAxiDma_LookupConfig(XPAR_AXI_DMA_RES_3_DEVICE_ID);
	if (axiDMA_cfg_res_3){
		status = XAxiDma_CfgInitialize(&axiDMA_res_3, axiDMA_cfg_res_3);
	}
	//Disable interrupts
	XAxiDma_IntrDisable(&axiDMA_res_3, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&axiDMA_res_3, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);
}


/*
 * PREPROCESSING
 */

void init_memory(){
    //Weights
    //CONVs
    weights_CONV = (W_CONV*) W_CONV_ADDR;
    bias_CONV = (W_CONV*) B_CONV_ADDR;

    //FC
    weights_FC = (W_FC*) W_FC_ADDR;
    bias_FC = (W_FC*) B_FC_ADDR;

    //AXI Stream
    axi_stream_in = (DATA*) IN_STREAM_ADDR_0;
    axi_stream_out = (DATA*) OUT_STREAM_ADDR_0;
    axi_stream_res_read = (DATA*) RES_READ_STREAM_ADDR_0;
    axi_stream_res_write = (DATA*) RES_WRITE_STREAM_ADDR_0;

    //Outs layers and residual features_extraction (conv, relu, batch)
    cpu_map = (DATA*) BASEADDR_MEM;
    cpu_array = cpu_map + map_size;
    res_map_A = cpu_array + array_size;
    res_map_B = res_map_A + save_input_LEN;

    //Array Softmax
    array_softmax = res_map_B + save_input_LEN;

    //UART SEND DATA
    uart_send = array_softmax + 1000;

    //TILE
    tile_3x3 = (DATA*) TILE_3X3_ADDR;
	tile_convs = (DATA*) TILE_CONVS_ADDR;
    tile_avg = (DATA*) TILE_AVG_ADDR;
    tile_fc = (DATA*) TILE_FC_ADDR;

    //INFO
    info_3x3 = (DATA*) INFO_3X3_ADDR;
	info_convs = (DATA*) INFO_CONVS_ADDR;
    info_avg = (DATA*) INFO_AVG_ADDR;
    info_fc = (DATA*) INFO_FC_ADDR;

    //Quant
    conv_quant = (DATA*) CONV_QUANT_ADDR;
    avg_quant = (DATA*) AVG_QUANT_ADDR;
    fc_quant = (DATA*) FC_QUANT_ADDR;
}

void free_memory(){
	free(cpu_map);
	free(cpu_array);
	free(res_map_A);
	free(res_map_B);
    free(array_softmax);
    free(uart_send);
}

/*
 * PROCESSING
 */

//LAYERS
void CONV_BATCH_RELU(DATA *in_map_0, DATA *out_map_0, DATA *res_map_write_0, DATA *res_map_read_0,
					 W_CONV *w_conv, W_CONV *b_conv,
					 int layer, int inter_layer, int type_layer, int len_in, int len_out, int len_res_write, int len_res_read){
    //LOAD DATA in PL
	//PE 0
    XMobilenet_stream_Set_layer(&doMobilenet_stream_0, layer);
    XMobilenet_stream_Set_inter_layer(&doMobilenet_stream_0, inter_layer);
    XMobilenet_stream_Set_type_layer(&doMobilenet_stream_0, type_layer);
	//PE 1
    XMobilenet_stream_Set_layer(&doMobilenet_stream_1, layer);
    XMobilenet_stream_Set_inter_layer(&doMobilenet_stream_1, inter_layer);
    XMobilenet_stream_Set_type_layer(&doMobilenet_stream_1, type_layer);
	//PE 2
    XMobilenet_stream_Set_layer(&doMobilenet_stream_2, layer);
    XMobilenet_stream_Set_inter_layer(&doMobilenet_stream_2, inter_layer);
    XMobilenet_stream_Set_type_layer(&doMobilenet_stream_2, type_layer);
	//PE 3
    XMobilenet_stream_Set_layer(&doMobilenet_stream_3, layer);
    XMobilenet_stream_Set_inter_layer(&doMobilenet_stream_3, inter_layer);
    XMobilenet_stream_Set_type_layer(&doMobilenet_stream_3, type_layer);

	//CONV layer 3x3
	if (type_layer == 0){
		//Set address from DRAM in PL
		XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_0, tile_3x3 + 0*3*MAX_CONV_3X3);
		XMobilenet_stream_Set_ext_info(&doMobilenet_stream_0, info_3x3  + 0*size_info*MAX_CONV_3X3);
		XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_1, tile_3x3 + 1*3*MAX_CONV_3X3);
		XMobilenet_stream_Set_ext_info(&doMobilenet_stream_1, info_3x3  + 1*size_info*MAX_CONV_3X3);
		XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_2, tile_3x3 + 2*3*MAX_CONV_3X3);
		XMobilenet_stream_Set_ext_info(&doMobilenet_stream_2, info_3x3  + 2*size_info*MAX_CONV_3X3);
		XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_3, tile_3x3 + 3*3*MAX_CONV_3X3);
		XMobilenet_stream_Set_ext_info(&doMobilenet_stream_3, info_3x3  + 3*size_info*MAX_CONV_3X3);

		/*Process*/
		XMobilenet_stream_Start(&doMobilenet_stream_0);
		XMobilenet_stream_Start(&doMobilenet_stream_1);
		XMobilenet_stream_Start(&doMobilenet_stream_2);
		XMobilenet_stream_Start(&doMobilenet_stream_3);
	    //Flush cache
	    Xil_DCacheFlushRange((u32)in_map_0, len_in*sizeof(DATA));
		Xil_DCacheFlushRange((u32)out_map_0, len_out*number_PE*sizeof(DATA));
		//Transfer AXI Stream
		//In map
		XAxiDma_SimpleTransfer(&axiDMA_data_0, (u32)in_map_0, len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&axiDMA_data_1, (u32)in_map_0, len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&axiDMA_data_2, (u32)in_map_0, len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&axiDMA_data_3, (u32)in_map_0, len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
		//Receive AXI Stream
		//Out map
		XAxiDma_SimpleTransfer(&axiDMA_data_0, (u32)out_map_0, len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&axiDMA_data_1, (u32)out_map_0 + len_out*sizeof(DATA), len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&axiDMA_data_2, (u32)out_map_0 + 2*len_out*sizeof(DATA), len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&axiDMA_data_3, (u32)out_map_0 + 3*len_out*sizeof(DATA), len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
	    //Invalidate cache
	    Xil_DCacheInvalidateRange((u32)in_map_0, len_in*sizeof(DATA));
	    Xil_DCacheInvalidateRange((u32)out_map_0, len_out*number_PE*sizeof(DATA));
	    //Wait Core
		while(!XMobilenet_stream_IsDone(&doMobilenet_stream_0));
		while(!XMobilenet_stream_IsDone(&doMobilenet_stream_1));
		while(!XMobilenet_stream_IsDone(&doMobilenet_stream_2));
		while(!XMobilenet_stream_IsDone(&doMobilenet_stream_3));
		/**/
	}
    //CONV layer 1x1
	else if (type_layer == 1){
		//Set memory address
		XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_0, tile_convs + inter_layer*3*MAX_CONVS + (layer-1)*3*MAX_CONVS*3 + 0*3*MAX_CONVS*3*18);
		XMobilenet_stream_Set_ext_info(&doMobilenet_stream_0, info_convs + inter_layer*size_info*MAX_CONVS + (layer-1)*size_info*MAX_CONVS*3 + 0*size_info*MAX_CONVS*3*18);
		XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_1, tile_convs + inter_layer*3*MAX_CONVS + (layer-1)*3*MAX_CONVS*3 + 1*3*MAX_CONVS*3*18);
		XMobilenet_stream_Set_ext_info(&doMobilenet_stream_1, info_convs + inter_layer*size_info*MAX_CONVS + (layer-1)*size_info*MAX_CONVS*3 + 1*size_info*MAX_CONVS*3*18);
		XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_2, tile_convs + inter_layer*3*MAX_CONVS + (layer-1)*3*MAX_CONVS*3 + 2*3*MAX_CONVS*3*18);
		XMobilenet_stream_Set_ext_info(&doMobilenet_stream_2, info_convs + inter_layer*size_info*MAX_CONVS + (layer-1)*size_info*MAX_CONVS*3 + 2*size_info*MAX_CONVS*3*18);
		XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_3, tile_convs + inter_layer*3*MAX_CONVS + (layer-1)*3*MAX_CONVS*3 + 3*3*MAX_CONVS*3*18);
		XMobilenet_stream_Set_ext_info(&doMobilenet_stream_3, info_convs + inter_layer*size_info*MAX_CONVS + (layer-1)*size_info*MAX_CONVS*3 + 3*size_info*MAX_CONVS*3*18);

		int len_res;
		if (len_res_write > len_res_read){
			len_res = len_res_write;
		}
		else{
			len_res = len_res_read;
		}

		/*Process*/
		XMobilenet_stream_Start(&doMobilenet_stream_0);
		XMobilenet_stream_Start(&doMobilenet_stream_1);
		XMobilenet_stream_Start(&doMobilenet_stream_2);
		XMobilenet_stream_Start(&doMobilenet_stream_3);
	    //Flush cache
	    Xil_DCacheFlushRange((u32)in_map_0, len_in*sizeof(DATA));
		Xil_DCacheFlushRange((u32)out_map_0, len_out*number_PE*sizeof(DATA));
		Xil_DCacheFlushRange((u32)res_map_write_0, len_res*number_PE*sizeof(DATA));
		//Transfer AXI Stream
		//In map
		XAxiDma_SimpleTransfer(&axiDMA_data_0, (u32)in_map_0, len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&axiDMA_data_1, (u32)in_map_0, len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&axiDMA_data_2, (u32)in_map_0, len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&axiDMA_data_3, (u32)in_map_0, len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
		//Residual
		if (CONV_res[layer][inter_layer][1] || CONV_res[layer][inter_layer][0]){   //read residual (store to send information of dma)
			XAxiDma_SimpleTransfer(&axiDMA_res_0, (u32)res_map_write_0, len_res_read*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&axiDMA_res_1, (u32)res_map_write_0 + len_res_read*sizeof(DATA), len_res_read*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&axiDMA_res_2, (u32)res_map_write_0 + 2*len_res_read*sizeof(DATA), len_res_read*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&axiDMA_res_3, (u32)res_map_write_0 + 3*len_res_read*sizeof(DATA), len_res_read*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
		}
		//Receive AXI Stream
		//Out map
		XAxiDma_SimpleTransfer(&axiDMA_data_0, (u32)out_map_0, len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&axiDMA_data_1, (u32)out_map_0 + len_out*sizeof(DATA), len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&axiDMA_data_2, (u32)out_map_0 + 2*len_out*sizeof(DATA), len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&axiDMA_data_3, (u32)out_map_0 + 3*len_out*sizeof(DATA), len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
		//Residual
		if (CONV_res[layer][inter_layer][0]){   //write residual (store map)
			XAxiDma_SimpleTransfer(&axiDMA_res_0, (u32)res_map_write_0, len_res_write*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
			XAxiDma_SimpleTransfer(&axiDMA_res_1, (u32)res_map_write_0 + len_res_write*sizeof(DATA), len_res_write*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
			XAxiDma_SimpleTransfer(&axiDMA_res_2, (u32)res_map_write_0 + 2*len_res_write*sizeof(DATA), len_res_write*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
			XAxiDma_SimpleTransfer(&axiDMA_res_3, (u32)res_map_write_0 + 3*len_res_write*sizeof(DATA), len_res_write*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
		}
	    //Invalidate cache
	    Xil_DCacheInvalidateRange((u32)in_map_0, len_in*sizeof(DATA));
	    Xil_DCacheInvalidateRange((u32)out_map_0, len_out*number_PE*sizeof(DATA));
        Xil_DCacheInvalidateRange((u32)res_map_write_0, len_res*number_PE*sizeof(DATA));
	    //Wait Core
		while(!XMobilenet_stream_IsDone(&doMobilenet_stream_0));
		while(!XMobilenet_stream_IsDone(&doMobilenet_stream_1));
		while(!XMobilenet_stream_IsDone(&doMobilenet_stream_2));
		while(!XMobilenet_stream_IsDone(&doMobilenet_stream_3));
		/**/
    }
    //depthwise layer
    else if (type_layer == 2){
    	//Set address from DRAM in PL
		XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_0, tile_convs + inter_layer*3*MAX_CONVS + (layer-1)*3*MAX_CONVS*3 + 0*3*MAX_CONVS*3*18);
		XMobilenet_stream_Set_ext_info(&doMobilenet_stream_0, info_convs + inter_layer*size_info*MAX_CONVS + (layer-1)*size_info*MAX_CONVS*3 + 0*size_info*MAX_CONVS*3*18);
		XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_1, tile_convs + inter_layer*3*MAX_CONVS + (layer-1)*3*MAX_CONVS*3 + 1*3*MAX_CONVS*3*18);
		XMobilenet_stream_Set_ext_info(&doMobilenet_stream_1, info_convs + inter_layer*size_info*MAX_CONVS + (layer-1)*size_info*MAX_CONVS*3 + 1*size_info*MAX_CONVS*3*18);
		XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_2, tile_convs + inter_layer*3*MAX_CONVS + (layer-1)*3*MAX_CONVS*3 + 2*3*MAX_CONVS*3*18);
		XMobilenet_stream_Set_ext_info(&doMobilenet_stream_2, info_convs + inter_layer*size_info*MAX_CONVS + (layer-1)*size_info*MAX_CONVS*3 + 2*size_info*MAX_CONVS*3*18);
		XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_3, tile_convs + inter_layer*3*MAX_CONVS + (layer-1)*3*MAX_CONVS*3 + 3*3*MAX_CONVS*3*18);
		XMobilenet_stream_Set_ext_info(&doMobilenet_stream_3, info_convs + inter_layer*size_info*MAX_CONVS + (layer-1)*size_info*MAX_CONVS*3 + 3*size_info*MAX_CONVS*3*18);

		/*Process*/
		XMobilenet_stream_Start(&doMobilenet_stream_0);
		XMobilenet_stream_Start(&doMobilenet_stream_1);
		XMobilenet_stream_Start(&doMobilenet_stream_2);
		XMobilenet_stream_Start(&doMobilenet_stream_3);
	    //Flush cache
	    Xil_DCacheFlushRange((u32)in_map_0, len_in*number_PE*sizeof(DATA));
		Xil_DCacheFlushRange((u32)out_map_0, len_out*number_PE*sizeof(DATA));
		//Transfer AXI Stream
		//In map
		XAxiDma_SimpleTransfer(&axiDMA_data_0, (u32)in_map_0, len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&axiDMA_data_1, (u32)in_map_0 + len_in*sizeof(DATA), len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&axiDMA_data_2, (u32)in_map_0 + 2*len_in*sizeof(DATA), len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&axiDMA_data_3, (u32)in_map_0 + 3*len_in*sizeof(DATA), len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
		//Receive AXI Stream
		//Out map
		XAxiDma_SimpleTransfer(&axiDMA_data_0, (u32)out_map_0, len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&axiDMA_data_1, (u32)out_map_0 + len_out*sizeof(DATA), len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&axiDMA_data_2, (u32)out_map_0 + 2*len_out*sizeof(DATA), len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&axiDMA_data_3, (u32)out_map_0 + 3*len_out*sizeof(DATA), len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
	    //Invalidate cache
	    Xil_DCacheInvalidateRange((u32)in_map_0, len_in*number_PE*sizeof(DATA));
	    Xil_DCacheInvalidateRange((u32)out_map_0, len_out*number_PE*sizeof(DATA));
	    //Wait Core
		while(!XMobilenet_stream_IsDone(&doMobilenet_stream_0));
		while(!XMobilenet_stream_IsDone(&doMobilenet_stream_1));
		while(!XMobilenet_stream_IsDone(&doMobilenet_stream_2));
		while(!XMobilenet_stream_IsDone(&doMobilenet_stream_3));
		/**/
    }
}

void InvertedResidual(DATA *res_map_write, DATA *res_map_read, W_CONV *w_conv, W_CONV *b_conv, int layer, int length_in_map){
    int inter_layer, type_layer;
    DATA len_in, len_out, len_res_read, len_res_write;
    //GROUP 1
    inter_layer = 0;
    type_layer = 1;
    len_in = DRAM_2_STREAM_1x1(axi_stream_out, axi_stream_in,
    		CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][1], length_in_map, type_layer, 0);
    len_out = (CONV_size[layer][inter_layer][0]*CONV_size[layer][inter_layer][6]*CONV_size[layer][inter_layer][6])/number_PE;
    CONV_BATCH_RELU(axi_stream_in, axi_stream_out, axi_stream_res_write, axi_stream_res_read,
    				w_conv, b_conv, layer, inter_layer, type_layer, len_in, len_out, len_res_write, len_res_read);
    STREAM_2_DRAM_1x1(axi_stream_out, cpu_map, CONV_size[layer][inter_layer][0], length_in_map);
    //GROUP 2
	inter_layer = 1;
	type_layer = 2;
	len_in = DRAM_2_STREAM_3x3(cpu_map, axi_stream_in,
			CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][0], length_in_map, type_layer);
	len_out = (CONV_size[layer][inter_layer][0]*(CONV_size[layer][inter_layer][6]/CONV_size[layer][inter_layer][3])*(CONV_size[layer][inter_layer][6]/CONV_size[layer][inter_layer][3]))/number_PE;
    CONV_BATCH_RELU(axi_stream_in, axi_stream_out, axi_stream_res_write, axi_stream_res_read,
    				w_conv, b_conv, layer, inter_layer, type_layer, len_in, len_out, len_res_write, len_res_read);
    if (CONV_size[layer][1][3] == 1){
    	//GROUP 3
    	inter_layer = 2;
    	type_layer = 1;
    	len_in = DRAM_2_STREAM_1x1(axi_stream_out, axi_stream_in,
    			CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][1], CONV_size[layer][inter_layer][6], type_layer, 0);
    }
    else{
    	STREAM_2_DRAM_3x3(axi_stream_out, cpu_map, CONV_size[layer][inter_layer][0], length_in_map, CONV_size[layer][inter_layer][3]);
    	//GROUP 3
    	inter_layer = 2;
    	type_layer = 1;
    	len_in = DRAM_2_STREAM_1x1(cpu_map, axi_stream_in,
    			CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][1], CONV_size[layer][inter_layer][6], type_layer, 1);
    }
	len_out = (CONV_size[layer][inter_layer][0]*CONV_size[layer][inter_layer][6]*CONV_size[layer][inter_layer][6])/number_PE;
	len_res_read = DRAM_2_STREAM_res(CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][6], CONV_res[layer][inter_layer][1], CONV_res[layer][inter_layer][0]);
	len_res_write = (CONV_size[layer][inter_layer][0]*CONV_size[layer][inter_layer][6]*CONV_size[layer][inter_layer][6])/number_PE;
    CONV_BATCH_RELU(axi_stream_in, axi_stream_out, axi_stream_res_write, axi_stream_res_read,
    				w_conv, b_conv, layer, inter_layer, type_layer, len_in, len_out, len_res_write, len_res_read);
}

void AVG(DATA *in_map_0, DATA *out_array_0,
		DATA len_in, DATA len_out){
    //LOAD DATA in PL
    XMobilenet_stream_Set_layer(&doMobilenet_stream_0, 0);
    XMobilenet_stream_Set_type_layer(&doMobilenet_stream_0, 3);
    XMobilenet_stream_Set_layer(&doMobilenet_stream_1, 0);
    XMobilenet_stream_Set_type_layer(&doMobilenet_stream_1, 3);
    XMobilenet_stream_Set_layer(&doMobilenet_stream_2, 0);
    XMobilenet_stream_Set_type_layer(&doMobilenet_stream_2, 3);
    XMobilenet_stream_Set_layer(&doMobilenet_stream_3, 0);
    XMobilenet_stream_Set_type_layer(&doMobilenet_stream_3, 3);
    //Set address from DRAM in PL
	XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_0, tile_avg + 0*3*MAX_AVG);
	XMobilenet_stream_Set_ext_info(&doMobilenet_stream_0, info_avg + 0*size_info*MAX_AVG);
	XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_1, tile_avg + 1*3*MAX_AVG);
	XMobilenet_stream_Set_ext_info(&doMobilenet_stream_1, info_avg + 1*size_info*MAX_AVG);
	XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_2, tile_avg + 2*3*MAX_AVG);
	XMobilenet_stream_Set_ext_info(&doMobilenet_stream_2, info_avg + 2*size_info*MAX_AVG);
	XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_3, tile_avg + 3*3*MAX_AVG);
	XMobilenet_stream_Set_ext_info(&doMobilenet_stream_3, info_avg + 3*size_info*MAX_AVG);

	/*Process*/
	XMobilenet_stream_Start(&doMobilenet_stream_0);
	XMobilenet_stream_Start(&doMobilenet_stream_1);
	XMobilenet_stream_Start(&doMobilenet_stream_2);
	XMobilenet_stream_Start(&doMobilenet_stream_3);
    //Flush cache
    Xil_DCacheFlushRange((u32)in_map_0, len_in*number_PE*sizeof(DATA));
	Xil_DCacheFlushRange((u32)out_array_0, len_out*number_PE*sizeof(DATA));
	//Transfer AXI Stream
	//In map
	XAxiDma_SimpleTransfer(&axiDMA_data_0, (u32)in_map_0, len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&axiDMA_data_1, (u32)in_map_0 + len_in*sizeof(DATA), len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&axiDMA_data_2, (u32)in_map_0 + 2*len_in*sizeof(DATA), len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&axiDMA_data_3, (u32)in_map_0 + 3*len_in*sizeof(DATA), len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
	//Receive AXI Stream
	//Out array
	XAxiDma_SimpleTransfer(&axiDMA_data_0, (u32)out_array_0, len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&axiDMA_data_1, (u32)out_array_0 + len_out*sizeof(DATA), len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&axiDMA_data_2, (u32)out_array_0 + 2*len_out*sizeof(DATA), len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&axiDMA_data_3, (u32)out_array_0 + 3*len_out*sizeof(DATA), len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
    //Invalidate cache
    Xil_DCacheInvalidateRange((u32)in_map_0, len_in*number_PE*sizeof(DATA));
    Xil_DCacheInvalidateRange((u32)out_array_0, len_out*number_PE*sizeof(DATA));
    //Wait Core
	while(!XMobilenet_stream_IsDone(&doMobilenet_stream_0));
	while(!XMobilenet_stream_IsDone(&doMobilenet_stream_1));
	while(!XMobilenet_stream_IsDone(&doMobilenet_stream_2));
	while(!XMobilenet_stream_IsDone(&doMobilenet_stream_3));
	/**/
}

void Fully_Connected_layer(DATA *in_array_0, DATA *out_array_0,
		W_FC *w_fc, W_FC *b_fc, DATA len_in, DATA len_out){
    //LOAD DATA in PL
    XMobilenet_stream_Set_layer(&doMobilenet_stream_0, 0);
    XMobilenet_stream_Set_type_layer(&doMobilenet_stream_0, 4);
    XMobilenet_stream_Set_layer(&doMobilenet_stream_1, 0);
    XMobilenet_stream_Set_type_layer(&doMobilenet_stream_1, 4);
    XMobilenet_stream_Set_layer(&doMobilenet_stream_2, 0);
    XMobilenet_stream_Set_type_layer(&doMobilenet_stream_2, 4);
    XMobilenet_stream_Set_layer(&doMobilenet_stream_3, 0);
    XMobilenet_stream_Set_type_layer(&doMobilenet_stream_3, 4);
    //Set address from DRAM in PL
	XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_0, tile_fc + 0*3*MAX_FC);
	XMobilenet_stream_Set_ext_info(&doMobilenet_stream_0, info_fc + 0*size_info*MAX_FC);
	XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_1, tile_fc + 1*3*MAX_FC);
	XMobilenet_stream_Set_ext_info(&doMobilenet_stream_1, info_fc + 1*size_info*MAX_FC);
	XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_2, tile_fc + 2*3*MAX_FC);
	XMobilenet_stream_Set_ext_info(&doMobilenet_stream_2, info_fc + 2*size_info*MAX_FC);
	XMobilenet_stream_Set_ext_tile(&doMobilenet_stream_3, tile_fc + 3*3*MAX_FC);
	XMobilenet_stream_Set_ext_info(&doMobilenet_stream_3, info_fc + 3*size_info*MAX_FC);

	/*Process*/
	XMobilenet_stream_Start(&doMobilenet_stream_0);
	XMobilenet_stream_Start(&doMobilenet_stream_1);
	XMobilenet_stream_Start(&doMobilenet_stream_2);
	XMobilenet_stream_Start(&doMobilenet_stream_3);
    //Flush cache
    Xil_DCacheFlushRange((u32)in_array_0, len_in*sizeof(DATA));
	Xil_DCacheFlushRange((u32)out_array_0, len_out*number_PE*sizeof(DATA));
	//Transfer AXI Stream
	//In array
	XAxiDma_SimpleTransfer(&axiDMA_data_0, (u32)in_array_0, len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&axiDMA_data_1, (u32)in_array_0, len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&axiDMA_data_2, (u32)in_array_0, len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&axiDMA_data_3, (u32)in_array_0, len_in*sizeof(DATA), XAXIDMA_DMA_TO_DEVICE);
	//Receive AXI Stream
	//Out array
	XAxiDma_SimpleTransfer(&axiDMA_data_0, (u32)out_array_0, len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&axiDMA_data_1, (u32)out_array_0 + len_out*sizeof(DATA), len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&axiDMA_data_2, (u32)out_array_0 + len_out*sizeof(DATA)*2, len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&axiDMA_data_3, (u32)out_array_0 + len_out*sizeof(DATA)*3, len_out*sizeof(DATA), XAXIDMA_DEVICE_TO_DMA);
	//while(XAxiDma_Busy(&axiDMA_data, XAXIDMA_DEVICE_TO_DMA));
    //Invalidate cache
    Xil_DCacheInvalidateRange((u32)in_array_0, len_in*sizeof(DATA));
    Xil_DCacheInvalidateRange((u32)out_array_0, len_out*number_PE*sizeof(DATA));
    //Wait Core
	while(!XMobilenet_stream_IsDone(&doMobilenet_stream_0));
	while(!XMobilenet_stream_IsDone(&doMobilenet_stream_1));
	while(!XMobilenet_stream_IsDone(&doMobilenet_stream_2));
	while(!XMobilenet_stream_IsDone(&doMobilenet_stream_3));
	/**/
}

void Softmax_layer(ACT_FC *in_array, float *out_array, int length){
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

int find_max_int(ACT_FC *in_array, int length){
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

/*
 * MODEL
 */

void model(DATA first_len_in){
    int layer, inter_layer, length, type_layer;
    DATA len_in = first_len_in, len_out, len_res_write, len_res_read;

    ///CONVBNReLU 0
    XUartPs_SendByte(UART_BASEADDR, 'a');
    layer = 0;
    length = 224;
    inter_layer = 0;
    type_layer = 0;
    len_out = (CONV_size[layer][inter_layer][0]*(CONV_size[layer][inter_layer][6]/CONV_size[layer][inter_layer][3])*(CONV_size[layer][inter_layer][6]/CONV_size[layer][inter_layer][3]))/number_PE;
    CONV_BATCH_RELU(axi_stream_in, axi_stream_out, axi_stream_res_write, axi_stream_res_read,
    				weights_CONV, bias_CONV, 0, 0, 0, len_in, len_out, len_res_write, len_res_read);
    STREAM_2_DRAM_3x3(axi_stream_out, cpu_map, CONV_size[layer][inter_layer][0], length, CONV_size[layer][inter_layer][3]);

    ///InvertedResidual 1
    XUartPs_SendByte(UART_BASEADDR, 'b');
    layer = 1;
    length = 112;
    //Group 1
    inter_layer = 0;
    type_layer = 2;
    len_in = DRAM_2_STREAM_3x3(cpu_map, axi_stream_in,
    		CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][0], length, type_layer);
    len_out = (CONV_size[layer][inter_layer][0]*(CONV_size[layer][inter_layer][6]/CONV_size[layer][inter_layer][3])*(CONV_size[layer][inter_layer][6]/CONV_size[layer][inter_layer][3]))/number_PE;
    CONV_BATCH_RELU(axi_stream_in, axi_stream_out, axi_stream_res_write, axi_stream_res_read,
    				weights_CONV, bias_CONV, 1, 0, 2, len_in, len_out, len_res_write, len_res_read);
    //Group 2
    inter_layer = 1;
    type_layer = 1;
    len_in = DRAM_2_STREAM_1x1(axi_stream_out, axi_stream_in,
    		CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][1], length, type_layer, 0);
    len_out = (CONV_size[layer][inter_layer][0]*CONV_size[layer][inter_layer][6]*CONV_size[layer][inter_layer][6])/number_PE;
    CONV_BATCH_RELU(axi_stream_in, axi_stream_out, axi_stream_res_write, axi_stream_res_read,
    				weights_CONV, bias_CONV, 1, 1, 1, len_in, len_out, len_res_write, len_res_read);

    ///InvertedResidual 2
    XUartPs_SendByte(UART_BASEADDR, 'c');
    layer = 2;
    length = 112;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 2, 112);

    ///InvertedResidual 3
    XUartPs_SendByte(UART_BASEADDR, 'd');
    layer = 3;
    length = 56;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 3, 56);

    ///InvertedResidual 4
    XUartPs_SendByte(UART_BASEADDR, 'e');
    layer = 4;
    length = 56;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 4, 56);

    ///InvertedResidual 5
    XUartPs_SendByte(UART_BASEADDR, 'f');
    layer = 5;
    length = 28;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 5, 28);

    ///InvertedResidual 6
    XUartPs_SendByte(UART_BASEADDR, 'g');
    layer = 6;
    length = 28;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 6, 28);

    ///InvertedResidual 7
    XUartPs_SendByte(UART_BASEADDR, 'h');
    layer = 7;
    length = 28;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 7, 28);

    ///InvertedResidual 8
    XUartPs_SendByte(UART_BASEADDR, 'i');
    layer = 8;
    length = 14;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 8, 14);

    ///InvertedResidual 9
    XUartPs_SendByte(UART_BASEADDR, 'j');
    layer = 9;
    length = 14;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 9, 14);

    ///InvertedResidual 10
    XUartPs_SendByte(UART_BASEADDR, 'k');
    layer = 10;
    length = 14;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 10, 14);

    ///InvertedResidual 11
    XUartPs_SendByte(UART_BASEADDR, 'l');
    layer = 11;
    length = 14;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 11, 14);

    ///InvertedResidual 12
    XUartPs_SendByte(UART_BASEADDR, 'm');
    layer = 12;
    length = 14;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 12, 14);

    ///InvertedResidual 13
    XUartPs_SendByte(UART_BASEADDR, 'n');
    layer = 13;
    length = 14;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 13, 14);

    ///InvertedResidual 14
    XUartPs_SendByte(UART_BASEADDR, 'o');
    layer = 14;
    length = 14;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 14, 14);

    ///InvertedResidual 15
    XUartPs_SendByte(UART_BASEADDR, 'p');
    layer = 15;
    length = 7;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 15, 7);

    ///InvertedResidual 16
    XUartPs_SendByte(UART_BASEADDR, 'q');
    layer = 16;
    length = 7;
    InvertedResidual(res_map_A, res_map_B, weights_CONV, bias_CONV, 16, 7);

    ///InvertedResidual 17
    XUartPs_SendByte(UART_BASEADDR, 'r');
    layer = 17;
    length = 7;
    InvertedResidual(res_map_B, res_map_A, weights_CONV, bias_CONV, 17, 7);

    ///CONVBNReLU 18
    XUartPs_SendByte(UART_BASEADDR, 's');
    layer = 18;
    length = 7;
    inter_layer = 0;
    type_layer = 1;
    len_in = DRAM_2_STREAM_1x1(axi_stream_out, axi_stream_in,
    		CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][1], length, type_layer, 0);
    len_out = (CONV_size[layer][inter_layer][0]*CONV_size[layer][inter_layer][6]*CONV_size[layer][inter_layer][6])/number_PE;
    CONV_BATCH_RELU(axi_stream_in, axi_stream_out, axi_stream_res_write, axi_stream_res_read,
    				weights_CONV, bias_CONV, 18, 0, 1, len_in, len_out, len_res_write, len_res_read);

    ///Flatten 19
    XUartPs_SendByte(UART_BASEADDR, 't');
    len_in = DRAM_2_STREAM_1x1(axi_stream_out, axi_stream_in,
    		1, CONV_size[18][0][0], 7, 3, 0);
    len_out = 1280/number_PE;
    AVG(axi_stream_in, axi_stream_out,
    	len_in, len_out);

    ///FC 20
    XUartPs_SendByte(UART_BASEADDR, 'u');
    len_in = DRAM_2_STREAM_array(axi_stream_out, axi_stream_in,
    		FC_size[0], FC_size[1]);
    len_out = 1000/number_PE;
    Fully_Connected_layer(axi_stream_in, axi_stream_out,
						  weights_FC, bias_FC, len_in, len_out);

    ///Softmax
    XUartPs_SendByte(UART_BASEADDR, 'v');
    Softmax_layer(axi_stream_out, array_softmax, 1000);
}

/*
 * MAIN
 */

int main(){
    char *weights_path, *bias_path;
    XTime t_ini, t_fin;
    float secs;

    //Paths
	weights_path = "0:weights.dat";
	bias_path = "0:bias.dat";

	//PL init
	init_PL();

    //Memory init
    init_memory();

    //read and normalize image
    image_read(cpu_map);
    DATA first_len_in = DRAM_2_STREAM_3x3(cpu_map, axi_stream_in, CONV_size[0][0][0], CONV_size[0][0][1], 224, 0);

    //read weights MobileNet
    weights_read(weights_path, weights_CONV, weights_FC);
    bias_read(bias_path, bias_CONV, bias_FC);
    XUartPs_SendByte(UART_BASEADDR, 'z');

	//set info and tile
	set_tile_info(tile_3x3, tile_convs, tile_avg, tile_fc,
				  info_3x3, info_convs, info_avg, info_fc);

    //Set address from DRAM in PL
    XMobilenet_stream_Set_ext_w_conv(&doMobilenet_stream_0, weights_CONV);
    XMobilenet_stream_Set_ext_b_conv(&doMobilenet_stream_0, bias_CONV);
    XMobilenet_stream_Set_ext_w_fc(&doMobilenet_stream_0, weights_FC);
    XMobilenet_stream_Set_ext_b_fc(&doMobilenet_stream_0, bias_FC);
    XMobilenet_stream_Set_ext_w_conv(&doMobilenet_stream_1, weights_CONV);
    XMobilenet_stream_Set_ext_b_conv(&doMobilenet_stream_1, bias_CONV);
    XMobilenet_stream_Set_ext_w_fc(&doMobilenet_stream_1, weights_FC);
    XMobilenet_stream_Set_ext_b_fc(&doMobilenet_stream_1, bias_FC);
    XMobilenet_stream_Set_ext_w_conv(&doMobilenet_stream_2, weights_CONV);
    XMobilenet_stream_Set_ext_b_conv(&doMobilenet_stream_2, bias_CONV);
    XMobilenet_stream_Set_ext_w_fc(&doMobilenet_stream_2, weights_FC);
    XMobilenet_stream_Set_ext_b_fc(&doMobilenet_stream_2, bias_FC);
    XMobilenet_stream_Set_ext_w_conv(&doMobilenet_stream_3, weights_CONV);
    XMobilenet_stream_Set_ext_b_conv(&doMobilenet_stream_3, bias_CONV);
    XMobilenet_stream_Set_ext_w_fc(&doMobilenet_stream_3, weights_FC);
    XMobilenet_stream_Set_ext_b_fc(&doMobilenet_stream_3, bias_FC);

    //Flush cache
    Xil_DCacheFlushRange((u32)weights_CONV, w_conv_layer*sizeof(W_CONV));
    Xil_DCacheFlushRange((u32)bias_CONV, b_conv_layer*sizeof(W_CONV));
    Xil_DCacheFlushRange((u32)weights_FC, fc_layer*sizeof(W_FC));
    Xil_DCacheFlushRange((u32)bias_FC, b_fc_layer*sizeof(W_FC));
    Xil_DCacheFlushRange((u32)tile_3x3, number_PE*3*MAX_CONV_3X3*sizeof(DATA));
    Xil_DCacheFlushRange((u32)tile_convs, number_PE*18*3*3*MAX_CONVS*sizeof(DATA));
    Xil_DCacheFlushRange((u32)tile_avg, number_PE*3*MAX_AVG*sizeof(DATA));
    Xil_DCacheFlushRange((u32)tile_fc, number_PE*3*MAX_FC*sizeof(DATA));
    Xil_DCacheFlushRange((u32)info_3x3, number_PE*size_info*MAX_CONV_3X3*sizeof(DATA));
    Xil_DCacheFlushRange((u32)info_convs, number_PE*18*3*size_info*MAX_CONVS*sizeof(DATA));
    Xil_DCacheFlushRange((u32)info_avg, number_PE*size_info*MAX_AVG*sizeof(DATA));
    Xil_DCacheFlushRange((u32)info_fc, number_PE*size_info*MAX_FC*sizeof(DATA));
    Xil_DCacheFlushRange((u32)conv_quant, 3*4*19*sizeof(DATA));
    Xil_DCacheFlushRange((u32)avg_quant, 4*sizeof(DATA));
    Xil_DCacheFlushRange((u32)fc_quant, 4*sizeof(DATA));

    //MODEL AND OTHERS
    XTime_GetTime(&t_ini);
    model(first_len_in);
    XTime_GetTime(&t_fin);
    XUartPs_SendByte(UART_BASEADDR, 'z');

    //Invalidate cache
    Xil_DCacheInvalidateRange((u32)weights_CONV, w_conv_layer*sizeof(W_CONV));
    Xil_DCacheInvalidateRange((u32)bias_CONV, b_conv_layer*sizeof(W_CONV));
    Xil_DCacheInvalidateRange((u32)weights_FC, fc_layer*sizeof(W_FC));
    Xil_DCacheInvalidateRange((u32)bias_FC, b_fc_layer*sizeof(W_FC));
    Xil_DCacheInvalidateRange((u32)tile_3x3, number_PE*3*MAX_CONV_3X3*sizeof(DATA));
    Xil_DCacheInvalidateRange((u32)tile_convs, number_PE*18*3*3*MAX_CONVS*sizeof(DATA));
    Xil_DCacheInvalidateRange((u32)tile_avg, number_PE*3*MAX_AVG*sizeof(DATA));
    Xil_DCacheInvalidateRange((u32)tile_fc, number_PE*3*MAX_FC*sizeof(DATA));
    Xil_DCacheInvalidateRange((u32)info_3x3, number_PE*size_info*MAX_CONV_3X3*sizeof(DATA));
    Xil_DCacheInvalidateRange((u32)info_convs, number_PE*18*3*size_info*MAX_CONVS*sizeof(DATA));
    Xil_DCacheInvalidateRange((u32)info_avg, number_PE*size_info*MAX_AVG*sizeof(DATA));
    Xil_DCacheInvalidateRange((u32)info_fc, number_PE*size_info*MAX_FC*sizeof(DATA));
    Xil_DCacheInvalidateRange((u32)conv_quant, 3*4*19*sizeof(DATA));
    Xil_DCacheInvalidateRange((u32)avg_quant, 4*sizeof(DATA));
    Xil_DCacheInvalidateRange((u32)fc_quant, 4*sizeof(DATA));

    //Measuring time
    secs = (float)(t_fin - t_ini) / COUNTS_PER_SECOND;
    //Send Time
    float2char(secs, uart_send);
    XUartPs_SendByte(UART_BASEADDR, 'z');

    //Send results
    int limits[3] = {1000, 0, 0};   //section of map to send by uart
    int max_values[3] = {1000, 0, 0};   //max values of map
    send_float_data(array_softmax, uart_send, limits, max_values);   //in 0, array is array_softmax (dont care array_FC_1)

    //Intermedium results

    //image input
    /*int limits[3] = {224, 224, 3};   //section of map to send by uart
    int max_values[3] = {224, 224, 3};   //max values of map
    send_int_data(cpu_fea_2, uart_send, limits, max_values, 3);*/

    //weights
    /*int limits[3] = {448, 0, 0};   //section of map to send by uart
    int max_values[3] = {0, 0, 0};   //max values of map
    send_int_data(weights_CONV, uart_send, limits, max_values, 1);*/

    //feature map
    /*int layer = 0;
    int inter_layer = 0;
    int length = CONV_size[layer][inter_layer][6]/CONV_size[layer][inter_layer][3];
    int limits[3] = {length, length, CONV_size[layer][inter_layer][0]};   //section of map to send by uart
    int max_values[3] = {length, length, CONV_size[layer][inter_layer][0]};   //max values of map
    //order2CPU_map(res_map_2, res_map_1, CONV_size[layer][inter_layer][0], CONV_size[layer][inter_layer][6], CONV_size[layer][inter_layer][3]);
    send_int_data(cpu_map, uart_send, limits, max_values, 3);*/

    //arrays
    /*int limits[3] = {1280, 0, 0};   //section of map to send by uart
    int max_values[3] = {0, 0, 0};   //max values of map
    send_int_data(cpu_array, uart_send, limits, max_values, 1);*/

    //Free memory
    free_memory();
}
