#include "type_data.h"
#include "xuartps.h"
#include "ff.h"
#include "parameters.h"

void image_read(ACT_CONV *image);
void weights_read(char *filename_weights, W_CONV *w_conv, W_FC *w_fc);
void bias_read(char *filename_bias, W_CONV *b_conv, W_FC *b_fc);
int char2int(int32_t *vector, u8 data, int stop, int *index, int *change_number, int *neg_value);
void send_int_data(int32_t *vector, u8 *uart, int limits[], int max_values[], int len_limits);
void send_float_data(float *vector, u8 *uart, int limits[], int max_values[]);
void int2char(int32_t data, uint8_t *v_send);
void float2char(float data, uint8_t *v_send);

/*
 * Receive DATA
 */

void image_read(ACT_CONV *image){   //pixel images are sending in 3 bytes
    int8_t data_receive;
    ACT_CONV add = 0;
    int counter_sub_pixel = 0, counter_pixel = 0;

    while(1){
    	if(XUartPs_IsReceiveData(UART_BASEADDR)){
    		data_receive = ((int8_t) XUartPs_RecvByte(UART_BASEADDR));
    		add += (ACT_CONV) data_receive;
    		counter_sub_pixel++;
    		if (counter_sub_pixel >= 3){
    			image[counter_pixel] = add;
    			counter_sub_pixel = 0;
    			add = 0;
    			counter_pixel++;
    		}
    	}
    	if (counter_pixel >= IMAGE_SIZE*IMAGE_SIZE*3){
    		break;
    	}
    }
}

void weights_read(char *filename_weights, W_CONV *w_conv, W_FC *w_fc){
	FIL file;
	FATFS def_drive;
	UINT bytes_read;
	UINT BytesToRead = sizeof(u8);
	u8 data;
    W_CONV temp_W_CONV = 0;
    W_FC temp_W_FC = 0;
	int index = 0, sign = 1;


    f_mount(&def_drive, "0:", 0);
    f_open(&file, filename_weights, FA_OPEN_EXISTING | FA_READ);

    //CONVs (weights)
    XUartPs_SendByte(UART_BASEADDR, 'a');
    while(index <= w_conv_layer - 1){
    	f_read(&file, &data, BytesToRead, &bytes_read);
        if (data != 32){
            if (data ==  '-'){
                sign = -1;
            }
            else{
                temp_W_CONV = temp_W_CONV*10 + sign*(data - 48);
            }
        }
        else{
            w_conv[index] = temp_W_CONV;
            temp_W_CONV = 0;
            sign = 1;
            index++;
        }
    }
    //FC (weights)
    index = 0, sign = 1;
    XUartPs_SendByte(UART_BASEADDR, 'b');
    while(index <= fc_layer - 1){
    	f_read(&file, &data, BytesToRead, &bytes_read);
        if (data != 32){
            if (data ==  '-'){
                sign = -1;
            }
            else{
                temp_W_FC = temp_W_FC*10 + sign*(data - 48);
            }
        }
        else{
            w_fc[index] = temp_W_FC;
            temp_W_FC = 0;
            sign = 1;
            index++;
        }
    }

    f_close(&file);
}

void bias_read(char *filename_bias, W_CONV *b_conv, W_FC *b_fc){
	FIL file;
	FATFS def_drive;
	UINT bytes_read;
	UINT BytesToRead = sizeof(u8);
	u8 data;
    W_CONV temp_W_CONV = 0;
    W_FC temp_W_FC = 0;
	int index = 0, sign = 1;


    f_mount(&def_drive, "0:", 0);
    f_open(&file, filename_bias, FA_OPEN_EXISTING | FA_READ);

    //CONVs (bias)
    index = 0, temp_W_CONV = 0, sign = 1;
    while(index <= b_conv_layer - 1){
		f_read(&file, &data, BytesToRead, &bytes_read);
        if (data != 32){
            if (data ==  '-'){
                sign = -1;
            }
            else{
                temp_W_CONV = temp_W_CONV*10 + sign*(data - 48);
            }
        }
        else{
            b_conv[index] = temp_W_CONV;
            temp_W_CONV = 0;
            sign = 1;
            index++;
        }
    }
    //FC (bias)
    index = 0, temp_W_FC = 0, sign = 1;
    while(index <= b_fc_layer - 1){
    	f_read(&file, &data, BytesToRead, &bytes_read);
        if (data != 32){
            if (data ==  '-'){
                sign = -1;
            }
            else{
                temp_W_FC = temp_W_FC*10 + sign*(data - 48);
            }
        }
        else{
            b_fc[index] = temp_W_FC;
            temp_W_FC = 0;
            sign = 1;
            index++;
        }
    }

    f_close(&file);
}


int char2int(int32_t *vector, u8 data, int stop, int *index, int *change_number, int *neg_value){
	if(data == 'f'){   //NULL
		return 1;
	}
	else if(data != ' ' && data != '-'){
		if (*change_number){
			vector[*index] = (data - 48) * (*neg_value);   //char to int
			*change_number = 0;
		}
		else{
			vector[*index] = vector[*index]*10 + (data - 48) * (*neg_value);   //add a new digit to the number
		}
	}
	else if(data == '-'){
		*neg_value = -1;
	}
	else{
		*neg_value = 1;
		*index = *index + 1;
		*change_number = 1;
	}
	if(stop == *index){
		return 1;
	}
	else{
		return 0;
	}
}

/*
 * SEND DATA
 */

void send_int_data(int32_t *vector, u8 *uart, int limits[], int max_values[], int len_limits){
	if(len_limits == 0){
		for (int i = 0; i < limits[0]; i++){
			float2char(vector[i]*100, uart);
		}
	}
	else if(len_limits == 1){
		for (int i = 0; i < limits[0]; i++){
			int2char(vector[i], uart);
		}
	}
	else if(len_limits == 2){
		for (int i = 0; i < limits[0]; i++){
			for (int j = 0; j < limits[1]; j++){
				int2char(vector[j + i*max_values[1]], uart);
			}
		}
	}
	else if(len_limits == 3){
		for (int k = 0; k < limits[2]; k++){
			for (int i = 0; i < limits[0]; i++){
				for (int j = 0; j < limits[1]; j++){
					int2char(vector[j + i*max_values[0] + k*max_values[0]*max_values[1]], uart);
				}
			}
		}
	}
	else{
		for (int i = 0; i < limits[0]; i++){
			for (int j = 0; j < limits[1]; j++){
				int2char(vector[j + i*max_values[1] + limits[2]*max_values[1]*max_values[0]], uart);
			}
		}
	}
	XUartPs_SendByte(UART_BASEADDR, ' ');
	XUartPs_SendByte(UART_BASEADDR, 'z');
}

void send_float_data(float *vector, u8 *uart, int limits[], int max_values[]){
	for (int i = 0; i < limits[0]; i++){
		float2char(vector[i]*100, uart);
	}

	XUartPs_SendByte(UART_BASEADDR, ' ');
	XUartPs_SendByte(UART_BASEADDR, 'z');
}

void int2char(int32_t data, uint8_t *v_send){
	int temp, index = 31;

	if(data > 0){
		temp = data;
	}
	else{
		temp = -data;
	}

	while(temp != 0){
		v_send[index] = temp%10 + 48;
		temp /= 10;
		index--;
	}

	if(data < 0){
		v_send[index] = '-';
		index--;
	}
	v_send[index] = ' ';

	for (int i = index; i < 32; i++){
		XUartPs_SendByte(UART_BASEADDR, v_send[i]);
	}
}

void float2char(float data, uint8_t *v_send){
    int index_int = 16;
    float temp_data;

    //Preprocessing

	if(data > 0){
		temp_data = data;
	}
	else{
		temp_data = -data;
	}

    int int_part = (int) temp_data;
    float fract_part = temp_data - int_part;

    //inttegral part

    int temp_int = int_part;
	while(temp_int != 0){
		v_send[index_int] = temp_int%10 + 48;
		temp_int /= 10;
		index_int--;
	}
	if(data < 0){
		v_send[index_int] = '-';
		index_int--;
	}
	v_send[index_int] = ' ';

    //fractional part

    v_send[17] = '.';
    int index_frac = 18, counter = 0;
    float temp_frac = fract_part;
    while(temp_frac != 0){
        v_send[index_frac] = (int)(temp_frac*10) + 48;
        temp_frac = temp_frac*10 - (int)(temp_frac*10);
        index_frac++;
        counter++;
        if (counter >= 4){
            if (temp_frac*10 >= 5){
               v_send[index_frac-1]++;
            }
            break;
        }
    }
    if (counter == 0){
        v_send[index_frac] = '0';
        index_frac++;
        v_send[index_frac] = '0';
        index_frac++;
    }


    for (int i = index_int; i <= index_frac-1; i++){
    	XUartPs_SendByte(UART_BASEADDR, v_send[i]);
    }
}
