#ifndef READ_WRITE_FUNCTIONS_H
#define READ_WRITE_FUNCTIONS_H

extern void image_read(ACT_CONV *image);
extern void weights_read(char *filename_weights, W_CONV *w_conv, W_FC *w_fc);
extern void bias_read(char *filename_bias, W_CONV *b_conv, W_FC *b_fc);
extern int char2int(int32_t *vector, u8 data, int stop, int *index, int *change_number, int *neg_value);
extern void send_int_data(int32_t *vector, u8 *uart, int limits[], int max_values[], int len_limits);
extern void send_float_data(float *vector, u8 *uart, int limits[], int max_values[]);
extern void int2char(int32_t data, uint8_t *v_send);
extern void float2char(float data, uint8_t *v_send);

#endif
