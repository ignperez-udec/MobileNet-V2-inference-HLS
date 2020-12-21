import functions
import numpy as np

CONV_size = [
        [3, 32],
        [[32, 32], [32, 16]],
        [[16, 96], [96, 96], [96, 24]],
        [[24, 144], [144, 144], [144, 24]],
        [[24, 144], [144, 144], [144, 32]],
        [[32, 192], [192, 192], [192, 32]],
        [[32, 192], [192, 192], [192, 32]],
        [[32, 192], [192, 192], [192, 64]],
        [[64, 384], [384, 384], [384, 64]],
        [[64, 384], [384, 384], [384, 64]],
        [[64, 384], [384, 384], [384, 64]],
        [[64, 384], [384, 384], [384, 96]],
        [[96, 576], [576, 576], [576, 96]],
        [[96, 576], [576, 576], [576, 96]],
        [[96, 576], [576, 576], [576, 160]],
        [[160, 960], [960, 960], [960, 160]],
        [[160, 960], [960, 960], [960, 160]],
        [[160, 960], [960, 960], [960, 320]],
        [320, 1280]
        ]

FC_size = [1280, 1000]

stride_CONV = [
        2,
        [1, 1],
        [1, 2, 1],
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        1]

#Quantification
#Activations (fractional bits)
CONV_quan_act = [
        7,
        [8, 7, 7],   #LAST ONLY FOR INVERTED
        [7, 8, 8],
        [8, 11, 9],
        [8, 10, 9],
        [9, 11, 10],
        [9, 12, 10],
        [8, 10, 9],
        [9, 12, 10],
        [9, 12, 11],
        [9, 12, 9],
        [8, 11, 9],
        [9, 11, 10],
        [8, 9, 9],
        [7, 10, 7],
        [8, 10, 8],
        [6, 10, 7],
        [5, 12 ,11],
        10]
FC_quant_act = [4, 3]   #AVG, FC
fixed_act = [12, 8, 10]   #CONV, AVG, FC

#Weights (fractional bits)
CONV_quan_w = [
        11,
        [7, 9, 9],   #LAST ONLY FOR INVERTED
        [11, 8, 10],
        [10, 8, 9],
        [12, 8, 10],
        [12, 8, 11],
        [12, 8, 11],
        [12, 9, 10],
        [12, 8, 10],
        [12, 7, 11],
        [12, 8, 11],
        [12, 8, 11],
        [11, 7, 11],
        [12, 7, 10],
        [12, 9, 10],
        [12, 7, 11],
        [12, 8, 11],
        [12, 7, 10],
        10]
FC_quant_w = 6   #FC weights


residual_CONV = [False, False, False, True, False, True, True, False, True, 
                  True, True, False, True, True, False, True, True, False, False]

activation_ReLU = 0

def ConvBNReLU(in_map, kernels, bias, stride, in_size, out_size, mean, std, activation, groups=0):
    out_map = functions.conv_layer(in_map, kernels, bias, stride, in_size, out_size, groups)
    out_map = functions.BatchNorm(out_map, out_size, mean, std)
    out_map = functions.ReLU6(out_map, activation)
    return out_map

def InvertedResidual(in_map, kernels, bias, stride, size, activation, residual):
    out_map = ConvBNReLU(in_map, kernels[0][0], False, stride[0], size[0][0], size[0][1], kernels[0][1], bias[0][1], activation)
    out_map = ConvBNReLU(out_map, kernels[1][0], False, stride[1], size[1][0], size[1][1], kernels[1][1], bias[1][1], activation, groups=size[1][0])
    out_map = functions.conv_layer(out_map, kernels[2], False, stride[2], size[2][0], size[2][1])
    out_map = functions.BatchNorm(out_map, size[2][1], kernels[3], bias[3])
    if residual:
        return in_map + out_map
    else:
        return out_map
    
def ConvBNReLU_merge(in_map, kernels, bias, stride, in_size, out_size, activation, bit_frac, bit_w, groups=0, quant = False, quant_int = False):
    out_map = functions.conv_layer(in_map, kernels, bias, stride, in_size, out_size, groups)
    if quant:
        out_map = functions.linear_quant_weights(out_map, bit_frac, fixed_act[0])
    if quant_int:
        out_map = functions.ReLU6_quant(out_map, activation, 6 * 2**(bit_frac + bit_w))
    else:    
        out_map = functions.ReLU6(out_map, activation)
    return out_map

def InvertedResidual_merge(in_map, kernels, bias, stride, size, activation, residual, bit_frac, layer, quant = False, quant_int = False):
    if quant_int:
        in_map = np.float64(np.round(in_map * (2**(CONV_quan_act[layer][0] - CONV_quan_act[layer-1][2] - CONV_quan_w[layer-1][2]))))
        kernels[0] = np.float64(np.round(kernels[0] * (2**(CONV_quan_w[layer][0]))))
        bias[0] = np.float64(np.round(bias[0] * (2**(CONV_quan_w[layer][0]))))*(2**CONV_quan_act[layer][0])
    out_map = ConvBNReLU_merge(in_map, kernels[0], bias[0], stride[0], size[0][0], size[0][1], activation, bit_frac[0], CONV_quan_w[layer][0], 0, quant, quant_int)
    if quant_int:
        out_map = np.float64(np.round(out_map * (2**(CONV_quan_act[layer][1] - CONV_quan_act[layer][0] - CONV_quan_w[layer][0]))))
        kernels[1] = np.float64(np.round(kernels[1] * (2**(CONV_quan_w[layer][1]))))
        bias[1] = np.float64(np.round(bias[1] * (2**(CONV_quan_w[layer][1]))))*(2**CONV_quan_act[layer][1])
    out_map = ConvBNReLU_merge(out_map, kernels[1], bias[1], stride[1], size[1][0], size[1][1], activation, bit_frac[1], CONV_quan_w[layer][1],size[1][0], quant, quant_int)
    if quant_int:
        out_map = np.float64(np.round(out_map * (2**(CONV_quan_act[layer][2] - CONV_quan_act[layer][1] - CONV_quan_w[layer][1]))))
        kernels[2] = np.float64(np.round(kernels[2] * (2**(CONV_quan_w[layer][2]))))
        bias[2] = np.float64(np.round(bias[2] * (2**(CONV_quan_w[layer][2]))))*(2**CONV_quan_act[layer][2])
    out_map = functions.conv_layer(out_map, kernels[2], bias[2], stride[2], size[2][0], size[2][1])
    if quant:
        out_map = functions.linear_quant_weights(out_map, bit_frac[2], fixed_act[0])
    if residual:
        if quant_int:
            return np.float64(np.int64(in_map * (2**(CONV_quan_act[layer][2] + CONV_quan_w[layer][2] - CONV_quan_act[layer][0])))) + out_map
        else:
            return in_map + out_map
    else:
        return out_map
    
def MobileNet_V2(image, kernels, bias, merge_batch, layer_out = None, quant = False, quant_int = False):
    # Model
    print("Applying model...")
    #GROUP 1
    print("ConvBNReLU 0...")
    layer = 0
    if merge_batch:
        if quant_int:
            image = np.float64(np.round(image * (2**CONV_quan_act[layer])))
            kernels[layer] = np.float64(np.round(kernels[layer] * (2**(CONV_quan_w[layer]))))
            bias[layer] = np.float64(np.round(bias[layer] * (2**(CONV_quan_w[layer]))))*(2**CONV_quan_act[layer])
        out_map = ConvBNReLU_merge(image, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer][0], CONV_size[layer][1],
                                    activation_ReLU, CONV_quan_act[layer], CONV_quan_w[layer],0, quant, quant_int)
    else:   
        out_map = ConvBNReLU(image, kernels[layer][0], False, stride_CONV[layer], CONV_size[layer][0], CONV_size[layer][1],
                              kernels[layer][1], bias[layer][1], activation_ReLU)
    if layer_out == 0:
        return out_map
    
    #GROUP 2
    print("InvertedResidual 1...")
    layer = 1
    if merge_batch:
        if quant_int:
            out_map = np.float64(np.round(out_map * (2**(CONV_quan_act[layer][0] - CONV_quan_act[layer-1] - CONV_quan_w[layer-1]))))
            kernels[layer][0] = np.float64(np.round(kernels[layer][0] * (2**(CONV_quan_w[layer][0]))))
            bias[layer][0] = np.float64(np.round(bias[layer][0] * (2**(CONV_quan_w[layer][0]))))*(2**CONV_quan_act[layer][0])
        out_map = ConvBNReLU_merge(out_map, kernels[layer][0], bias[layer][0], stride_CONV[layer][0], CONV_size[layer][0][0], CONV_size[layer][0][1],
                                    activation_ReLU, CONV_quan_act[layer][0], CONV_quan_w[layer][0], CONV_size[layer][0][1], quant, quant_int)
        if quant_int:
            out_map = np.float64(np.round(out_map * (2**(CONV_quan_act[layer][1] - CONV_quan_act[layer][0] - CONV_quan_w[layer][0]))))
            kernels[layer][1] = np.float64(np.round(kernels[layer][1] * (2**(CONV_quan_w[layer][1]))))
            bias[layer][1] = np.float64(np.round(bias[layer][1] * (2**(CONV_quan_w[layer][1]))))*(2**CONV_quan_act[layer][1])
        out_map = functions.conv_layer(out_map, kernels[layer][1], bias[layer][1], stride_CONV[layer][1], CONV_size[layer][1][0],
                                        CONV_size[layer][1][1])
        if quant:
            out_map = functions.linear_quant_weights(out_map, CONV_quan_act[layer][1] , fixed_act[0])
    else:
        out_map = ConvBNReLU(out_map, kernels[layer][0][0], False, stride_CONV[layer][0], CONV_size[layer][0][0], CONV_size[layer][0][1],
                              kernels[layer][0][1], bias[layer][0][1], activation_ReLU, groups=CONV_size[layer][0][1])
        out_map = functions.conv_layer(out_map, kernels[layer][1], False, stride_CONV[layer][1], CONV_size[layer][1][0], CONV_size[layer][1][1])
        out_map = functions.BatchNorm(out_map, CONV_size[layer][1][1], kernels[layer][2], bias[layer][2])
    if layer_out == 1:
        return out_map
    
    #GROUP 3
    print("InvertedResidual 2...")
    layer = 2
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int) 
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 2:
        return out_map
    
    #GROUP 4
    print("InvertedResidual 3...")
    layer = 3
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int)  
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 3:
        return out_map

    #GROUP 5
    print("InvertedResidual 4...")
    layer = 4
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int) 
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 4:
        return out_map
    
    #GROUP 6
    print("InvertedResidual 5...")
    layer = 5
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int) 
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 5:
        return out_map

    #GROUP 7
    print("InvertedResidual 6...")
    layer = 6
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int) 
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 6:
        return out_map

    #GROUP 8
    print("InvertedResidual 7...")
    layer = 7
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int) 
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 7:
        return out_map

    #GROUP 9
    print("InvertedResidual 8...")
    layer = 8
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int) 
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 8:
        return out_map
    
    #GROUP 10
    print("InvertedResidual 9...")
    layer = 9
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int)  
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 9:
        return out_map

    #GROUP 11
    print("InvertedResidual 10...")
    layer = 10
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int) 
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 10:
        return out_map

    #GROUP 12
    print("InvertedResidual 11...")
    layer = 11
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int) 
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 11:
        return out_map

    #GROUP 13
    print("InvertedResidual 12...")
    layer = 12
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int) 
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 12:
        return out_map

    #GROUP 14
    print("InvertedResidual 13...")
    layer = 13
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int) 
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 13:
        return out_map
    
    #GROUP 15
    print("InvertedResidual 14...")
    layer = 14
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int) 
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 14:
        return out_map

    #GROUP 16
    print("InvertedResidual 15...")
    layer = 15
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int) 
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 15:
        return out_map

    #GROUP 17
    print("InvertedResidual 16...")
    layer = 16
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int) 
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 16:
        return out_map

    #GROUP 18
    print("InvertedResidual 17...")
    layer = 17
    if merge_batch:
        out_map = InvertedResidual_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer], activation_ReLU, 
                                          residual_CONV[layer], CONV_quan_act[layer], layer, quant, quant_int) 
    else:
        out_map = InvertedResidual(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer],
                                    activation_ReLU, residual_CONV[layer])
    if layer_out == 17:
        return out_map

    #GROUP 19
    print("ConvBNReLU 18...")
    layer = 18
    if merge_batch:
        if quant_int:
            out_map = np.float64(np.round(out_map * (2**(CONV_quan_act[layer] - CONV_quan_act[layer-1][2] - CONV_quan_w[layer-1][2]))))
            kernels[layer] = np.float64(np.round(kernels[layer] * (2**(CONV_quan_w[layer]))))
            bias[layer] = np.float64(np.round(bias[layer] * (2**(CONV_quan_w[layer]))))*(2**CONV_quan_act[layer])
        out_map = ConvBNReLU_merge(out_map, kernels[layer], bias[layer], stride_CONV[layer], CONV_size[layer][0],
                              CONV_size[layer][1], activation_ReLU, CONV_quan_act[layer], CONV_quan_w[layer], 0, quant, quant_int)
    else:        
        out_map = ConvBNReLU(out_map, kernels[layer][0], False, stride_CONV[layer], CONV_size[layer][0], CONV_size[layer][1],
                              kernels[layer][1], bias[layer][1], activation_ReLU)
    if layer_out == 18:
        return out_map
    
    #GROUP 20
    print("GlobalAvg 19...")
    layer = 19
    if quant_int:
        out_map = np.float64(np.round(out_map * (2**(FC_quant_act[0] - CONV_quan_act[layer-1] - CONV_quan_w[layer-1]))))
    out_map = functions.Global_Avg(out_map, CONV_size[layer-1][1])
    if quant:
        out_map = functions.linear_quant_weights(out_map, FC_quant_act[0] , fixed_act[1])
    if layer_out == 19:
        return out_map

    #FC 1(Dense)
    print("FC 20...")
    layer = 20
    if quant_int:
        out_map = np.float64(np.round(out_map * (2**(FC_quant_act[1] - FC_quant_act[0]))))
        kernels[layer-1] = np.float64(np.round(kernels[layer-1] * (2**(FC_quant_w))))
        bias[layer-1] = np.float64(np.round(bias[layer-1] * (2**(FC_quant_w))))*(2**FC_quant_act[1])
    out_map = functions.fully_connected_layer(out_map, kernels[layer-1], bias[layer-1])
    if quant:
        out_map = functions.linear_quant_weights(out_map, FC_quant_act[1] , fixed_act[2])
    if layer_out == 20:
        return out_map

    #Softmax
    print("Softmax...")
    layer = 21
    if quant_int:
        out_map = out_map / (2**(FC_quant_act[1] + FC_quant_w))
    out_map = functions.softmax(out_map, FC_size[1])
    
    return out_map















