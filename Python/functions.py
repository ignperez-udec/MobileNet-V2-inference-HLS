from torchvision import models
import numpy as np
import torch.nn as nn
import cv2
from matplotlib import pylab
from numpy.lib.stride_tricks import as_strided

#%%READ AND EXTRAS
def read_weights(model, type_model):
    weights = []
    bias = []
    #VGG
    if type_model == "VGG16":
        for child in model.children():
            if isinstance(child, nn.Sequential):
                for i in child:
                    if isinstance(i, (nn.Conv2d, nn.Linear)):
                        weights.append(i.weight.data.cpu().numpy())
                        bias.append(i.bias.data.cpu().numpy())
    #SqueezeNet
    elif type_model == "SqueezeNet":
        for child in model.children():
            if isinstance(child, nn.Sequential):
                for i in child:
                    if isinstance(i, nn.Conv2d):
                        weights.append(i.weight.data.cpu().numpy())
                        bias.append(i.bias.data.cpu().numpy())
                    elif isinstance(i, models.squeezenet.Fire):
                        fire_weights = []
                        fire_bias = []
                        for j in i.children():
                            if isinstance(j, nn.Conv2d):
                                fire_weights.append(j.weight.data.cpu().numpy())
                                fire_bias.append(j.bias.data.cpu().numpy())
                        weights.append(fire_weights)
                        bias.append(fire_bias)
    #MobileNet V2
    elif type_model == "MobileNet":
        for child in model.children():
            if isinstance(child, nn.Sequential):
                for i in child:
                    if isinstance(i, nn.Linear):
                        weights.append(i.weight.data.cpu().numpy())
                        bias.append(i.bias.data.cpu().numpy())
                    elif isinstance(i, models.mobilenet.ConvBNReLU):
                        convbn_weights = []
                        convbn_bias = []
                        for j in i.children():
                            if isinstance(j, nn.Conv2d):
                                convbn_weights.append(j.weight.data.cpu().numpy())
                                if j.bias == None:
                                    convbn_bias.append(0)
                                else:
                                    convbn_bias.append(j.bias.data.cpu().numpy())
                            elif isinstance(j, nn.BatchNorm2d):
                                batch_weights = []
                                batch_bias = []
                                #Mean and Var
                                batch_weights.append(j.running_mean.data.cpu().numpy())
                                batch_bias.append(j.running_var.data.cpu().numpy())
                                #Gamma and Beta
                                batch_weights.append(j.weight.data.cpu().numpy())
                                batch_bias.append(j.bias.data.cpu().numpy())
                                #Weights to CONBN
                                convbn_weights.append(batch_weights)
                                convbn_bias.append(batch_bias)
                        weights.append(convbn_weights)
                        bias.append(convbn_bias)
                    elif isinstance(i, models.mobilenet.InvertedResidual):
                        inv_weights = []
                        inv_bias = []
                        for j in i.children():
                            if isinstance(j, nn.Sequential):
                                for k in j.children():
                                    if isinstance(k, models.mobilenet.ConvBNReLU):
                                        convbn_weights = []
                                        convbn_bias = []
                                        for l in k.children():
                                            if isinstance(l, nn.Conv2d):
                                                convbn_weights.append(l.weight.data.cpu().numpy())
                                                if l.bias == None:
                                                    convbn_bias.append(0)
                                                else:
                                                    convbn_bias.append(l.bias.data.cpu().numpy())
                                            elif isinstance(l, nn.BatchNorm2d):
                                                batch_weights = []
                                                batch_bias = []
                                                #Mean and Var
                                                batch_weights.append(l.running_mean.data.cpu().numpy())
                                                batch_bias.append(l.running_var.data.cpu().numpy())
                                                #Gamma and Beta
                                                batch_weights.append(l.weight.data.cpu().numpy())
                                                batch_bias.append(l.bias.data.cpu().numpy())
                                                #Weights to CONBN
                                                convbn_weights.append(batch_weights)
                                                convbn_bias.append(batch_bias)
                                        inv_weights.append(convbn_weights)
                                        inv_bias.append(convbn_bias)
                                    elif isinstance(k, nn.Conv2d):
                                        inv_weights.append(k.weight.data.cpu().numpy())
                                        if k.bias == None:
                                            inv_bias.append(0)
                                        else:
                                            inv_bias.append(k.bias.data.cpu().numpy())
                                    elif isinstance(k, nn.BatchNorm2d):
                                        batch_weights = []
                                        batch_bias = []
                                        #Mean and Var
                                        batch_weights.append(k.running_mean.data.cpu().numpy())
                                        batch_bias.append(k.running_var.data.cpu().numpy())
                                        #Gamma and Beta
                                        batch_weights.append(k.weight.data.cpu().numpy())
                                        batch_bias.append(k.bias.data.cpu().numpy())
                                        #Weights to CONBN
                                        inv_weights.append(batch_weights)
                                        inv_bias.append(batch_bias)
                        weights.append(inv_weights)
                        bias.append(inv_bias)
    return weights, bias

def weights_reduction(kernels, semi_prune, merge_weight_index):
    vector_weights_CONV = []
    vector_weights_FC = []
    vector_index_CONV = []
    vector_index_FC = []
    vector_weights_layer = []
    vector_index_layer = []
    vector_bias_layer = []
    number_zeros_CONV = []
    number_zeros_FC = []
    counter_w = 0
    counter_i = 0
    index = 0
    zeros = 0
    
    vector_weights_layer.append(0)
    vector_index_layer.append(0)
    vector_bias_layer.append(0)
    
    #MobileNet V2
    #CONVs
    for i in range(19):
        #CONVBNReLU 0
        if i == 0:
            counter_w = 0
            counter_i = 0
            index = 0
            for j in range(kernels[i].shape[0]):
                for k in range(kernels[i].shape[1]):
                    for l in range(kernels[i].shape[2]):
                        for m in range(kernels[i].shape[3]):
                            if semi_prune:
                                vector_weights_CONV.append(kernels[i][j][k][l][m])
                                if merge_weight_index:
                                    vector_index_CONV.append(index)
                                counter_w = counter_w + 1
                            else:
                                if kernels[i][j][k][l][m] != 0:
                                    vector_weights_CONV.append(kernels[i][j][k][l][m])
                                    vector_index_CONV.append(index)
                                    index = 0
                                    counter_i = counter_i + 1
                                    counter_w = counter_w + 1
                                else:
                                    index = index + 1
                    index = 0
            vector_weights_layer.append(counter_w)
            vector_index_layer.append(counter_i)
            vector_bias_layer.append(kernels[i].shape[0])
        #InvertedResidual 1 (second group)
        elif i == 1:
            #CONVBNRelu 0 (internal)
            counter_w = 0
            counter_i = 0
            index = 0
            for j in range(kernels[i][0].shape[0]):
                for k in range(kernels[i][0].shape[1]):
                    for l in range(kernels[i][0].shape[2]):
                        for m in range(kernels[i][0].shape[3]):
                            if semi_prune:
                                vector_weights_CONV.append(kernels[i][0][j][k][l][m])
                                if merge_weight_index:
                                    vector_index_CONV.append(index)
                                counter_w = counter_w + 1
                            else:
                                if kernels[i][0][j][k][l][m] != 0:
                                    vector_weights_CONV.append(kernels[i][0][j][k][l][m])
                                    vector_index_CONV.append(index)
                                    index = 0
                                    counter_i = counter_i + 1
                                    counter_w = counter_w + 1
                                else:
                                    index = index + 1
                    index = 0
            vector_weights_layer.append(counter_w)
            vector_index_layer.append(counter_i)
            vector_bias_layer.append(kernels[i][0].shape[0])
            #CONV 1 (internal)
            counter_w = 0
            counter_i = 0
            index = 0
            for j in range(kernels[i][1].shape[0]):
                zeros = 0
                for k in range(kernels[i][1].shape[1]):
                    for l in range(kernels[i][1].shape[2]):
                        for m in range(kernels[i][1].shape[3]):
                            if kernels[i][1][j][k][l][m] != 0:
                                vector_weights_CONV.append(kernels[i][1][j][k][l][m])
                                vector_index_CONV.append(index)
                                index = 0
                                counter_w = counter_w + 1
                                counter_i = counter_i + 1
                            else:
                                zeros = zeros + 1
                                index = index + 1
                number_zeros_CONV.append(zeros)
            vector_weights_layer.append(counter_w)
            vector_index_layer.append(counter_i)
            vector_bias_layer.append(kernels[i][1].shape[0])
        #CONVBNReLU 18
        elif i == 18:
            counter_w = 0
            counter_i = 0
            index = 0
            for j in range(kernels[i].shape[0]):
                zeros = 0
                for k in range(kernels[i].shape[1]):
                    for l in range(kernels[i].shape[2]):
                        for m in range(kernels[i].shape[3]):
                            if kernels[i][j][k][l][m] != 0:
                                vector_weights_CONV.append(kernels[i][j][k][l][m])
                                vector_index_CONV.append(index)
                                index = 0
                                counter_w = counter_w + 1
                                counter_i = counter_i + 1
                            else:
                                zeros = zeros + 1
                                index = index + 1
                number_zeros_CONV.append(zeros)
            vector_weights_layer.append(counter_w)
            vector_index_layer.append(counter_i)
            vector_bias_layer.append(kernels[i].shape[0])
        #InvertedResidual (all)
        else:
            #CONVBNRelu 0 (internal)
            counter_w = 0
            counter_i = 0
            index = 0
            for j in range(kernels[i][0].shape[0]):
                zeros = 0
                for k in range(kernels[i][0].shape[1]):
                    for l in range(kernels[i][0].shape[2]):
                        for m in range(kernels[i][0].shape[3]):
                            if kernels[i][0][j][k][l][m] != 0:
                                vector_weights_CONV.append(kernels[i][0][j][k][l][m])
                                vector_index_CONV.append(index)
                                index = 0
                                counter_w = counter_w + 1
                                counter_i = counter_i + 1
                            else:
                                zeros = zeros + 1
                                index = index + 1
                number_zeros_CONV.append(zeros)
            vector_weights_layer.append(counter_w)
            vector_index_layer.append(counter_i)
            vector_bias_layer.append(kernels[i][0].shape[0])
            #CONVBNRelu 1 (internal)
            counter_w = 0
            counter_i = 0
            index = 0
            for j in range(kernels[i][1].shape[0]):
                for k in range(kernels[i][1].shape[1]):
                    for l in range(kernels[i][1].shape[2]):
                        for m in range(kernels[i][1].shape[3]):
                            if semi_prune:
                                vector_weights_CONV.append(kernels[i][1][j][k][l][m])
                                if merge_weight_index:
                                    vector_index_CONV.append(index)
                                counter_w = counter_w + 1
                            else:
                                if kernels[i][1][j][k][l][m] != 0:
                                    vector_weights_CONV.append(kernels[i][1][j][k][l][m])
                                    vector_index_CONV.append(index)
                                    index = 0
                                    counter_i = counter_i + 1
                                    counter_w = counter_w + 1
                                else:
                                    index = index + 1
                    index = 0
            vector_weights_layer.append(counter_w)
            vector_index_layer.append(counter_i)
            vector_bias_layer.append(kernels[i][1].shape[0])
            #CONV 2 (internal)
            counter_w = 0
            counter_i = 0
            index = 0
            for j in range(kernels[i][2].shape[0]):
                zeros = 0
                for k in range(kernels[i][2].shape[1]):
                    for l in range(kernels[i][2].shape[2]):
                        for m in range(kernels[i][2].shape[3]):
                            if kernels[i][2][j][k][l][m] != 0:
                                vector_weights_CONV.append(kernels[i][2][j][k][l][m])
                                vector_index_CONV.append(index)
                                index = 0
                                counter_w = counter_w + 1
                                counter_i = counter_i + 1
                            else:
                                zeros = zeros + 1
                                index = index + 1
                number_zeros_CONV.append(zeros)
            vector_weights_layer.append(counter_w)
            vector_index_layer.append(counter_i)
            vector_bias_layer.append(kernels[i][2].shape[0])
    #FCs
    index = 0
    for i in range(kernels[19].shape[0]):
        zeros = 0
        for j in range(kernels[19].shape[1]):
            if kernels[19][i][j] != 0:
                vector_weights_FC.append(kernels[19][i][j])
                vector_index_FC.append(index)
                index = 0
            else:
                zeros = zeros + 1
                index = index + 1 
        number_zeros_FC.append(zeros)

    return vector_weights_CONV, vector_weights_FC, vector_index_CONV, vector_index_FC, vector_weights_layer, vector_index_layer, vector_bias_layer, number_zeros_CONV, number_zeros_FC

def merge_conv_batch(kernels, bias):
    out_weights=[]
    out_bias=[]
    e=0.00001
    #Merge BATCHs in CONV layers
    for i in range(19):
        #Mean -> kernel[0]
        #Gamma -> kernel[1]
        #Var -> bias[0]
        #Beta -> bias[1]
        #CONVBNReLU 0 y 18 (first and last CONVs)
        if i == 0 or i == 18:
            N, M, K = kernels[i][0].shape[0:3]
            W_bn = kernels[i][1][1]/np.sqrt(bias[i][1][0] + e) * np.identity(N)
            b_bn = bias[i][1][1] - kernels[i][1][1]*kernels[i][1][0]/np.sqrt(bias[i][1][0] + e)
            out_weights.append(np.matmul(W_bn, kernels[i][0].reshape(N, M*K*K)).reshape(N, M, K, K))
            out_bias.append(b_bn)
        #InvertedResidual 1 (second group)
        elif i == 1:
            temp_w = []
            temp_b = []
            #CONVBNRelu 0 (internal)
            N, M, K = kernels[i][0][0].shape[0:3]
            W_bn = kernels[i][0][1][1]/np.sqrt(bias[i][0][1][0] + e) * np.identity(N)
            b_bn = bias[i][0][1][1] - kernels[i][0][1][1]*kernels[i][0][1][0]/np.sqrt(bias[i][0][1][0] + e)
            temp_w.append(np.matmul(W_bn, kernels[i][0][0].reshape(N, M*K*K)).reshape(N, M, K, K))
            temp_b.append(b_bn)
            #BATCH 1 (internal)
            N, M, K = kernels[i][1].shape[0:3]
            W_bn = kernels[i][2][1]/np.sqrt(bias[i][2][0] + e) * np.identity(N)
            b_bn = bias[i][2][1] - kernels[i][2][1]*kernels[i][2][0]/np.sqrt(bias[i][2][0] + e)
            temp_w.append(np.matmul(W_bn, kernels[i][1].reshape(N, M*K*K)).reshape(N, M, K, K))
            temp_b.append(b_bn)
            out_weights.append(temp_w)
            out_bias.append(temp_b)
        #InvertedResidual (all)
        else:
            temp_w = []
            temp_b = []
            #CONVBNRelu 0 (internal)
            N, M, K = kernels[i][0][0].shape[0:3]
            W_bn = kernels[i][0][1][1]/np.sqrt(bias[i][0][1][0] + e) * np.identity(N)
            b_bn = bias[i][0][1][1] - kernels[i][0][1][1]*kernels[i][0][1][0]/np.sqrt(bias[i][0][1][0] + e)
            temp_w.append(np.matmul(W_bn, kernels[i][0][0].reshape(N, M*K*K)).reshape(N, M, K, K))
            temp_b.append(b_bn)
            #CONVBNRelu 1 (internal)
            N, M, K = kernels[i][1][0].shape[0:3]
            W_bn = kernels[i][1][1][1]/np.sqrt(bias[i][1][1][0] + e) * np.identity(N)
            b_bn = bias[i][1][1][1] - kernels[i][1][1][1]*kernels[i][1][1][0]/np.sqrt(bias[i][1][1][0] + e)
            temp_w.append(np.matmul(W_bn, kernels[i][1][0].reshape(N, M*K*K)).reshape(N, M, K, K))
            temp_b.append(b_bn)
            #BATCH 2 (internal)
            N, M, K = kernels[i][2].shape[0:3]
            W_bn = kernels[i][3][1]/np.sqrt(bias[i][3][0] + e) * np.identity(N)
            b_bn = bias[i][3][1] - kernels[i][3][1]*kernels[i][3][0]/np.sqrt(bias[i][3][0] + e)
            temp_w.append(np.matmul(W_bn, kernels[i][2].reshape(N, M*K*K)).reshape(N, M, K, K))
            temp_b.append(b_bn)
            out_weights.append(temp_w)
            out_bias.append(temp_b)
    #FCs
    out_weights.append(kernels[19])
    out_bias.append(bias[19])
    return out_weights, out_bias

def merge_weights_index(weights_CONV, weights_FC, index_CONV, index_FC, weights_for_layer, index_for_layer, fixed):
    out_CONV = []
    out_FC = []
    index = 0
    add = weights_for_layer[1]
    #WEIGHTS
    #CONV
    for i in range(len(weights_CONV)):
        if i == add:
            if index + 2 < len(weights_for_layer):
                add = add + weights_for_layer[index + 2]
            index = index + 1
        if index_for_layer[index] == 0:
            out_CONV.append(weights_CONV[i])
        else:
            if weights_CONV[i] >= 0:
                out_CONV.append(weights_CONV[i] + index_CONV[i]*(2**fixed[0]))
            else:
                out_CONV.append(2**fixed[0] - np.abs(weights_CONV[i]) + index_CONV[i]*(2**fixed[0]))
    #FC
    for i in range(len(weights_FC)):
        if weights_FC[i] >= 0:
            out_FC.append(weights_FC[i] + index_FC[i]*(2**fixed[1]))
        else:
            out_FC.append(2**fixed[1] - np.abs(weights_FC[i]) + index_FC[i]*(2**fixed[1]))
    
    return out_CONV, out_FC

def merge_weights_index_float(weights_CONV, weights_FC, index_CONV, index_FC, weights_for_layer, index_for_layer):
    out_CONV = []
    out_FC = []
    index = 0
    add = weights_for_layer[1]
    #WEIGHTS
    #CONV
    for i in range(len(weights_CONV)):
        if i == add:
            if index + 2 < len(weights_for_layer):
                add = add + weights_for_layer[index + 2]
            index = index + 1
        if index_for_layer[index] == 0:
            out_CONV.append(weights_CONV[i])
        else:
            if weights_CONV[i] >= 0:
                out_CONV.append(weights_CONV[i] + index_CONV[i]*1000)
            else:
                out_CONV.append(weights_CONV[i] - index_CONV[i]*1000)
    #FC
    for i in range(len(weights_FC)):
        if weights_FC[i] >= 0:
            out_FC.append(weights_FC[i] + index_FC[i]*1000)
        else:
            out_FC.append(weights_FC[i] - index_FC[i]*1000)
    
    return out_CONV, out_FC

def number_zeros_PE(kernels, tile_conv_out, tile_conv_in, tile_fc_out, tile_fc_in):
    weights_no_zeros = []
    zeros_for_all_PE = []
    zeros_for_final_PE = []
    
    #MobileNet V2
    #CONVs
    for i in range(1, 19):
        #CONVBNReLU 0 (i == 0)
        #NO PRUNING
        #InvertedResidual 1 (second group)
        if i == 1:
            #CONVBNRelu 0 (internal)
            #NO PRUNING
            #CONV 1 (internal)
            counter_w_no_zeros = 0
            counter_zeros_all = 0
            counter_zeros_final = 0
            for j in range(0, np.min([kernels[i][1].shape[1], tile_conv_in])):
                if kernels[i][1][0][j][0][0] == 0:
                    counter_zeros_all = counter_zeros_all + 1
            final_list = list(range(0, kernels[i][1].shape[1], tile_conv_in))
            final_value = final_list[len(final_list) - 1]
            for k in range(final_value, (kernels[i][1].shape[1])):
                if kernels[i][1][0][k][0][0] == 0:
                    counter_zeros_final = counter_zeros_final + 1
            for j in range(kernels[i][1].shape[1]):
                if kernels[i][1][0][j][0][0] != 0:
                    counter_w_no_zeros = counter_w_no_zeros + 1
            weights_no_zeros.append(counter_w_no_zeros)
            zeros_for_all_PE.append(counter_zeros_all)
            zeros_for_final_PE.append(counter_zeros_final)
        #CONVBNReLU 18
        elif i == 18:
            counter_w_no_zeros = 0
            counter_zeros_all = 0
            counter_zeros_final = 0
            for j in range(0, np.min([kernels[i].shape[1], tile_conv_in])):
                if kernels[i][0][j][0][0] == 0:
                    counter_zeros_all = counter_zeros_all + 1
            final_list = list(range(0, kernels[i].shape[1], tile_conv_in))
            final_value = final_list[len(final_list) - 1]
            for k in range(final_value, kernels[i].shape[1]):
                if kernels[i][0][k][0][0] == 0:
                    counter_zeros_final = counter_zeros_final + 1
            for j in range(kernels[i].shape[1]):
                if kernels[i][0][j][0][0] != 0:
                    counter_w_no_zeros = counter_w_no_zeros + 1
            weights_no_zeros.append(counter_w_no_zeros)
            zeros_for_all_PE.append(counter_zeros_all)
            zeros_for_final_PE.append(counter_zeros_final)
        #InvertedResidual (all)
        else:
            #CONVBNRelu 0 (internal)
            counter_w_no_zeros = 0
            counter_zeros_all = 0
            counter_zeros_final = 0
            for j in range(0, np.min([kernels[i][0].shape[1], tile_conv_in])):
                if kernels[i][0][0][j][0][0] == 0:
                    counter_zeros_all = counter_zeros_all + 1
            final_list = list(range(0, kernels[i][0].shape[1], tile_conv_in))
            final_value = final_list[len(final_list) - 1]
            for k in range(final_value, (kernels[i][0].shape[1])):
                if kernels[i][0][0][k][0][0] == 0:
                    counter_zeros_final = counter_zeros_final + 1
            for j in range(kernels[i][0].shape[1]):
                if kernels[i][0][0][j][0][0] != 0:
                    counter_w_no_zeros = counter_w_no_zeros + 1
            weights_no_zeros.append(counter_w_no_zeros)
            zeros_for_all_PE.append(counter_zeros_all)
            zeros_for_final_PE.append(counter_zeros_final)
            #CONVBNRelu 1 (internal)
            #NO PRUNING
            #CONV 2 (internal)
            counter_w_no_zeros = 0
            counter_zeros_all = 0
            counter_zeros_final = 0
            for j in range(0, np.min([kernels[i][2].shape[1], tile_conv_in])):
                if kernels[i][2][0][j][0][0] == 0:
                    counter_zeros_all = counter_zeros_all + 1
            final_list = list(range(0, kernels[i][2].shape[1], tile_conv_in))
            final_value = final_list[len(final_list) - 1]
            for k in range(final_value, (kernels[i][2].shape[1])):
                if kernels[i][2][0][k][0][0] == 0:
                    counter_zeros_final = counter_zeros_final + 1
            for j in range(kernels[i][2].shape[1]):
                if kernels[i][2][0][j][0][0] != 0:
                    counter_w_no_zeros = counter_w_no_zeros + 1
            weights_no_zeros.append(counter_w_no_zeros)
            zeros_for_all_PE.append(counter_zeros_all)
            zeros_for_final_PE.append(counter_zeros_final)
    #FCs
    counter_w_no_zeros = 0
    counter_zeros_all = 0
    counter_zeros_final = 0
    for j in range(0, np.min([kernels[19].shape[1], tile_fc_in])):
        if kernels[19][0][j] == 0:
            counter_zeros_all = counter_zeros_all + 1
    final_list = list(range(0, kernels[19].shape[1], tile_fc_in))
    final_value = final_list[len(final_list) - 1]
    for k in range(final_value, (kernels[19].shape[1])):
        if kernels[19][0][k] == 0:
            counter_zeros_final = counter_zeros_final + 1
    for j in range(kernels[19].shape[1]):
        if kernels[19][0][j] != 0:
            counter_w_no_zeros = counter_w_no_zeros + 1
    weights_no_zeros.append(counter_w_no_zeros)
    zeros_for_all_PE.append(counter_zeros_all)
    zeros_for_final_PE.append(counter_zeros_final)

    return weights_no_zeros, zeros_for_all_PE, zeros_for_final_PE


def quant_weights(kernels, bias, bit_conv, bit_fc, type_quant="linear", print_bit = False):
    out_weights=[]
    out_bias=[]
    #CONV
    for i in range(19):
        if i == 0 or i == 18:
            if type_quant == "linear":
                #LINEAR
                if isinstance(bias[i], int) == False:
                    weights_layer = np.concatenate((kernels[i], bias[i]), axis=None)
                else:
                    weights_layer = kernels[i]
                bit_fractional = np.int(bit_conv - 1 - bits_fractional_part_weights(weights_layer))
                if bit_fractional > bit_conv:
                    bit_fractional = bit_conv
                if print_bit: print(np.int(bit_fractional))
                out_weights.append(linear_quant_weights(kernels[i], bit_fractional, bit_conv))
                if isinstance(bias[i], int) == False:
                    out_bias.append(linear_quant_weights(bias[i], bit_fractional, bit_conv))
            else:
                #LOG
                out_weights.append(log_quant_weights(kernels[i], bit_conv))
                if isinstance(bias[i], int) == False:
                    out_bias.append(log_quant_weights(bias[i], bit_conv))
        #InvertedResidual 1 (second group)
        elif i == 1:
            temp_w = []
            temp_b = []
            #CONVBNRelu 0 (internal)
            if type_quant == "linear":
                #LINEAR
                if isinstance(bias[i][0], int) == False:
                    weights_layer = np.concatenate((kernels[i][0], bias[i][0]), axis=None)
                else:
                    weights_layer = kernels[i][0]
                bit_fractional = np.int(bit_conv - 1 - bits_fractional_part_weights(weights_layer))
                if bit_fractional > bit_conv:
                    bit_fractional = bit_conv
                if print_bit: print(np.int(bit_fractional))
                temp_w.append(linear_quant_weights(kernels[i][0], bit_fractional, bit_conv))
                if isinstance(bias[i][0], int) == False:
                    temp_b.append(linear_quant_weights(bias[i][0], bit_fractional, bit_conv))
            else:
                #LOG
                temp_w.append(log_quant_weights(kernels[i][0], bit_conv))
                if isinstance(bias[i][0], int) == False:
                    temp_b.append(log_quant_weights(bias[i][0], bit_conv))
            #CONVBNRelu 1 (internal)
            if type_quant == "linear":
                #LINEAR
                if isinstance(bias[i][1], int) == False:
                    weights_layer = np.concatenate((kernels[i][1], bias[i][1]), axis=None)
                else:
                    weights_layer = kernels[i][1]
                bit_fractional = np.int(bit_conv - 1 - bits_fractional_part_weights(weights_layer))
                if bit_fractional > bit_conv:
                    bit_fractional = bit_conv
                if print_bit: print(np.int(bit_fractional))
                temp_w.append(linear_quant_weights(kernels[i][1], bit_fractional, bit_conv))
                if isinstance(bias[i][1], int) == False:
                    temp_b.append(linear_quant_weights(bias[i][1], bit_fractional, bit_conv))
            else:
                #LOG
                temp_w.append(log_quant_weights(kernels[i][1], bit_conv))
                if isinstance(bias[i][1], int) == False:
                    temp_b.append(log_quant_weights(bias[i][1], bit_conv))
            out_weights.append(temp_w)
            out_bias.append(temp_b)
        #InvertedResidual (all)
        else:
            temp_w = []
            temp_b = []
            #CONVBNRelu 0 (internal)
            if type_quant == "linear":
                #LINEAR
                if isinstance(bias[i][0], int) == False:
                    weights_layer = np.concatenate((kernels[i][0], bias[i][0]), axis=None)
                else:
                    weights_layer = kernels[i][0]
                bit_fractional = np.int(bit_conv - 1 - bits_fractional_part_weights(weights_layer))
                if bit_fractional > bit_conv:
                    bit_fractional = bit_conv
                if print_bit: print(np.int(bit_fractional))
                temp_w.append(linear_quant_weights(kernels[i][0], bit_fractional, bit_conv))
                if isinstance(bias[i][0], int) == False:
                    temp_b.append(linear_quant_weights(bias[i][0], bit_fractional, bit_conv))
            else:
                #LOG
                temp_w.append(log_quant_weights(kernels[i][0], bit_conv))
                if isinstance(bias[i][0], int) == False:
                    temp_b.append(log_quant_weights(bias[i][0], bit_conv))
            #CONVBNRelu 1 (internal)
            if type_quant == "linear":
                #LINEAR
                if isinstance(bias[i][1], int) == False:
                    weights_layer = np.concatenate((kernels[i][1], bias[i][1]), axis=None)
                else:
                    weights_layer = kernels[i][1]
                bit_fractional = np.int(bit_conv - 1 - bits_fractional_part_weights(weights_layer))
                if bit_fractional > bit_conv:
                    bit_fractional = bit_conv
                if print_bit: print(np.int(bit_fractional))
                temp_w.append(linear_quant_weights(kernels[i][1], bit_fractional, bit_conv))
                if isinstance(bias[i][1], int) == False:
                    temp_b.append(linear_quant_weights(bias[i][1], bit_fractional, bit_conv))
            else:
                #LOG
                temp_w.append(log_quant_weights(kernels[i][1], bit_conv))
                if isinstance(bias[i][1], int) == False:
                    temp_b.append(log_quant_weights(bias[i][1], bit_conv))
            #BATCH 2 (internal)
            if type_quant == "linear":
                #LINEAR
                if isinstance(bias[i][2], int) == False:
                    weights_layer = np.concatenate((kernels[i][2], bias[i][2]), axis=None)
                else:
                    weights_layer = kernels[i][2]
                bit_fractional = np.int(bit_conv - 1 - bits_fractional_part_weights(weights_layer))
                if bit_fractional > bit_conv:
                    bit_fractional = bit_conv
                if print_bit: print(np.int(bit_fractional))
                temp_w.append(linear_quant_weights(kernels[i][2], bit_fractional, bit_conv))
                if isinstance(bias[i][2], int) == False:
                    temp_b.append(linear_quant_weights(bias[i][2], bit_fractional, bit_conv))
            else:
                #LOG
                temp_w.append(log_quant_weights(kernels[i][2], bit_conv))
                if isinstance(bias[i][2], int) == False:
                    temp_b.append(log_quant_weights(bias[i][2], bit_conv))
            out_weights.append(temp_w)
            out_bias.append(temp_b)
    #FCs
    if type_quant == "linear":
        #LINEAR
        if isinstance(bias[19], int) == False:
            weights_layer = np.concatenate((kernels[19], bias[19]), axis=None)
        else:
            weights_layer = kernels[19]
        bit_fractional = np.int(bit_fc - 1 - bits_fractional_part_weights(weights_layer))
        if bit_fractional > bit_fc:
            bit_fractional = bit_fc
        if print_bit: print(np.int(bit_fractional))
        out_weights.append(linear_quant_weights(kernels[19], bit_fractional, bit_fc))
        if isinstance(bias[19], int) == False:
            out_bias.append(linear_quant_weights(bias[19], bit_fractional, bit_fc))
    else:
        #LOG
        out_weights.append(log_quant_weights(kernels[i][1], bit_conv))
        if isinstance(bias[i][1], int) == False:
            out_bias.append(log_quant_weights(bias[i][1], bit_conv))
    return out_weights, out_bias

#Cuantification methods weights
#Lineal quantification
def bits_fractional_part_weights(data):
    abs_value = np.abs(data)
    max_value = np.max(abs_value)
#        print(max_value)
    bit_fractional = np.ceil(np.log2(max_value))
    return bit_fractional

def linear_quant_weights(data, bit_fractional, fixed):
    delta = 2**(-bit_fractional)
    bound = 2**(fixed-1)
    min_val = - bound
    max_val = bound - 1
    rounded = np.round(data / delta)
    clipped_value = np.clip(rounded, min_val, max_val) * delta
    return clipped_value

#Log quantification
def log_quant_weights(data, bits):
    log_data = np.log(np.abs(data))
    value = minmax_quant_weights(log_data, bits)
    value = np.exp(value)*np.sign(data)
    return value

def minmax_quant_weights(data, bits):
    min_val = np.min(data) 
    max_val = np.max(data)    
    value_rescale = (data - min_val) / (max_val - min_val)   
    value_bits = 2**(bits) - 1
    value_round = np.round(value_rescale * value_bits) / value_bits
    value_round =  value_round * (max_val - min_val) + min_val
    return value_round

##QUANTIFICATION TO INT
def float_to_int(weights_CONV, weights_FC, bias, weights_for_layer, CONV_quan_w, FC_quan_w, CONV_quan_act, FC_quan_act):
    out_CONV = []
    out_FC = []
    out_bias = []
    temp_b = []
    index = 0
    add = weights_for_layer[1]
    #WEIGHTS
    #CONV
    for i in range(len(weights_CONV)):
        if i == add:
            if index + 2 < len(weights_for_layer):
                add = add + weights_for_layer[index + 2]
            index = index + 1
        out_CONV.append(np.int32(np.round(weights_CONV[i] * (2**CONV_quan_w[index]))))
    #FC
    for i in range(len(weights_FC)):
        out_FC.append(np.int32(np.round(weights_FC[i] * (2**FC_quan_w))))
    #BIAS
    #CONV
    index = -1
    for i in range(len(bias) - 1):
        if i == 0 or i == 18:
            index = index + 1
            out_bias.append(np.int32(np.round(bias[i] * (2**CONV_quan_w[index])))*(2**CONV_quan_act[index]))
        elif i == 1:
            temp_b = []
            #Group 0
            index = index + 1
            temp_b.append(np.int32(np.round(bias[i][0] * (2**CONV_quan_w[index])))*(2**CONV_quan_act[index]))
            #Group 1
            index = index + 1
            temp_b.append(np.int32(np.round(bias[i][1] * (2**CONV_quan_w[index])))*(2**CONV_quan_act[index]))
            #Copy weights
            out_bias.append(temp_b)
        else:
            temp_b = []
            #Group 0
            index = index + 1
            temp_b.append(np.int32(np.round(bias[i][0] * (2**CONV_quan_w[index])))*(2**CONV_quan_act[index]))
            #Group 1
            index = index + 1
            temp_b.append(np.int32(np.round(bias[i][1] * (2**CONV_quan_w[index])))*(2**CONV_quan_act[index]))
            #Group 2
            index = index + 1
            temp_b.append(np.int32(np.round(bias[i][2] * (2**CONV_quan_w[index])))*(2**CONV_quan_act[index]))
            #Copy weights
            out_bias.append(temp_b)
    #FC
    out_bias.append(np.int32(np.round(bias[19] * (2**FC_quan_w)))*(2**FC_quan_act[1]))
    
    return out_CONV, out_FC, out_bias 

def quant_values_layers(act_conv, act_avg, act_fc, w_conv, w_fc, residual):
    act_conv_quant = []
    bias_conv_quant = []
    relu_conv_quant = []
    res_conv_quant = []
    avg_quant = []
    act_fc_quant = []
    bias_fc_quant = []
    
    #CONV
    index = 0
    for i in range(19):
        #CONVBNRelu 0
        if i == 0:
            act_conv_quant.append(act_conv[index] - act_conv[index] - 0)
            bias_conv_quant.append(act_conv[index])
            relu_conv_quant.append(6 * (2**(act_conv[index] + w_conv[index])))
            res_conv_quant.append(0)
            index = index + 1
        #CONVBNRelu 1 
        elif i == 1:
            #CONV 0 (internal)
            act_conv_quant.append(act_conv[index] - act_conv[index - 1] - w_conv[index - 1])
            bias_conv_quant.append(act_conv[index])
            relu_conv_quant.append(6 * (2**(act_conv[index] + w_conv[index])))
            res_conv_quant.append(0)
            index = index + 1
            #CONV 1 (internal)
            act_conv_quant.append(act_conv[index] - act_conv[index - 1] - w_conv[index - 1])
            bias_conv_quant.append(act_conv[index])
            relu_conv_quant.append(6 * (2**(act_conv[index] + w_conv[index])))
            res_conv_quant.append(0)
            index = index + 1
        #CONVBNRelu 18 
        elif i == 18:   
            act_conv_quant.append(act_conv[index] - act_conv[index - 1] - w_conv[index - 1])
            bias_conv_quant.append(act_conv[index])
            relu_conv_quant.append(6 * (2**(act_conv[index] + w_conv[index])))
            res_conv_quant.append(0)
        #InvertedResidual
        else:
            #CONV 0 (internal)
            act_conv_quant.append(act_conv[index] - act_conv[index - 1] - w_conv[index - 1])
            bias_conv_quant.append(act_conv[index])
            relu_conv_quant.append(6 * (2**(act_conv[index] + w_conv[index])))
            res_conv_quant.append(0)
            index = index + 1
            #CONV 1 (internal)
            act_conv_quant.append(act_conv[index] - act_conv[index - 1] - w_conv[index - 1])
            bias_conv_quant.append(act_conv[index])
            relu_conv_quant.append(6 * (2**(act_conv[index] + w_conv[index])))
            res_conv_quant.append(0)
            index = index + 1
            #CONV 2 (internal)
            act_conv_quant.append(act_conv[index] - act_conv[index - 1] - w_conv[index - 1])
            bias_conv_quant.append(act_conv[index])
            relu_conv_quant.append(6 * (2**(act_conv[index] + w_conv[index])))
            if residual[i]:
                res_conv_quant.append(act_conv[index] + w_conv[index] - act_conv[index-3] - w_conv[index-3])
            else:
                res_conv_quant.append(0)
            index = index + 1
    
    #AVG
    avg_quant.append(act_conv[index] + w_conv[index] - act_avg)
    #FC
    act_fc_quant.append(act_fc - act_avg)
    bias_fc_quant.append(act_fc)
            
    return act_conv_quant, bias_conv_quant, relu_conv_quant, res_conv_quant, avg_quant, act_fc_quant, bias_fc_quant

def read_class(path):
    f = open (path,'r')
    clase = f.read()
    clase=clase.split('\n')
    key=[]
    value=[]
    diccionario={}
    for i in range(len(clase)):
        clase[i]=clase[i][1:len(clase[i])-1]
        key.append(int(clase[i].split(':')[0]))
        value.append(clase[i].split(':')[1])
        diccionario[key[i]]=value[i][2:len(value[i])-1]
    f.close()
    return diccionario

def load_image(path, show, test, C = False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    #Transform
    image = cv2.imread(path)
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lim = image_RGB/255
    if image_lim.shape[0]>=image_lim.shape[1]:
        image_resize = cv2.resize(image_lim, (256, np.int(image_lim.shape[0]/(image_lim.shape[1]/256))), 
                                  interpolation=cv2.INTER_LINEAR).astype(np.float32)
    else:
        image_resize = cv2.resize(image_lim, (np.int(image_lim.shape[1]/(image_lim.shape[0]/256)), 256), 
                                  interpolation=cv2.INTER_LINEAR).astype(np.float32)   
    image_trans = image_resize.transpose((2, 0, 1))
    image_crop = image_trans[:, np.int((image_trans.shape[1]-224)/2):image_trans.shape[1]-np.int((image_trans.shape[1]-224)/2),
                             16:240]
    image_batch = np.copy(image_crop)
    #Normalization
    for i in range(3):
        image_batch[i,:,:] = (image_crop[i,:,:]-mean[i])/(std[i])
    #Imshow
    if show:
        fig=pylab.figure()
        pylab.imshow(np.uint8(image_RGB))      
    if C == True:
        return image_crop
    else:
        return image_batch

def imshow(image, title=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = image*np.mean(std)+np.mean(mean)
    image = np.clip(image*255, 0, 255)
    fig=pylab.figure()
    pylab.imshow(np.uint8(image), cmap=pylab.cm.Greys_r)
    if title is not None:
        pylab.title(title)

def image2dat(image, dir_path):
    f = open(dir_path, "w")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                f.write(str(image[i][j][k])+" ")
    f.close()
    
def weights2dat(kernels, bias, merge_batch, dir_path, CNN):
    f = open(dir_path, "w")
    if CNN == "VGG":
        #KERNELS
        for i in range(16):
            for j in range(kernels[i].shape[0]):
                for k in range(kernels[i].shape[1]):
                    #CONVs
                    if i<13:
                        for l in range(kernels[i].shape[2]):
                            for m in range(kernels[i].shape[3]):
                                f.write(str(kernels[i][j][k][l][m])+" ")
                    #FCs
                    else:
                        f.write(str(kernels[i][j][k])+" ")
        #BIAS
        for i in range(16):
            #CONVs y FCs
            for j in range(bias[i].shape[0]):
                f.write(str(bias[i][j])+" ")
                            
#    elif CNN == "SqueezeNet":
    elif CNN == "MobileNet":
        #KERNELS
        #CONVs
        if merge_batch == False:
            for i in range(19):
                #CONVBNReLU 0 y 18 (first and last CONVs)
                if i == 0 or i == 18:
                    for j in range(kernels[i][0].shape[0]):
                        for k in range(kernels[i][0].shape[1]):
                            for l in range(kernels[i][0].shape[2]):
                                for m in range(kernels[i][0].shape[3]):
                                   f.write(str(kernels[i][0][j][k][l][m])+" ") 
                #InvertedResidual 1 (second group)
                elif i == 1:
                    #CONVBNRelu 0 (internal)
                    for j in range(kernels[i][0][0].shape[0]):
                        for k in range(kernels[i][0][0].shape[1]):
                            for l in range(kernels[i][0][0].shape[2]):
                                for m in range(kernels[i][0][0].shape[3]):
                                   f.write(str(kernels[i][0][0][j][k][l][m])+" ")
                    #CONV 1 (internal)
                    for j in range(kernels[i][1].shape[0]):
                        for k in range(kernels[i][1].shape[1]):
                            for l in range(kernels[i][1].shape[2]):
                                for m in range(kernels[i][1].shape[3]):
                                   f.write(str(kernels[i][1][j][k][l][m])+" ")
                #InvertedResidual (all)
                else:
                    #CONVBNRelu 0 (internal)
                    for j in range(kernels[i][0][0].shape[0]):
                        for k in range(kernels[i][0][0].shape[1]):
                            for l in range(kernels[i][0][0].shape[2]):
                                for m in range(kernels[i][0][0].shape[3]):
                                   f.write(str(kernels[i][0][0][j][k][l][m])+" ")
                    #CONVBNRelu 1 (internal)
                    for j in range(kernels[i][1][0].shape[0]):
                        for k in range(kernels[i][1][0].shape[1]):
                            for l in range(kernels[i][1][0].shape[2]):
                                for m in range(kernels[i][1][0].shape[3]):
                                   f.write(str(kernels[i][1][0][j][k][l][m])+" ")
                    #CONV 2 (internal)
                    for j in range(kernels[i][2].shape[0]):
                        for k in range(kernels[i][2].shape[1]):
                            for l in range(kernels[i][2].shape[2]):
                                for m in range(kernels[i][2].shape[3]):
                                   f.write(str(kernels[i][2][j][k][l][m])+" ")
            #BATCHs
            for i in range(19):
                #CONVBNReLU 0 y 18 (first and last CONVs)
                if i == 0 or i == 18:
                    for j in range(kernels[i][1][0].shape[0]):
                        f.write(str(kernels[i][1][0][j])+" ") #Mean
                        f.write(str(bias[i][1][0][j])+" ")    #Var
                        f.write(str(kernels[i][1][1][j])+" ") #Gamma
                        f.write(str(bias[i][1][1][j])+" ")    #Beta
                #InvertedResidual 1 (second group)
                elif i == 1:
                    #CONVBNRelu 0 (internal)
                    for j in range(kernels[i][0][1][0].shape[0]):
                        f.write(str(kernels[i][0][1][0][j])+" ") #Mean
                        f.write(str(bias[i][0][1][0][j])+" ")    #Var
                        f.write(str(kernels[i][0][1][1][j])+" ") #Gamma
                        f.write(str(bias[i][0][1][1][j])+" ")    #Beta
                    #BATCH 1 (internal)
                    for j in range(kernels[i][2][0].shape[0]):
                        f.write(str(kernels[i][2][0][j])+" ") #Mean
                        f.write(str(bias[i][2][0][j])+" ")    #Var
                        f.write(str(kernels[i][2][1][j])+" ") #Gamma
                        f.write(str(bias[i][2][1][j])+" ")    #Beta
                #InvertedResidual (all)
                else:
                    #CONVBNRelu 0 (internal)
                    for j in range(kernels[i][0][1][0].shape[0]):
                        f.write(str(kernels[i][0][1][0][j])+" ") #Mean
                        f.write(str(bias[i][0][1][0][j])+" ")    #Var
                        f.write(str(kernels[i][0][1][1][j])+" ") #Gamma
                        f.write(str(bias[i][0][1][1][j])+" ")    #Beta
                    #CONVBNRelu 1 (internal)
                    for j in range(kernels[i][1][1][0].shape[0]):
                        f.write(str(kernels[i][1][1][0][j])+" ") #Mean
                        f.write(str(bias[i][1][1][0][j])+" ")    #Var
                        f.write(str(kernels[i][1][1][1][j])+" ") #Gamma
                        f.write(str(bias[i][1][1][1][j])+" ")    #Beta
                    #BATCH 2 (internal)
                    for j in range(kernels[i][3][0].shape[0]):
                        f.write(str(kernels[i][3][0][j])+" ") #Mean
                        f.write(str(bias[i][3][0][j])+" ")    #Var
                        f.write(str(kernels[i][3][1][j])+" ") #Gamma
                        f.write(str(bias[i][3][1][j])+" ")    #Beta
        else:
            for i in range(19):
                #CONVBNReLU 0 y 18 (first and last CONVs)
                if i == 0 or i == 18:
                    for j in range(kernels[i].shape[0]):
                        for k in range(kernels[i].shape[1]):
                            for l in range(kernels[i].shape[2]):
                                for m in range(kernels[i].shape[3]):
                                   f.write(str(kernels[i][j][k][l][m])+" ") 
                #InvertedResidual 1 (second group)
                elif i == 1:
                    #CONVBNRelu 0 (internal)
                    for j in range(kernels[i][0].shape[0]):
                        for k in range(kernels[i][0].shape[1]):
                            for l in range(kernels[i][0].shape[2]):
                                for m in range(kernels[i][0].shape[3]):
                                   f.write(str(kernels[i][0][j][k][l][m])+" ")
                    #CONV 1 (internal)
                    for j in range(kernels[i][1].shape[0]):
                        for k in range(kernels[i][1].shape[1]):
                            for l in range(kernels[i][1].shape[2]):
                                for m in range(kernels[i][1].shape[3]):
                                   f.write(str(kernels[i][1][j][k][l][m])+" ")
                #InvertedResidual (all)
                else:
                    #CONVBNRelu 0 (internal)
                    for j in range(kernels[i][0].shape[0]):
                        for k in range(kernels[i][0].shape[1]):
                            for l in range(kernels[i][0].shape[2]):
                                for m in range(kernels[i][0].shape[3]):
                                   f.write(str(kernels[i][0][j][k][l][m])+" ")
                    #CONVBNRelu 1 (internal)
                    for j in range(kernels[i][1].shape[0]):
                        for k in range(kernels[i][1].shape[1]):
                            for l in range(kernels[i][1].shape[2]):
                                for m in range(kernels[i][1].shape[3]):
                                   f.write(str(kernels[i][1][j][k][l][m])+" ")
                    #CONV 2 (internal)
                    for j in range(kernels[i][2].shape[0]):
                        for k in range(kernels[i][2].shape[1]):
                            for l in range(kernels[i][2].shape[2]):
                                for m in range(kernels[i][2].shape[3]):
                                   f.write(str(kernels[i][2][j][k][l][m])+" ")
            #Bias CONV
            for i in range(19):
                #CONVBNReLU 0 y 18 (first and last CONVs)
                if i == 0 or i == 18:
                    for j in range(bias[i].shape[0]):
                        f.write(str(bias[i][j])+" ") 
                #InvertedResidual 1 (second group)
                elif i == 1:
                    #CONVBNRelu 0 (internal)
                    for j in range(bias[i][0].shape[0]):
                        f.write(str(bias[i][0][j])+" ")
                    #CONV 1 (internal)
                    for j in range(bias[i][1].shape[0]):
                        f.write(str(bias[i][1][j])+" ")
                #InvertedResidual (all)
                else:
                    #CONVBNRelu 0 (internal)
                    for j in range(bias[i][0].shape[0]):
                        f.write(str(bias[i][0][j])+" ")
                    #CONVBNRelu 1 (internal)
                    for j in range(bias[i][1].shape[0]):
                        f.write(str(bias[i][1][j])+" ")
                    #CONV 2 (internal)
                    for j in range(bias[i][2].shape[0]):
                        f.write(str(bias[i][2][j])+" ")
        #FCs
        #kernels
        for i in range(kernels[19].shape[0]):
            for j in range(kernels[19].shape[1]):
                f.write(str(kernels[19][i][j])+" ")
        #bias
        for i in range(kernels[19].shape[0]):
            f.write(str(bias[19][i])+" ")

def weights_no_zeros2dat(weights_CONV, weights_FC, bias, dir_path, hardware = False):
    f = open(dir_path, "w")
    #MOBILENET
    #KERNELS CONVs
    for i in range(len(weights_CONV)):
        f.write(str(weights_CONV[i])+" ") 
    #Bias CONV
    for i in range(19):
        #CONVBNReLU 0 y 18 (first and last CONVs)
        if i == 0 or i == 18:
            for j in range(bias[i].shape[0]):
                f.write(str(bias[i][j])+" ") 
        #InvertedResidual 1 (second group)
        elif i == 1:
            #CONVBNRelu 0 (internal)
            for j in range(bias[i][0].shape[0]):
                f.write(str(bias[i][0][j])+" ")
            #CONV 1 (internal)
            for j in range(bias[i][1].shape[0]):
                f.write(str(bias[i][1][j])+" ")
        #InvertedResidual (all)
        else:
            #CONVBNRelu 0 (internal)
            for j in range(bias[i][0].shape[0]):
                f.write(str(bias[i][0][j])+" ")
            #CONVBNRelu 1 (internal)
            for j in range(bias[i][1].shape[0]):
                f.write(str(bias[i][1][j])+" ")
            #CONV 2 (internal)
            for j in range(bias[i][2].shape[0]):
                f.write(str(bias[i][2][j])+" ")
    #FCs
    #kernels
    for i in range(len(weights_FC)):
        f.write(str(weights_FC[i])+" ")
    #bias
    for i in range(bias[19].shape[0]):
        f.write(str(bias[19][i])+" ")
    #FINAL VALUE
    if hardware:
        f.write("f")

def index2dat(index_CONV, index_FC, dir_path, hardware = False):
    f = open(dir_path, "w") 
    #INDEX CONV         
    for i in range(len(index_CONV)):
        f.write(str(index_CONV[i])+" ") 
    #INDEX FC
    for i in range(len(index_FC)):
        f.write(str(index_FC[i])+" ")
    if hardware:
        f.write("f") 

def max_values(weights_CONV, weights_FC, bias, index_CONV, index_FC):
    max_w_conv = np.max(np.abs(weights_CONV))
    max_w_fc = np.max(np.abs(weights_FC))
    max_i_conv = np.max(np.abs(index_CONV))
    max_i_fc = np.max(np.abs(index_FC))
    max_b_conv = 0
    for i in range(19):
        #CONVBNReLU 0 y 18 (first and last CONVs)
        if i == 0 or i == 18:
            if (np.max(np.abs(bias[i])) > max_b_conv):
                max_b_conv = np.max(np.abs(bias[i]))
        #InvertedResidual 1 (second group)
        elif i == 1:
            #CONVBNRelu 0 (internal)
            if (np.max(np.abs(bias[i][0])) > max_b_conv):
                max_b_conv = np.max(np.abs(bias[i][0]))
            #CONV 1 (internal)
            if (np.max(np.abs(bias[i][1])) > max_b_conv):
                max_b_conv = np.max(np.abs(bias[i][1]))
        #InvertedResidual (all)
        else:
            #CONVBNRelu 0 (internal)
            if (np.max(np.abs(bias[i][0])) > max_b_conv):
                max_b_conv = np.max(np.abs(bias[i][0]))
            #CONVBNRelu 1 (internal)
            if (np.max(np.abs(bias[i][1])) > max_b_conv):
                max_b_conv = np.max(np.abs(bias[i][1]))
            #CONV 2 (internal)
            if (np.max(np.abs(bias[i][2])) > max_b_conv):
                max_b_conv = np.max(np.abs(bias[i][2]))
    max_b_fc = np.max(np.abs(bias[19]))
    print("\nMax value weights CONV: ", max_w_conv)
    print("Max value weights FC: ", max_w_fc)
    print("Max value bias CONV: ", max_b_conv)
    print("Max value bias FC: ", max_b_fc)
    print("Max value index CONV: ", max_i_conv)
    print("Max value index FC: ", max_i_fc)
    print("\nBits weights CONV: ", np.ceil(np.log2(max_w_conv)))
    print("Bits weights FC: ", np.ceil(np.log2(max_w_fc)))
    print("Bits bias CONV: ", np.ceil(np.log2(max_b_conv)))
    print("Bits bias FC: ", np.ceil(np.log2(max_b_fc)))
    print("Bits index CONV: ", np.ceil(np.log2(max_i_conv)))
    print("Bits index FC: ", np.ceil(np.log2(max_i_fc)))
                            

#%%CNN

def conv_filter(fea_map, kernel, stride):
    len_x, len_y = fea_map.shape
    len_kernel = kernel.shape[0]
    offset = np.int(len_kernel/2)
    ext = np.pad(fea_map, offset, mode='constant')
    if len_kernel == 1:
        conv = ext*kernel
    else:
        conv = cv2.filter2D(ext, -1, kernel, anchor=(1,1))
    conv_real = conv[offset:len_x+offset, offset:len_y+offset]
    return conv_real[::stride, ::stride]

def conv_layer(in_map, kernels, bias, stride, map_size, out_size, groups=0):
    out_map = np.zeros([out_size, np.int(in_map.shape[1]/stride), np.int(in_map.shape[2]/stride)])
    for i in range(out_size):
        if groups == 0:
            for j in range(map_size):
                out_map[i,:,:] = out_map[i,:,:] + conv_filter(in_map[j], kernels[i,j,:,:], stride)
        else:
            out_map[i,:,:] = conv_filter(in_map[i], kernels[i,0,:,:], stride)
        if type(bias)!=bool:
            out_map[i,:,:] = out_map[i,:,:] + bias[i]
    return out_map

def BatchNorm(in_map, map_size, mean, std):
    eps = 0.00001
    out_map = np.zeros([map_size, in_map.shape[1], in_map.shape[2]])
    for i in range(map_size):
        out_map[i,:,:] = (in_map[i,:,:] - mean[0][i])/(np.sqrt(std[0][i] + eps))*mean[1][i] + std[1][i]
    return out_map

def ReLU(in_map, activation):
    mask = in_map>=activation
    out_map = in_map*mask
    return out_map

def ReLU6(in_map, activation):
    mask_max = in_map>=activation
    mask_min = in_map>=6
    mask = mask_max & ~mask_min
    out_map = in_map*mask + mask_min*6
    return out_map

def ReLU6_quant(in_map, min_value, max_value):
    mask_max = in_map>=min_value
    mask_min = in_map>=max_value
    mask = mask_max & ~mask_min
    out_map = in_map*mask + mask_min*max_value
    return out_map

def pool(in_map, kernel_size, stride, padding, pool_mode='max'):
    in_map = np.pad(in_map, padding, mode='constant')
    output_shape = ((in_map.shape[0] - kernel_size)//stride + 1,
                    (in_map.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    new_map = as_strided(in_map, shape = output_shape + kernel_size, 
                        strides = (stride*in_map.strides[0],
                                   stride*in_map.strides[1]) + in_map.strides)
    new_map = new_map.reshape(-1, *kernel_size)
    if pool_mode == 'max':
        return new_map.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return new_map.mean(axis=(1,2)).reshape(output_shape)
    
def max_pool_layer(in_map, kernel_size, stride, padding, mode='max'):
    out_map = np.zeros([in_map.shape[0], int((in_map.shape[1]-kernel_size)/stride+1), int((in_map.shape[2]-kernel_size)/stride+1)])
    for i in range(in_map.shape[0]):
        out_map[i,:,:] = pool(in_map[i], kernel_size, stride, padding, mode)
    return out_map

def flatten(in_map, out_size):
    return in_map.reshape(out_size)

def Global_Avg(in_map, map_size):
    out_vector = np.zeros([map_size,])
    for i in range(map_size):
        out_vector[i] = np.mean(in_map[i,:,:])
    return out_vector

def fully_connected_layer(in_map, kernel, bias, keras = False):
    if keras == False:
        return np.dot(kernel, in_map) + bias
    else:
        return np.dot(in_map, kernel) + bias

def softmax(in_map, number_class):
    max_value = np.max(in_map)
    exponential = np.exp(in_map-np.ones([number_class]) * max_value)
    sum_exp = np.sum(exponential)
    return exponential/sum_exp
    
#%%Testing
#
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
#
#%% Weights
#
##VGG_16
#VGG_16 = models.vgg16(pretrained=True).to(device)
#weights, bias = read_weights(VGG_16, "VGG16")
#
##SqueezeNet
#SqueezeNet = models.squeezenet1_0(pretrained=True).to(device)
#weights, bias = read_weights(SqueezeNet, "SqueezeNet")
#
#MobileNet V2
#MobileNet = models.mobilenet_v2(pretrained=True).to(device)
#weights, bias = read_weights(MobileNet, "MobileNet")
#
#%%load images
#
#number_images = 1
#
##PC UNIVERSIDAD
#dir_path= r'''E:\nacho\Universidad\Magister\Imagenes'''
#if (os.path.exists(dir_path)==False):
#    #PC CASA
#    dir_path= r'''D:\nacho\Universidad\Magister\Imagenes'''
#dir_image=r'''\ImageNet\Validation\images_test'''
#name_labels=r'''\ImageNet\val.txt'''
#
#f = open (dir_path+name_labels,'r')
#labels = f.read()
#labels = labels.split('\n')
#name_image=[]
#label_image=[]
#for i in range(len(labels)-1):
#    temp=labels[i].split()
#    name_image.append(temp[0])
#    label_image.append(temp[1])
#
#f.close()
#
#index_label=[]
#for i in range(number_images):
#    temp=randint(0,len(name_image)-1)
#    if not temp in index_label:
#        index_label.append(temp)
#
#for k in range(number_images):
#    path_image = dir_path+dir_image+"\\"+name_image[index_label[k]]
#    path_image=dir_path+dir_image+"\\ILSVRC2012_val_00000293.JPEG"
#    image_out = load_image(path_image, True)
#
#%%Load class
#
#path_class = "clases_imagenet.txt"
#dic_class = read_class(path_class)

    




