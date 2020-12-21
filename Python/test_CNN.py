from torchvision import models
import torch
import functions
import functions_test_keras as functions_keras
from random import randint
import os
import model_vgg
import model_mobilenet
import numpy as np

CNN_models = ["VGG_16", "SqueezeNet", "MobileNet"]
CNN = CNN_models[2]
show_image = True
test = True
layer = 0
keras = False
prune_semistructured = True
merge_batch = True
quant = False #quant works only with merge active
quant_int = False  #quant_int true if quant is false
type_quant = "linear"
fixed = [12, 6]   #conv, fc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%Weights

#Prune path 
#PC UNIVERSIDAD
path_mobilenet_prune_semistructured = r'''PATH//mobilenet.pth'''

if CNN == "VGG_16":
    if keras == False:
        VGG_16 = models.vgg16(pretrained=True).to(device)
        kernels, bias = functions.read_weights(VGG_16, "VGG16")
    else:
        #PC UNIVERSIDAD
        file_weights_keras = r'''E:\nacho\Universidad\Magister\Kernels redes\VGG16\vgg16_weights_th_dim_ordering_th_kernels.h5'''
        if (os.path.exists(file_weights_keras)==False):
            #PC CASA
            file_weights_keras = r'''D:\nacho\Universidad\Magister\Kernels redes\VGG16\vgg16_weights_th_dim_ordering_th_kernels.h5'''
        kernels, bias = functions_keras.read_weights(file_weights_keras)
elif CNN == "SqueezeNet":
    SqueezeNet = models.squeezenet1_0(pretrained=True).to(device)
    kernels, bias = functions.read_weights(SqueezeNet, "SqueezeNet")
elif CNN == "MobileNet":
    if prune_semistructured:
        if torch.cuda.is_available():
            MobileNet = torch.load(path_mobilenet_prune_semistructured).to(device)
        else:
            MobileNet = torch.load(path_mobilenet_prune_semistructured, map_location="cpu").to(device)
    else:
        MobileNet = models.mobilenet_v2(pretrained=True).to(device)
    kernels, bias = functions.read_weights(MobileNet, "MobileNet")
    if merge_batch:
        kernels, bias = functions.merge_conv_batch(kernels, bias)
    if quant or quant_int:
        kernels, bias = functions.quant_weights(kernels, bias, fixed[0], fixed[1], type_quant, False)
    
#%%Load class
    
path_class = "imagenet_class.txt"
dic_class = functions.read_class(path_class)

#%%Load image

#PC UNIVERSIDAD
dir_path_images= r'''E:\nacho\Universidad\Magister\Imagenes'''
if (os.path.exists(dir_path_images)==False):
    #PC CASA
    dir_path_images= r'''D:\nacho\Universidad\Magister\Imagenes'''
dir_image=r'''\ImageNet\Validation\images_test'''
name_labels=r'''\ImageNet\val.txt'''

f = open (dir_path_images+name_labels,'r')
labels = f.read()
labels = labels.split('\n')
name_image=[]
label_image=[]
for i in range(len(labels)-1):
    temp=labels[i].split()
    name_image.append(temp[0])
    label_image.append(temp[1])
f.close()

random_images=randint(0,len(name_image)-1)

if test == False:
    path_image = dir_path_images+dir_image+"\\"+name_image[random_images]
else:
    path_image = dir_path_images+dir_image+"\\ILSVRC2012_val_00000067.JPEG"

if keras == False:
    image = functions.load_image(path_image, show_image, test)
else:
    image = functions_keras.load_image(path_image, show_image)

#%%Proccess

if CNN == "VGG_16":
    result = model_vgg.VGG_16(image, kernels, bias, layer, keras)
elif CNN == "MobileNet":
    result = model_mobilenet.MobileNet_V2(image, kernels, bias, merge_batch, layer, quant, quant_int)

if (layer <= 12 and CNN =="VGG_16") or (layer <= 18 and CNN =="MobileNet"):
    if keras == False:     
        functions.imshow(result[0])
    else:
        functions_keras.imshow(result[0])

#%%Results

result_save = np.copy(result)

#Real
if (layer > 15 and CNN =="VGG_16") or (layer > 20 and CNN =="MobileNet"):
    if test == False:
        correct_label = label_image[random_images]
    else:
        correct_label = label_image[66]
    print("\nCorrect label: " + str(dic_class.get(int(correct_label))))
    
    #Out
    print("\nResults: \n")
    
    for i in range(5):
        max_arg_result = np.argmax(result_save)
        print(dic_class.get(max_arg_result) + ':   ' + str(round(np.max(result_save)*100,2)) + '%')
        result_save[max_arg_result] = 0















