import serial
import sys
import numpy as np
import cv2
from serial.tools import list_ports
from random import randint
import os
import functions
from matplotlib import pylab

show = True
test = True
image_quant = 7
MDK = False

#%%Load class
    
path_class = "imagenet_class.txt"
dic_class = functions.read_class(path_class)

#%% image charge

#Load path

#PC UNIVERSIDAD
dir_path_images= r'''PATH_IMAGENET'''
dir_image = r'''PATH_IMAGES_IMAGENET'''
name_labels=r'''PATH_LABELS_IMAGENET'''

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
    path_image = dir_path_images+dir_image+"\\ILSVRC2012_val_00000067.JPEG"  #IMAGE FOR TEST
    
#Load images    

image = functions.load_image(path_image, show, test)
image = image * (2**image_quant)
image = image.astype(int)
if MDK:
    image = image.transpose((1, 2, 0))

#%% Divide pixel image

image_to_send = []   #Max number is 2.7 times 127

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        for k in range(image.shape[2]):
            if image[i,j,k] < 0:
                neg_value = -1
            else:
                neg_value = 1
            if np.abs(image[i,j,k]) > (2**7 - 1)*(2):
                image_to_send.append(np.uint8(image[i,j,k] - (2**7 - 1)*(2)*neg_value))
                image_to_send.append(np.uint8(127*neg_value))
                image_to_send.append(np.uint8(127*neg_value))
            elif np.abs(image[i,j,k]) > (2**7 - 1):
                image_to_send.append(np.uint8(0))
                image_to_send.append(np.uint8(image[i,j,k] - (2**7 - 1)*neg_value))
                image_to_send.append(np.uint8(127*neg_value))
            else:
                image_to_send.append(np.uint8(0))
                image_to_send.append(np.uint8(0))
                image_to_send.append(np.uint8(image[i,j,k]))
                
#%% UART PORST

B = list(list_ports.comports())

i=0
PUERTOS=[]

if (len(B)>1):
    print ('There is more than one COM port available','(',len(B),').')    
elif (len(B)==0):
    print ('No COM ports found!')
    sys.exit()

for Tuple in B:
    PUERTOS.append(Tuple[0])
    if (len(B)>1):
        print ('To use port ',Tuple[0],' enter' ,i)
    i = i+1


if (len(B)>1):
    PUERTO = input('Enter number: ')
    try:
        COM = PUERTOS[int(PUERTO)]
        print ('You have selected the port ', COM)
    except:
        print ('Port no valid!')
        sys.exit()
else:
    COM = PUERTOS[0]

try:
    RS232 = serial.Serial(COM, 115200, timeout=1, stopbits=1)
except:
    i=0
    
RS232.close()


try:
    RS232.open()
except:
    if (RS232.isOpen()):
        print ('COM port open')
    else:
        print ('Error opening COM port!')
        
RS232.flushInput()    

#%% UART SEND

print("Sending input image")
            
RS232.write(bytearray(image_to_send))

print("Sent input image")

#%%CONTROL

#read weights
while True:
    msjeArduino = RS232.read(1)
    if msjeArduino != b'z':
        if msjeArduino == b'a':
            print("Reading CONV weights ...")
        elif msjeArduino == b'b':
            print("Reading FC weights ...")
    else:
        break
        
#Process
while True:
    msjeArduino = RS232.read(1)
    if msjeArduino != b'z':
        if msjeArduino == b'a':
            print("\nCONVBNReLU 0 ...")
        elif msjeArduino == b'b':
            print("InvertedResidual 1 ...");
        elif msjeArduino == b'c':
            print("InvertedResidual 2 ...");
        elif msjeArduino == b'd':
            print("InvertedResidual 3 ...");
        elif msjeArduino == b'e':
            print("InvertedResidual 4 ...");
        elif msjeArduino == b'f':
            print("InvertedResidual 5 ...");
        elif msjeArduino == b'g':
            print("InvertedResidual 6 ...");
        elif msjeArduino == b'h':
            print("InvertedResidual 7 ...");
        elif msjeArduino == b'i':
            print("InvertedResidual 8 ...");
        elif msjeArduino == b'j':
            print("InvertedResidual 9 ...");
        elif msjeArduino == b'k':
            print("InvertedResidual 10 ...");
        elif msjeArduino == b'l':
            print("InvertedResidual 11 ...");
        elif msjeArduino == b'm':
            print("InvertedResidual 12 ...");
        elif msjeArduino == b'n':
            print("InvertedResidual 13 ...");
        elif msjeArduino == b'o':
            print("InvertedResidual 14 ...");
        elif msjeArduino == b'p':
            print("InvertedResidual 15 ...");
        elif msjeArduino == b'q':
            print("InvertedResidual 16 ...");
        elif msjeArduino == b'r':
            print("InvertedResidual 17 ...");
        elif msjeArduino == b's':
            print("CONVBNReLU 18 ...")
        elif msjeArduino == b't':
            print("Global AVG 19 ...");
        elif msjeArduino == b'u':
            print("FC 20 ...");
        elif msjeArduino == b'v':
            print("Softmax 21 ...");
    else:
        break
    
#%% Time received

time = 0
frac_part = False;
num_frac_part = 1       
        
while True:
    try:
        msjeArduino = RS232.read(1) 
        if msjeArduino == b'z':
            print("Time processing: " + str(time) + " seconds")
            if time > 0:
                print("Frame per second: " + str(1/time) + " fps")
            break
        if msjeArduino != b'' and msjeArduino != b' ': 
            if msjeArduino == b'.':
                frac_part = True
                num_frac_part = 1;
            else:
                if frac_part == False:
                    time = time*10 + int(msjeArduino)
                else:
                    if msjeArduino == b':':
                        print("Transfer ERROR in TIME")
                        msjeArduino = b'0'
                    time = time + float(msjeArduino)/(10**num_frac_part)
                    num_frac_part = num_frac_part = num_frac_part + 1
    except KeyboardInterrupt:  
        print('Loop interrumpido')
        break

#%% UART Received

limits = [112, 112, 32]

vector_receive = [] 
error = []
first_data = True
neg_value = 1;
frac_part = False;
data = 0
counter = 0  
num_frac_part = 1  
while True:
    try:
        msjeArduino = RS232.read(1) 
        if msjeArduino == b'z':
            break
        if msjeArduino != b'': 
            if msjeArduino == b'-':
                neg_value = -1
            elif msjeArduino == b' ':
                if first_data:
                    print("Receiving feature map")
                    first_data = False
                else:
                    counter = counter + 1
                    vector_receive.append(data) 
                    data = 0
                    neg_value = 1
                    frac_part = False
            elif msjeArduino == b'.':
                frac_part = True
                num_frac_part = 1;
            else:
                if msjeArduino == b':':
                    msjeArduino = b'0'
                if frac_part == False:
                    data = data*10 + int(msjeArduino)*neg_value
                else:
                    data = data + float(msjeArduino)*neg_value/(10**num_frac_part)
                    num_frac_part = num_frac_part + 1
    except KeyboardInterrupt:  
        print('Loop interrumpido')
        break
    
RS232.close()

# image_receive = np.array(vector_receive).reshape(limits[2], limits[0], limits[1])   #image of one channel (DESCOMENT TO SEE DATA)
            
#%%Results

result_save = np.copy(vector_receive)

#Real
if test == False:
    correct_label = label_image[random_images]
else:
    correct_label = label_image[66]
print("\nCorrect label: " + str(dic_class.get(int(correct_label))))

#Out
print("\nResults: \n")

for i in range(5):
    max_arg_result = np.argmax(result_save)
    print(dic_class.get(max_arg_result) + ':   ' + str(round(np.max(result_save),2)) + '%')
    result_save[max_arg_result] = 0
            
                
    

