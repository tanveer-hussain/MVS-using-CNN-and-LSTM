

from __future__ import print_function

import os, sys, numpy as np
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil
import cv2
import scipy.io as sio
from matplotlib import pyplot as plt


DatasetFolder = '/media/imlab/IMLab Server Data/Ubuntu/TanveerHussain/MVS/Training'


proto = '/media/imlab/IMLab Server Data/Ubuntu/TanveerHussain/MVS/Models/alexnet.prototxt'
model = '/media/imlab/IMLab Server Data/Ubuntu/TanveerHussain/MVS/Models/alexnet-model.caffemodel'
caffe.set_mode_cpu()
net = caffe.Net(proto, model, caffe.TEST)
img_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)


DatabaseFeautres = []
DatabaseLabel = []

for folderName in os.listdir(DatasetFolder):
    print(folderName)
    subFolder = DatasetFolder+'/'+ folderName
    for filename in os.listdir(subFolder):
        vidcap = cv2.VideoCapture(DatasetFolder+'/'+ folderName +'/'+filename)
        print('Feature Extraction of : ',filename)
        videolength = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        videoFeatures=[]
        frame_no=-1;
        
        while (frame_no < videolength-1):  #(videolength%30)
            
            frame_no = frame_no + 1
            vidcap.set(1,frame_no)
            ret0,img0 = vidcap.read()
            
            if(ret0 == 1):
                resized_image = caffe.io.resize_image(img0,[224,224])
                transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
                transformer.set_transpose('data',(2, 0, 1))
                transformer.set_channel_swap('data', (2, 1, 0))
                transformer.set_raw_scale('data', 255)
                transformer.set_mean('data',img_mean)
                net.blobs['data'].reshape(1, 3, 224, 224)
                net.blobs['data'].data[...] = transformer.preprocess('data', resized_image)
                net.forward()
                features = net.blobs['fc1000'].data[0].reshape(1,1000)
                bb = np.matrix(features)
                features = bb.max(0)
                videoFeatures.append(features)
                print(frame_no % 15)
                if frame_no % 15 == 14:
                    aa = np.asarray(videoFeatures)
                    DatabaseFeautres.append(aa)
                    DatabaseLabel.append(folderName)
                    videoFeatures=[]

#np.save('DatabaseFeaturesList',DatabaseFeautres)
#np.save('DatabaseLabelList',DatabaseLabel)

##################### One Hot and Train Test spilt
TotalFeatures= []
for sample in DatabaseFeautres:
    TotalFeatures.append(sample.reshape([1,15000]))
    
    
TotalFeatures = np.asarray(TotalFeatures)
TotalFeatures = TotalFeatures.reshape([len(DatabaseFeautres),15000])


OneHotArray = []
kk=1;
for i in range(len(DatabaseFeautres)-1):
    OneHotArray.append(kk)
    if (DatabaseLabel[i] != DatabaseLabel[i+1]):
        kk=kk+1;


OneHot=  np.zeros([len(DatabaseFeautres),2], dtype='int');


for i in range(len(DatabaseFeautres)-1):
    print(i)
    OneHot[i,OneHotArray[i]-1] = 1



np.save('MVS_TotalFeatures',TotalFeatures)
sio.savemat('MVS_Labels.mat', mdict={'DatabaseLabel': OneHot})
sio.savemat('MVS_TotalFeatures.mat', mdict={'TotalFeatures': TotalFeatures},appendmat=True, format='5',
    long_field_names=False, do_compression=True, oned_as='row')

#import random
#list=[]
#for i in range(1500):
#      r=random.randint(1,7999)
#      if r not in list: list.append(r)

#
#
#import os, sys, numpy as np
#
#
#
#DatasetFolder = '/media/imlab/IMLab Server Data/Datasets/UCF101/UCF-101'
#
#
#
#for folderName in os.listdir(DatasetFolder):
#    print(folderName)
#       




