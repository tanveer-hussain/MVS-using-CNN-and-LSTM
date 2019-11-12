#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:08:22 2018

@author: imlab
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:22:19 2018

@author: imlab
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:55:09 2017

@author: AMIN
"""

import tensorflow as tf
import os, numpy as np


import cv2
from Tkinter import Tk
from tkFileDialog import askopenfilename


import caffe
import tempfile
from math import ceil

import time


text_file = open("Trained Model/ClassNames.txt", "r")
ClassNames =text_file.readlines()


Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename(initialdir='/media/imlab/IMLab Server Data/Ubuntu/TanveerHussain/MVS') # show an "Open" dialog box and return the path to the selected
patth = filename.split('/')
VideoName = patth[len(patth)-1]

vidcap = cv2.VideoCapture(filename)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('OutPutSummary_18_Bl-7F.avi',fourcc, 30.0, (320,240))


font                   = cv2.FONT_HERSHEY_TRIPLEX
bottomLeftCornerOfText = (20,20)
fontScale              = 0.3
fontColor              = (255,255,255)
lineType               = 1

#vidcap = cv2.VideoCapture('/media/imlab/IMLab Server Data/Datasets/KTH Dataset/running/person01_running_d1_uncomp.avi')

proto = '/media/imlab/IMLab Server Data/Ubuntu/TanveerHussain/MVS/Models/deploy.prototxt'
model = '/media/imlab/IMLab Server Data/Ubuntu/TanveerHussain/MVS/Models/bvlc_alexnet.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(proto, model, caffe.TEST)
img_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)


videoFeatures=[]




n_classes = 2
chunk_size =1000
n_chunks =15
rnn_size = 256
 
 


x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')


def recurrent_neural_network(x):
    
    
  
    W = {
            'hidden': tf.Variable(tf.random_normal([chunk_size, rnn_size])),
            'output': tf.Variable(tf.random_normal([rnn_size, n_classes]))
        }
    biases = {
            'hidden': tf.Variable(tf.random_normal([rnn_size], mean=1.0)),
            'output': tf.Variable(tf.random_normal([n_classes]))
        }


    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1,chunk_size])
    x = tf.nn.relu(tf.matmul(x, W['hidden']) + biases['hidden'])
    x = tf.split (x,n_chunks, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, final_states = tf.contrib.rnn.static_rnn(lstm_cells, x, dtype=tf.float32)
    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
#    lstm_last_output=tf.transpose(outputs, [1,0,2])
    # Linear activation
    return tf.matmul(outputs[-1], W['output']) + biases['output']
    
#####################################################################  
    


def train_recurrnet_neural_network(x):

    prediction= recurrent_neural_network(x)
#    tf.device('/cpu:0')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, "Trained Model/model.chk")
        #print(sess.run(tf.all_variables()))
        videolength = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        videoFeatures=[]
        frame_no=-1;
        startTime = time.time()
        
        while (frame_no < videolength-1):  #(videolength%30)
            
            frame_no = frame_no + 1
            vidcap.set(1,frame_no)
            ret0,img0 = vidcap.read()
            
            if(ret0 == 1):
                resized_image = caffe.io.resize_image(img0,[227,227])
                transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
                transformer.set_transpose('data',(2, 0, 1))
                transformer.set_channel_swap('data', (2, 1, 0))
                transformer.set_raw_scale('data', 255)
                transformer.set_mean('data',img_mean)
                net.blobs['data'].reshape(1, 3, 227, 227)
                net.blobs['data'].data[...] = transformer.preprocess('data', resized_image)
                net.forward()
                features = net.blobs['fc8'].data[0].reshape(1,1000)
                bb = np.matrix(features)
                features = bb.max(0)
                videoFeatures.append(features)
                print(frame_no % 15)
                if frame_no % 15 == 14:
                    X_test = np.asarray(videoFeatures)
                    labled =sess.run(prediction, feed_dict={x: X_test.reshape((-1,n_chunks, chunk_size))})
                    label = labled.argmax(axis=1)
                    Confidence = labled[0,label[0]]
#                    print(label[0],'    ', Confidence)
                    print('Writing resutls....')
                    i = frame_no-14
                    for kk  in range(15):
                        vidcap.set(1,i)
                        ret1,img1 = vidcap.read()
                        if(ret1 == 1):
                            var = 'Confidence =   ' + str(Confidence) +  '   Category : '+ ClassNames[label[0]].replace('\n',' ')
                            cv2.putText(img1,  var ,  bottomLeftCornerOfText,  font,  fontScale, fontColor, lineType) #+ '     Confidence: ' + str(Confidence)
                            out.write(img1)
#                            cv2.imshow('Resutls',img1)
                            pathh = 'test/image_'+ str(i) + '.jpg'
                            cv2.imwrite(pathh,img1)
#                            cv2.waitKey(10)
                            i=i+1
                    videoFeatures=[]
            
        endTime = time.time()
        print ('total time taken', endTime - startTime)
#
train_recurrnet_neural_network(x)

out.release()
#cap = cv2.VideoCapture(filename,0)
#while True:
#    ret,img= cap.read()
#    if (ret == False):
#        break
#    else:
#        cv2.imshow(ClassNames[indexOfClass],img)
#        cv2.waitKey(50)


