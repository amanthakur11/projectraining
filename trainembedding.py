#import cv2
import numpy as np
import os
from random import shuffle
import pandas as pd
import untitled0 as ut
#train_dir='/home/amanthakur/Downloads/train'
#test_dir='/home/amanthakur/Downloads/test1'
img_size=7
lr=0.001

#pixel1,kern=main()
kern=7
dataset=pd.read_csv("training.file.csv").values

model_name="dogsvscats-()-()-model".format(lr,"6conv-basic")

def label_image(img):
    
    word_label=img[-1]
    #word_label=word_label[0][:3]
    print(word_label)
    if word_label=='ROI': return [1,0]
    elif word_label=='NROI': return [0,1]


def create_train_data(kern):
    training_data=[]
    for img in dataset[:-13]:
        label=label_image(img)
        print(label)
        a=[]
        for i in range(0,len(img)-1,kern):
            #b=[]
            #b.append(img[i:i+7])
            a.append(img[i:i+7])
        #path=os.path.join(train_dir,img)
        a=np.array(a)
        print(a)
        #print(path)
        #im=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        #img=cv2.resize(im,(img_size,img_size))
        #training_data.append([np.array(img),np.array(label)])
        training_data.append([a,np.array(label)])
    shuffle(training_data)
    np.save("training_data.npy",training_data)
    return training_data


def process_test_data(kern):
    testing_data=[]
    for img in dataset[-12:]:
        a=[]
        for i in range(0,len(img)-1,kern):
            #b=[]
            #b.append()
            a.append(img[i:i+7])
        #path=os.path.join(train_dir,img)
        a=np.array(a)
        #img_num=
        print(a)
        #path=os.path.join(test_dir,img)
        #img_num=img.split(',')[0]
        #img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size))
        
        testing_data.append(a)
    np.save("test_data.npy",testing_data)
    return testing_data

train_data=create_train_data(kern)
#if you already have train data
#train_data=np.load("training_data.npy")

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, img_size,img_size, 1], name='input')
#print(convnet)
convnet = conv_2d(convnet, 16, 2, activation='relu')
#print(convnet)
convnet = max_pool_2d(convnet, 2)
#print(convnet)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 16, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 16, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
#print(convnet)
convnet = fully_connected(convnet, 512, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet,tensorboard_dir='log')
#model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
   # snapshot_step=500, show_metric=True, run_id='mnist')
if os.path.exists('{}.meta'.format(model_name)):
    model.load(model_name)
    print("model loaded!")

train=train_data[:-500]
test=train_data[-500:-12]
x=np.array([i[0] for i in train]).reshape(-1,img_size,img_size,1)
y=[i[1] for i in train]

test_x=np.array([i[0] for i in test]).reshape(-1,img_size,img_size,1)
test_y=[i[1] for i in test]

model.fit({'input': x}, {'targets': y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=model_name)

model.save(model_name)
#tensorboard --logdir=/logs



test_data=process_test_data(kern)
#test_data=np.load("test_data.npy")

for data in test_data:
    img_data=data
    orig=img_data
    data=img_data.reshape(img_size,img_size,1)
    model_out=model.predict([data])[0]
    if np.argmax(model_out)==1:
        str_label='NROI'
    else:
        str_label='ROI'
    print("block is: ",str_label)
    
"""
with open("submission.file.csv","w") as f:
    f.write("id,label\n")

with open("submission.file.csv","a") as f:
    for data in test_data:
       img_num=data[1]
       img_data=data
       orig=img_data
       data=img_data.reshape(img_size,img_size,1)
       model_out=model.predict([data])[0]    
       f.write("{},{}\n".format(img_num,model_out[1]))
       """
    

        
    
