
import numpy as np
import os
from random import shuffle
import pandas as pd
import create_data as dt

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
#train_dir='/home/amanthakur/Downloads/train'
#test_dir='/home/amanthakur/Downloads/test1'
lr=0.001
img_size=256

model_name="Data_Hinding-()-()-model".format(lr,"6conv-basic")

def label_image(img):
    
    word_label=img[-1]
    print(word_label)
    if word_label=='ROI': return [1,0]
    elif word_label=='NROI': return [0,1]


def create_train_data(pixel1,q3,kern,thresold):
    print("\nCreating training data....")
    dt.make_dataset(pixel1,q3,kern,thresold,"training_file.csv")
    train_dataset=pd.read_csv("training_file.csv").values
    training_data=[]
    for img in train_dataset:
        label=label_image(img)
        print(label)
        a=[]
        for i in range(0,len(img)-1,kern):
            a.append(img[i:i+kern])
        a=np.array(a)
        print(a)
        training_data.append([a,np.array(label)])
    if training_data:
        print("\ntraining data has been created.")
    shuffle(training_data)
    np.save("training_data.npy",training_data)
    return training_data


def train_model(train_data,kern):
    
    train=train_data[:-500]
    test=train_data[-500:]
    x=np.array([i[0] for i in train]).reshape(-1,kern,kern,1)
    y=[i[1] for i in train]
    
    test_x=np.array([i[0] for i in test]).reshape(-1,kern,kern,1)
    test_y=[i[1] for i in test]
    
    tf.reset_default_graph()

    convnet = input_data(shape=[None, kern,kern, 1], name='input')
    #print(convnet)
    convnet = conv_2d(convnet, 32, 2, activation='tanh')
    #print(convnet)
    convnet = max_pool_2d(convnet, 2)
    #print(convnet)
    
    convnet = conv_2d(convnet, 64, 2, activation='tanh')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet, 32, 2, activation='tanh')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet, 64, 2, activation='tanh')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet, 32, 2, activation='tanh')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet, 64, 2, activation='tanh')
    convnet = max_pool_2d(convnet, 2)
    #print(convnet)
    convnet = fully_connected(convnet, 1024, activation='tanh')
    convnet = dropout(convnet, 0.8)
    
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')
    
    model = tflearn.DNN(convnet,tensorboard_dir='log')
    
    model.fit({'input': x}, {'targets': y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=model_name)

    model.save(model_name)
    
    if os.path.exists('{}.meta'.format(model_name)):
        model.load(model_name)
        print("model loaded!")
    return model


thresold=10
#kern=7
pixel1,q3,kern,strd=dt.main(img_size,thresold)

train_data=create_train_data(pixel1,q3,kern,thresold)
#if you already have train data
#train_data=np.load("training_data.npy")

#test_data,y_test,index,img_size1=np.load("test_data.npy")
print("\ntraining the model.")
model=train_model(train_data,kern)

print("\nModel has been trained!")
    

        
    