import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle
import pandas as pd
import starthide as dt
from sklearn.metrics import confusion_matrix,accuracy_score
from PIL import Image
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



def process_test_data(kern,thresold,strd):
    print("\nTesting Image!")
    img_size1=256
    img=cv2.resize(cv2.imread("/home/amanthakur/Downloads/train/cat.14.jpg",0),(img_size1,img_size1))
    plt.imshow(img,cmap="gray")
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    pixel=np.array(img)
    print("\nwait for a minute processing testing data....")
    blocks,index=dt.make_blocks(pixel,strd,kern,img_size1)
    q3,pixel1,e2=dt.diff_block_div_one(blocks)
    dt.make_dataset(pixel1,q3,kern,thresold,"testing_file.csv")
    test_dataset=pd.read_csv("testing_file.csv").values
    testing_data=[]
    y_test=[]
    for img in test_dataset:
        a=[]
        for i in range(0,len(img)-1,kern):
            a.append(img[i:i+kern])
        a=np.array(a)
        #print(a)
        
        testing_data.append(a)
        y_test.append(img[-1])
    if testing_data:
        print("\ntesting data has been processed..")
    np.save("test_data.npy",(testing_data,y_test,index,img_size1))
    return testing_data,y_test,index,img_size1

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
    
    model.fit({'input': x}, {'targets': y}, n_epoch=0, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=False, run_id=model_name)

    #model.save(model_name)
    
    if os.path.exists('{}.meta'.format(model_name)):
        model.load(model_name)
        print("model loaded!")
    return model
def test_model(model,test_data,kern,index,img_size1):
    print("Testing the Image........")
    test_path="/home/amanthakur/Downloads/train/cat.14.jpg"
    img=cv2.resize(cv2.imread(test_path,0),(img_size1,img_size1))
    y_pred=[]
    ind=0
    file_name=test_path.split("/")[-1]
    with open(file_name+".csv","w") as f:
        f.write("x,")
        f.write("y\n")
    
    with open(file_name+".csv","a") as f:
        
        for data in test_data:
            img_data=data
            data=img_data.reshape(kern,kern,1)
            model_out=model.predict([data])[0]
            if np.argmax(model_out)==1:
                str_label='NROI'
            else:
                str_label='ROI'
            if str_label=='ROI':
                #blocks,index1=dt.make_blocks(img_data,1,2,kern)
                #ind2=0
                #blocks=list(blocks)
                index1=index[ind]
                for i in range(len(data[0])):
                    for j in range(len(data[i])):
                        
                        if(data[i][j]==0):
                            x,y=index1[i][j]
                            x1,y1=index[ind][0][0]
                            x2,y2=index[ind][-1][-1]
                            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
                            test=pd.read_csv(file_name+".csv").values    
                            if x not in test[:,0]:
                                f.write("{},".format(x))
                                f.write("{}\n".format(y))
                            else:
                                if y not in test[:,1]:
                                    f.write("{},".format(x))
                                    f.write("{}\n".format(y))
                    
                    
            y_pred.append(str_label)
            ind+=1
    plt.imshow(img,cmap="gray")
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    return y_pred,file_name
        
def confusion_metrics(y_test,y_pred):
    return confusion_matrix(y_test,y_pred)

def accuracy(y_test,y_pred):
    return accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)


def hiding_image(filename):
    print("\nImage that will be hidden!")
    img_size1=256
    test_path="/home/amanthakur/Downloads/train/cat.14.jpg"
    test_img=cv2.resize(cv2.imread(test_path,0),(img_size1,img_size1))
    pixel=np.array(test_img)
    test_dataset=pd.read_csv(filename+".csv").values
    #shuffle(test_dataset)
    hiding_image_path="/home/amanthakur/Desktop/imagetohide.png"
    test_img=cv2.resize(cv2.imread(hiding_image_path,0),(50,50))
    plt.imshow(test_img,cmap="gray")
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    print("\nHiding the image...")
    hiding_pixel=np.array(test_img)
    ind=0
    pixel=list(pixel)
    #print(pixel)
    #print(test_dataset)
    for i in hiding_pixel:
        for j in i:
            x,y=test_dataset[ind]
            #print(x,y)
            #print(pixel[x][y])
            pixel[x][y]=j
            ind=ind+1
    pixel=np.array(pixel)
    
    img=Image.fromarray(pixel)
    img.save("test.png")
        
    
    
    
    
    

thresold=10
kern=4
strd=4
#pixel1,q3,kern,strd=dt.main(img_size,thresold)

#train_data=create_train_data(pixel1,q3,kern,thresold)
#if you already have train data
train_data=np.load("training_data.npy")

test_data,y_test,index,img_size1=process_test_data(kern,thresold,strd)
#test_data,y_test,index,img_size1=np.load("test_data.npy")
model=train_model(train_data,kern)
y_pred,file_name=test_model(model,test_data,kern,index,img_size1)

matrix=confusion_metrics(y_test,y_pred)

print("confusion matrix: \n",matrix)

print("accuracy of the model: ",accuracy(y_test,y_pred))

hiding_image(file_name)




    

        
    