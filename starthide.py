import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

img=cv2.resize(cv2.imread("/home/amanthakur/Desktop/lena.jpeg",0),(28,28))
#np.reshape(img,[28,28])
plt.imshow(img,cmap="gray")
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

pixel=np.array(img)
#print(pixel)
print(pixel.shape,len(pixel))

#imgg=Image.fromarray(array)
#imgg.save("len1.jpeg")
blocks = np.array([pixel[i:i+4, j:j+4] for j in range(0,28,4) for i in range(0,28,4)])
#print(blocks)
print(blocks.shape)
pixel1=[]
q=[]
ind=1
for array in blocks:
    b=[]
    q1=[]
    med=np.median(array)
    print(int(med))
    sl=0
    for i in range(len(array[0])):
        a=[]
        for j in range(len(array[i])):
            a.append(int(med)-array[i][j])
            if(int(med)-array[i][j]==0):
                sl=sl+1
        b.append(a)
    pixel1.append(b)
    q1.append(ind)
    q1.append(sl)
    q.append(q1)
    ind=ind+1
pixel1=np.array(pixel1) 
print(pixel1)
print(pixel1.shape) 
print(q)



