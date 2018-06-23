import numpy as np
import cv2
import matplotlib.pyplot as plt

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
blocks=[]
#blocks = np.array([pixel[i:i+4, j:j+4] for j in range(0,28,4) for i in range(0,28,4)])
for i in range(0,28,1):
    for j in range(0,28,1):
        
        if j+4>28 and i+4>28:
            k=i+4-28
            l=j+4-28
            a=[]
            for p in range(i,28):
                b=[]
                for q in range(j,28):
                    b.append(pixel[p][q])
                for m in range(l):
                    b.append(pixel[p][m])
                a.append(b)
            
            for p in range(k):
                b=[]
                for q in range(j,28):
                    b.append(pixel[p][q])
                for m in range(l):
                    b.append(pixel[p][m])
                a.append(b)
            blocks.append(a)
            
        elif j+4>28:
            k=j+4-28
            a=[]
            for p in range(i,i+4):
                b=[]
                for q in range(j,28):
                    b.append(pixel[p][q])
                for l in range(k):
                    b.append(pixel[p][l])
                a.append(b)
            blocks.append(a)
            
        elif i+4>28:
            k=i+4-28
            a=[]
            for p in range(i,28):
                b=[]
                for q in range(j,j+4):
                    b.append(pixel[p][q])
                a.append(b)
            for p in range(k):
                b=[]
                for q in range(j,j+4):
                    b.append(pixel[p][q])
                a.append(b)
            blocks.append(a)   
        else:
            blocks.append(pixel[i:i+4,j:j+4])
            
blocks=np.array(blocks)
#print(blocks)
print(blocks.shape)


pixel1=[]
q=[]
ind=1
for array in blocks:
    b=[]
    q1=[]
    med=np.median(array)
    #print(int(med))
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
#print(pixel1)
#print(pixel1.shape) 
#print(q)


print(blocks[10])
print(pixel1[10])
x=pixel1[10].ravel()

num_bins = 50

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=1)

ax.set_xlabel('Difference with the reference value')
ax.set_ylabel('Possibilty percentage')
#ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()


x1=np.array(q)[0:,1]
bins=10

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x1, bins, density=1)

ax.set_xlabel('Difference with the reference value')
ax.set_ylabel('Possibilty percentage')
#ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()
