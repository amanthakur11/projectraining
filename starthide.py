import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
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
for i in range(0,28,4):
    for j in range(0,28,4):
        
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
e2=[]
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
    s=sum(x.count(0) for x in b)
    if(s<=1):
        e2.append(ind)
    q1.append(ind)
    q1.append(sl)
    q.append(q1)
    ind=ind+1
pixel1=np.array(pixel1) 
#print(pixel1)
#print(pixel1.shape) 
print(q)
pixel2=[]
q2=[]
for b in e2:
    g=[]
    s=[]
    bl=[]
    q1=[]
    med1=0
    med2=0
    med=int(np.median(blocks[b-1]))
    #print(int(med))
    sl=0
    for i in range(len(blocks[b-1][0])):
        for j in blocks[b-1][i]:
            if(j>med):
                g.append(j)
            elif j<med:
                s.append(j)
            else:
                continue
    med1=int(np.median(g))
    med2=int(np.median(s))
    if not g:
        med1=0
    if not s:
        med2=0
    #print(med1,med2)
    #print(med)
    
    for i in range(len(blocks[b-1][0])):
        a=[]
        for j in blocks[b-1][i]:
            if(j>med):
                if(med1!=0):
                    a.append(med1-j)
                else:
                    a.append(j)
            elif j<med:
                if(med2!=0):
                    a.append(med2-j)
                else:
                    a.append(j)
            else:              
                a.append(med-j)
            if(med-j==0 or med1-j==0 or med2-j==0 ):
                sl=sl+1
        bl.append(a)
    pixel2.append(bl)
    pixel1[b-1]=bl
    q1.append(b)
    q1.append(sl)
    q2.append(q1)
        
#print(np.array(pixel2))
print("\n")
print(q2) 
    
        
        

#for smooth level 1

x1=[]
for i in q:
    if(i[1]==1):
        x1.append(pixel1[i[0]-1])
x1=np.array(x1).ravel()
num_bins = 50

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x1, num_bins, density=1)

ax.set_xlabel('Difference with the reference value')
ax.set_ylabel('Possibilty percentage')
#ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()


#for smooth level 2

x2=[]
for i in q:
    if(i[1]==2):
        x2.append(pixel1[i[0]-1])
x2=np.array(x2).ravel()
num_bins = 50

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x2, num_bins, density=1)

ax.set_xlabel('Difference with the reference value')
ax.set_ylabel('Possibilty percentage')
#ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()


#for smooth level 3

x3=[]
for i in q:
    if(i[1]==3):
        x3.append(pixel1[i[0]-1])
x3=np.array(x3).ravel()
num_bins = 50

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x3, num_bins, density=1)

ax.set_xlabel('Difference with the reference value')
ax.set_ylabel('Possibilty percentage')
#ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()


        

# List of five airlines to plot
sl = ['smooth level 1', 'smooth level 2', 'smooth level 3']

# Iterate through the five airlines
for sls in sl:
    
    if(sls=='smooth level 1'):   
        z=x1
    elif(sls=='smooth level 2'):
        z=x2
    else:
        z=x3
    # Draw the density plot
    sns.distplot(z, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label =sls)
    
# Plot formatting
plt.legend(prop={'size': 10}, title = 'levels')
plt.title('comparision of difference historam')
plt.xlabel('difference with the reference value')
plt.ylabel('possibility percentage')
