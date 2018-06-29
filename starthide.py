import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
s1=0
def make_blocks(pixel,stride,kernel):
    blocks=[]
    index=[]
    #blocks = np.array([pixel[i:i+kernel, j:j+kernel] for j in range(0,100,kernel) for i in range(0,100,kernel)])
    for i in range(0,100,stride):
        for j in range(0,100,stride):
            
            if j+kernel>100 and i+kernel>100:
                k=i+kernel-100
                l=j+kernel-100
                ind1=[]
                a=[]
                for p in range(i,100):
                    b=[]
                    ind2=[]
                    for q in range(j,100):
                        c=[]
                        b.append(pixel[p][q])
                        c.append(p)
                        c.append(list(pixel[p]).index(pixel[p][q]))
                        ind2.append(c)
                    for m in range(l):
                        c=[]
                        b.append(pixel[p][m])
                        c.append(p)
                        c.append(list(pixel[p]).index(pixel[p][m]))
                        ind2.append(c)
                    a.append(b)
                    ind1.append(ind2)
                    #print(ind1)
                
                for p in range(k):
                    b=[]
                    ind2=[]
                    for q in range(j,100):
                        c=[]
                        b.append(pixel[p][q])
                        c.append(p)
                        c.append(list(pixel[p]).index(pixel[p][q]))
                        ind2.append(c)
                    for m in range(l):
                        c=[]
                        b.append(pixel[p][m])
                        c.append(p)
                        c.append(list(pixel[p]).index(pixel[p][m]))
                        ind2.append(c)
                    a.append(b)
                    ind1.append(ind2)
                blocks.append(a)
                index.append(ind1)
                
            elif j+kernel>100:
                k=j+kernel-100
                a=[]
                ind1=[]
                for p in range(i,i+kernel):
                    b=[]
                    ind2=[]
                    for q in range(j,100):
                        c=[]
                        b.append(pixel[p][q])
                        c.append(p)
                        c.append(list(pixel[p]).index(pixel[p][q]))
                        ind2.append(c)
                    for l in range(k):
                        c=[]
                        b.append(pixel[p][l])
                        c.append(p)
                        c.append(list(pixel[p]).index(pixel[p][l]))
                        ind2.append(c)
                    a.append(b)
                    ind1.append(ind2)
                blocks.append(a)
                index.append(ind1)
                
            elif i+kernel>100:
                k=i+kernel-100
                a=[]
                ind1=[]
                for p in range(i,100):
                    b=[]
                    ind2=[]
                    for q in range(j,j+kernel):
                        c=[]
                        b.append(pixel[p][q])
                        c.append(p)
                        c.append(list(pixel[p]).index(pixel[p][q]))
                        ind2.append(c)
                    #print(ind2)
                    a.append(b)
                    ind1.append(ind2)
                #print(np.array(ind1).shape)
                for p in range(k):
                    b=[]
                    ind2=[]
                    for q in range(j,j+kernel):
                        c=[]
                        b.append(pixel[p][q])
                        c.append(p)
                        c.append(list(pixel[p]).index(pixel[p][q]))
                        ind2.append(c)
                    a.append(b)
                    ind1.append(ind2)
                blocks.append(a) 
                index.append(ind1)
            else:
                #blocks.append(pixel[i:i+kernel,j:j+kernel])
                a=[]
                ind1=[]
                for p in range(i,i+kernel):
                    b=[]
                    ind2=[]
                    for q in range(j,j+kernel):
                        c=[]
                        b.append(pixel[p][q])
                        c.append(p)
                        c.append(list(pixel[p]).index(pixel[p][q]))
                        ind2.append(c)
                    #print(ind2,"\n")
                    a.append(b)
                    ind1.append(ind2)
                #print(ind1,"\n")
                blocks.append(a)
                index.append(ind1)
                #print(np.array(ind1).shape)
                        
                
    blocks=np.array(blocks)
    index=np.array(index)
    #print(blocks)
    print(blocks.shape)
    print(index.shape)
    #print(blocks[0])
    #print(index[0])
    return blocks,index

def diff_block_div_one(blocks):
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
    #print(q)
    return q,pixel1,e2

def diff_block_div_two(blocks,pixel1,e2):
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
        #pixel1[b-1]=bl
        q1.append(b)
        q1.append(sl)
        q2.append(q1)
            
    #print(np.array(pixel2))
    #print("\n")
    #print(q2)
    return q2,pixel2


def draw_cumulative_histogram(x1,x2,x3):
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

def tsl_calculate(q1,q2,thresold):
    count={}
    count1={}
    block1={}
    block2={}
    for i in q1:
        if i[1] not in count:
            count[i[1]]=0
        count[i[1]]+=1
        if i[1]>=thresold:
            if i[1] not in block1:
                block1[i[1]]=[]
            block1[i[1]].append(i[0])
                
    for i in q2:
        if i[1] not in count1:
            count1[i[1]]=0
        count1[i[1]]+=1
        if i[1]>=thresold:
            if i[1] not in block2:
                block2[i[1]]=[]
            block2[i[1]].append(i[0])
    return count,count1,block1,block2

def make_dataset(pixel1,q3,kern,thresold):
    with open("training.file.csv","w") as f:
        for i in range(kern**2):
            f.write("x{},".format((i+1)))
        f.write("label\n")
        
    with open("training.file.csv","a") as f:
        c=0
        for i in q3:
            p=pixel1[i[0]-1]
            if i[1]>=thresold:
                c=c+1
                for k in range(len(p[0])):
                    for n in range(len(p[k])):
                        f.write("{},".format(p[k][n]))
                f.write("ROI\n")
            else:
                for k in range(len(p[0])):
                    for n in range(len(p[k])):
                        f.write("{},".format(p[k][n]))
                f.write("NROI\n")
   #return c

def main():
    img=cv2.resize(cv2.imread("/home/amanthakur/Desktop/lena.jpeg",0),(100,100))
    #np.reshape(img,[100,100])
    plt.imshow(img,cmap="gray")
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    pixel=np.array(img)
    #print(pixel)
    #print(pixel)
    print(pixel.shape,len(pixel))
    stride=[1,2,3,4,5]
    kernel=[3,4,5,7]
    optimal=[]
    thresold=10
    for i in stride:
        for l in kernel:
            opt=[]
            print("\nfor stride = ",i," and kernel = ",l)
            blocks,index=make_blocks(pixel,i,l)
            print("total smooth level before dividing blocks: ")
            q3,pixel1,e2=diff_block_div_one(blocks)
            q4,pixel2=diff_block_div_two(blocks,pixel1,e2)
            total_sl,total_sl1,block1,block2=tsl_calculate(q3,q4,thresold)
            print(total_sl)
            #print("\nblocks for different smooth level: ",sorted(block1))
            print("\ntotal smooth level after dividing blocks: ",total_sl1)
            #print("\nblocks for different smooth level: ",sorted(block2))
            tsl=0
            tsl1=0
            for j in total_sl.keys():
                if j>=10:
                    tsl+=total_sl[j]
                    tsl1=tsl1+j*total_sl[j]
            
            opt.append(i)
            opt.append(l)
            opt.append(tsl)
            opt.append(tsl1)
            optimal.append(opt)
    
    print(optimal) 
    strd=optimal[0][0]
    kern=optimal[0][1]
    m=optimal[0][2]
    m1=optimal[0][3]
    
    for j in optimal:
        if j[3]>m1:
            strd=j[0]
            kern=j[1]
            m=j[2]
            m1=j[3]
            
            
                
    print("\noptimal tecnique is for stride: ",strd," and kernel: ",kern," with total smooth level>5 : ",m," total smooth level: ",m1)
    print("\nfor stride = ",strd," and kernel = ",kern)
    
    blocks,index=make_blocks(pixel,strd,kern)
    q3,pixel1,e2=diff_block_div_one(blocks)
    q4,pixel2=diff_block_div_two(blocks,pixel1,e2)
    total_sl,total_sl1,block1,block2=tsl_calculate(q3,q4,thresold)

    for j in q3:
       x=[]
       y=[]
       if j[1]>=5:
            p=pixel1[j[0]-1]
            inx=index[j[0]-1]
  
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
            
            for k in range(len(p[0])):
                for n in range(len(p[k])):
                    if p[k][n]==0:
                        x.append(inx[k][n][0])
                        y.append(inx[k][n][1])
            for j in range(len(x)-1):
                cv2.line(img,(x[j],y[j]),(x[j+1],y[j+1]),(255,0,0),2)
    plt.imshow(img,cmap="gray")
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    make_dataset(pixel1,q3,kern,thresold)
    return pixel1,kern
    
    #hresold=[5,7,10,12]
   #roi=[]
   #roi.append(0)
   #j=0
   #th=0
    #or i in thresold:
       #roi.append(make_dataset(pixel1,q3,kern,i))
       #if(roi[j]-roi[j+1] in [1,2,3,4,5,6]):
          # th=i
          # break
   #print("thresold: ",th," roi: ",r)

            
        
    
    
    

                
                
                             
            

                        
                        
                
    
    

if __name__=="__main__":
    px,krn=main()
    
        
        
    
    
    
