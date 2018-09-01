
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
s1=0

""" make_blocks(pixel,stride,kernel,img_size): this function is made for making the block of different
size from the input image, it contain four argument pixel-it is the image in array form,
stride- it is the stride value for making the block e.g. 1,2,3.. , kernel- it is the block
size of the different blocks, img_size- it is the input image size.""" 
def make_blocks(pixel,stride,kernel,img_size):
    blocks=[] #it will conatin all divided blocks 
    index=[]#it will contain the index value of the elements corresponds to the blocks in the above list
    #blocks = np.array([pixel[i:i+kernel, j:j+kernel] for j in range(0,img_size,kernel) for i in range(0,img_size,kernel)])
    for i in range(0,img_size,stride):#this loop is to traverse the row of the input image
        for j in range(0,img_size,stride):#this loop is to traverse the column of input image
            
            if j+kernel>img_size and i+kernel>img_size:# this if is for that situation in which
                #the both coulmn and row of the divided block is going outside of the input image.
                k=i+kernel-img_size#the no of row going outside of the boundary of the i/p image
                l=j+kernel-img_size#no of column going outside of the boundary if i/p image
                ind1=[]#it is the dummy list which is used to store the index of element of divided block
                
                a=[]#it is the list will store the elemnts of the divide block
                for p in range(i,img_size):#it is for traversing each element in the row
                    b=[]#it is the dummy list which will contain the one row of the divided block
                    ind2=[]#it is the dummy list which will conatin the index value of each elemnt corresponds to the above list
                    for q in range(j,img_size):# it is to traverse each element in coloumn
                        c=[]#it is the dummy list which will store th x and y index of each element of the above ind2 list
                        b.append(pixel[p][q])#here appending the each element of divided block in list b
                        c.append(p)#here appending the x index of above element
                        c.append(q)#here appending the y value of above element
                        ind2.append(c)#appending both x and y value of above elemnt in ind2
                    for m in range(l):#it is to traverse the each element which are outside of the boundary
                        c=[]#same as above 
                        b.append(pixel[p][m])#same as above
                        c.append(p)#same as above 
                        c.append(m)#same as above
                        ind2.append(c)#same as above
                    a.append(b)#here appending the one row of the divided block into a
                    ind1.append(ind2)#here appending the one row of the position of the each element corresponds to the above row
                    #print(ind1)
                
                for p in range(k):#itis for traversing those rows which are outside the boundary of image
                    b=[]#same as above
                    ind2=[]#same as above
                    for q in range(j,img_size):#it is to traverse each element in coloumn
                        c=[]#same as above 
                        b.append(pixel[p][q])#same as above
                        c.append(p)#same as above
                        c.append(q)#same as above
                        ind2.append(c)#same as above
                    for m in range(l):#it is to traverse the each element which are outside of the boundary
                        c=[]#same as above 
                        b.append(pixel[p][m])#same as above
                        c.append(p)#same as above
                        c.append(l)#same as above
                        ind2.append(c)#same as above
                    a.append(b)#same as above
                    ind1.append(ind2)#same as above
                blocks.append(a)#appending the one divided block into blocks
                index.append(ind1)#appending the index value of each element corresponds to above divided block
                
            elif j+kernel>img_size:#it is the situation where the column of block is going outside of the boundary if the image
                k=j+kernel-img_size#how many column is going outside the image boundary
                a=[]
                ind1=[]
                for p in range(i,i+kernel):#to acces the row
                    b=[]
                    ind2=[]
                    for q in range(j,img_size):#to acces those column which are under th boundary
                        c=[]
                        b.append(pixel[p][q])
                        c.append(p)
                        c.append(q)
                        ind2.append(c)
                    for l in range(k):#acces those column which are outside the boundary
                        c=[]
                        b.append(pixel[p][l])
                        c.append(p)
                        c.append(l)
                        ind2.append(c)
                    a.append(b)
                    ind1.append(ind2)
                blocks.append(a)
                index.append(ind1)
                
            elif i+kernel>img_size:#it is the situation in which the no rows is greater than the boundary of the image
                k=i+kernel-img_size#calculating that how many rows are increasing
                a=[]
                ind1=[]
                for p in range(i,img_size):#to acces the rows under the boundary
                    b=[]
                    ind2=[]
                    for q in range(j,j+kernel):#to acces the column
                        c=[]
                        b.append(pixel[p][q])
                        c.append(p)
                        c.append(q)
                        ind2.append(c)
                    #print(ind2)
                    a.append(b)
                    ind1.append(ind2)
                #print(np.array(ind1).shape)
                for p in range(k):#to access the rows which are outside of the boundary
                    b=[]
                    ind2=[]
                    for q in range(j,j+kernel):#to acces the column which is outside th boundary
                        c=[]
                        b.append(pixel[p][q])
                        c.append(p)
                        c.append(q)
                        ind2.append(c)
                    a.append(b)
                    ind1.append(ind2)
                blocks.append(a) 
                index.append(ind1)
            else:#it is for normal condition when neither row nor column or both is outside the boundary
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
                        c.append(q)
                        ind2.append(c)
                    #print(ind2,"\n")
                    a.append(b)
                    ind1.append(ind2)
                #print(ind1,"\n")
                blocks.append(a)
                index.append(ind1)
                #print(np.array(ind1).shape)
                        
                
    blocks=np.array(blocks)#changing the block from list to array
    index=np.array(index)#here changing the index block from list to array
    #print(blocks)
    print(blocks.shape)
    print(index.shape)
    #print(blocks[0])
    #print(index[0])
    return blocks,index#returning the all block and index value for each elemnt in the particular block
""" diff_block_div_one(blocks): it is to get the refrence pixel in the each divided blocks,
here th parametr blocks: is  all the divided blocks by above function"""
def diff_block_div_one(blocks):
    pixel1=[]#it will have the difference value of each divided block]
    #the difference value is the difference of each element with the median of all the element
    q=[]#it will contain the index value of each block and the no of refrencing pixel 
    e2=[]#it will contain the index of those block woose total reference pixel is less than 1
    ind=1#it is used for accessing the indes
    for array in blocks:#accessing each divided block
        b=[]#it is the dummy list whihc will conatin the difference value of one block
        q1=[]#it will contain the index value of one block and referenec value of that block
        med=np.median(array)
        #print(int(med))
        sl=0#it is to calculate total smooth level
        for i in range(len(array[0])):#it is for row
            a=[]#it is the dummy list whihc will conatin the difference value of one row of each block
            for j in range(len(array[i])):#it is for column
                a.append(int(med)-array[i][j])#here we are appending the difference value in a
                if(int(med)-array[i][j]==0):
                    sl=sl+1
            b.append(a)#appending each row in b
        pixel1.append(b)#appending differnece block corresponds to divided block
        s=sum(x.count(0) for x in b)#if the no of smooth level is less than 1 than it will go for second differentiation
        if(s<=1):
            e2.append(ind)#appending the index of the block having sl<1
        q1.append(ind)#appending the index 
        q1.append(sl)#appending the total smooth level
        q.append(q1)#appending above list to q
        ind=ind+1
    pixel1=np.array(pixel1) #converting pixel1 from list to array
    #print(pixel1)
    #print(pixel1.shape) 
    #print(q)
    return q,pixel1,e2#returning the q-index and total sl, pixel1-difference block, e2-blocks index having sl <1

def diff_block_div_two(blocks,pixel1,e2):#this function is for again dividing the blocks which have sl <1
    #the argument blocks-it contains all the divided blocks
    #pixel1, it conatains the first time difference block
    #e2-contains the index of those blocks having sl<1
    pixel2=[]#it will contain the second difference blocks
    q2=[]#it will contain the index of block and total sl after second divison
    for b in e2:#it is to get all those block having sl<1
        g=[]#it will contain the elements of one block which are greater than median of all the elment of the block
        s=[]#similarly it will contain all less than
        bl=[]#it is the dummy list which will store one block
        q1=[]#it will contain the index and total sl in a block
        med1=0#median of the elements geater than the whole median
        med2=0#median of the elements less than the whole median
        med=int(np.median(blocks[b-1]))#median of all the element of the block
        #print(int(med))
        sl=0#to calculate the total smooth level
        for i in range(len(blocks[b-1][0])):#to traverse row
            for j in blocks[b-1][i]:#to traverse coulumn
                if(j>med):
                    g.append(j)#elements greater than whole median
                elif j<med:
                    s.append(j)#elements less than whole block
                else:
                    continue
        med1=int(np.median(g))#median of elements greater than the whole median
        med2=int(np.median(s))#median of elements less than the whole median
        if not g:#if there is no elements greater than the whole median, than we will make med1=0
            med1=0
        if not s:#if there is no elements less than the whole median, than we will make med2=0
            med2=0
        #print(med1,med2)
        #print(med)
        #now i am doing the second time differeence of elements
        for i in range(len(blocks[b-1][0])):#it is to traverse row
            a=[]#it is to store one row
            for j in blocks[b-1][i]:#it is o traverse coloumn
                if(j>med):
                    if(med1!=0):#it is for greater
                        a.append(med1-j)
                    else:
                        a.append(j)
                elif j<med: 
                    if(med2!=0):#it is for lesson
                        a.append(med2-j)
                    else:
                        a.append(j)
                else:              
                    a.append(med-j)
                if(med-j==0 or med1-j==0 or med2-j==0 ):#calculating sl
                    sl=sl+1
            bl.append(a)#appendig the second difference row into bl
        pixel2.append(bl)#appending the whole one block
        #pixel1[b-1]=bl
        q1.append(b)#index of block
        q1.append(sl)#total sl
        q2.append(q1)
            
    #print(np.array(pixel2))
    #print("\n")
    #print(q2)
    return q2,pixel2#returning q2,and pixel2


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

def make_dataset(pixel1,q3,kern,thresold,file_name):
    with open(file_name,"w") as f:
        for i in range(kern**2):
            f.write("x{},".format((i+1)))
        f.write("label\n")
        
    with open(file_name,"a") as f:
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

def main(img_size,thresold):
    print("\nTraining Image!")
    img_path="/home/amanthakur/Desktop/lena.jpeg"
    img=cv2.resize(cv2.imread(img_path,0),(img_size,img_size))
    #np.reshape(img,[img_size,img_size])
    plt.imshow(img,cmap="gray")
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    pixel=np.array(img)
    #print(pixel)
    #print(pixel)
    print(pixel.shape,len(pixel))
    
    print("\nfor stride = ",4," and kernel = ",4)
    blocks,index=make_blocks(pixel,4,4,img_size)
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
        if j>=thresold:
            tsl+=total_sl[j]
            tsl1=tsl1+j*total_sl[j]
    print("\nfor stride: ",4," and kernel: ",4," with total smooth level>",thresold,":",tsl,"total smooth level: ",tsl1)
    strd=4
    kern=4
    
    plt.imshow(img,cmap="gray")
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    #make_dataset(pixel1,q3,kern,thresold,"sdcvg.csv")
    return pixel1,q3,kern,strd
    
    #thresold=[5,7,10,12]
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

            

"""
if __name__=="__main__":
    img_size=256 #image size to 
    pixel1,q3,kern,strd=main(img_size,10)
    """
    
