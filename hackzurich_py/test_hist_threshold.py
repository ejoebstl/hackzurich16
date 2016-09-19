import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

filedir = '/Users/gabrielfior/Dropbox/Hackzurich16/pupils_cutout/'
readbgr = filedir+'left_pupil232.bmp'
frame = plt.imread(readbgr)

white=plt.imread('/Users/gabrielfior/Dropbox/Hackzurich16/pupils_bw/right_pupil61.bmp')
black=plt.imread('/Users/gabrielfior/Dropbox/Hackzurich16/pupils_bw/right_pupil203.bmp')

#convert to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

plt.figure(1)
plt.clf()
img = cv2.imread(readbgr)
color = ('b','g','r')

b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]

for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

plt.figure(2)
plt.clf()
plt.subplot(211)
ret,th1 = cv2.threshold(img[:,:,0],40,60,cv2.THRESH_BINARY)
plt.imshow(th1)
plt.subplot(212)
plt.imshow(hsv)

#Compare blue channel (when it is smaller than red channel)
#plt.figure(3)

new_mask = np.zeros_like(b)
for i in range(b.shape[0]):
    for j in range(b.shape[1]):
        #if b < r, put 1 else 0
        if (img[:,:,0])[i][j] < (img[:,:,2])[i][j]:
            new_mask[i][j]=1
            
plt.figure(3)
plt.clf()
plt.imshow(new_mask)
            
plt.figure(4)
plt.subplot(211)
plt.title('white')
for i,col in enumerate(color):
    histr = cv2.calcHist([white],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.subplot(212)
plt.title('black')
for i,col in enumerate(color):
    histr = cv2.calcHist([black],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

plt.show()
#################
#Compute diff
mask_white = np.zeros_like(white[:,:,0])
for i in range(white.shape[0]):
    for j in range(white.shape[1]):
        #if b < r, put 1 else 0
        if (white[:,:,0])[i][j] < (white[:,:,2])[i][j]:
            mask_white[i][j]=1

mask_black = np.zeros_like(black[:,:,0])
for i in range(black.shape[0]):
    for j in range(black.shape[1]):
        #if b < r, put 1 else 0
        if (black[:,:,0])[i][j] < (black[:,:,2])[i][j]:
            mask_black[i][j]=1

#Plot masks
plt.figure(5)
plt.subplot(211)
plt.title('white')
plt.imshow(mask_white)
plt.subplot(212)
plt.title('black')
plt.imshow(mask_black)
plt.show()

#Flat fill
