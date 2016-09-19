# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 23:31:23 2016

@author: gabrielfior
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


#convert to hsv images

filedir = '/Users/gabrielfior/Dropbox/Hackzurich16/pupils_cutout/'
img1 = plt.imread(filedir+'left_pupil232.bmp')

#Pre processing
#rgb channels

#line_height= int(img1.shape[0]/2)
#line_height=15

hsv = cv2.cvtColor(img1[:], cv2.COLOR_BGR2HSV)
h = img1[:,:,0]
s=img1[:,:,1]
v=img1[:,:,2]


#diff= (abs(r-g) + abs(r-b) + abs(g-b)) 

plt.figure(1)
plt.clf()
plt.imshow(img1)

#plt.figure(2)
#plt.clf()
#center lineplot of pupil
#plt.plot(r,'r',label='r')
#plt.plot(g,'g',label='g')
#plt.plot(b,'b',label='b')

#plt.figure(3)
#plt.clf()
#plt.plot(diff,'o',label='diff')
#plt.plot(np.gradient(diff),'o',label='diff_der')
#plt.plot(np.gradient(np.gradient(diff)),label='diff_der2')

fig2 = plt.figure(2)
plt.title('h')
plt.clf()
#red channel
plt.hist(h.ravel(), bins=256, range=(0.0, 255.), fc='k', ec='k')

fig3 = plt.figure(3)
plt.title('s')
plt.clf()
#red channel
plt.hist(s.ravel(), bins=256, range=(0.0, 255.), fc='k', ec='k')

fig4 = plt.figure(4)
plt.title('v')
plt.clf()
#red channel
plt.hist(v.ravel(), bins=256, range=(0.0, 255.), fc='k', ec='k')

fig5 = plt.figure(5)
plt.plot(h,label='h')
plt.plot(s,label='s')
plt.plot(v,label='v')

#examine v
plt.figure(6)
plt.plot(v[0,:],label='0')
plt.plot(v[int(v.shape[0]/2),:],label='middle')
plt.plot(v[int(v.shape[0]-1),:],label='top')
plt.legend()