# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 00:54:49 2016

@author: gabrielfior
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import cv2.cv as cv
import json

os.chdir('/Users/gabrielfior/hackzurich')

filedir = os.getcwd()+'/images/'

img_list=os.listdir(filedir)



# import the necessary packages

#img1 = plt.imread(filedir+'img00004.jpeg')

#Loop all images
store_circles=[]
for i in img_list:
    filepath = filedir+i
    print filepath
    
    img = cv2.imread(filepath,0)
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    #circles = cv2.HoughCircles(gray,cv.CV_HOUGH_GRADIENT, 1, 10)
    circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT,1,150,
                                param1=50,param2=30,minRadius=50,maxRadius=700)
    
    try:
        
        circles = np.uint16(np.around(circles))

        store_circles.append(circles[0][:,2].max())
    except AttributeError:
        pass
    #list_circles=circles[0,0]
    #for i in circles[0,:]:
    #
    #    # draw the outer circle
    #    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    #    # draw the center of the circle
    #    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
    #cv2.imshow('detected circles',cimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
###############
    