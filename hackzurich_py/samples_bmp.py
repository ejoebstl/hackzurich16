# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 01:55:45 2016

@author: gabrielfior
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

import scipy.ndimage as ndimage
from pupil_detection import GetPupil, detectPupilKMeans

plot_all = False

filedir = '/Users/gabrielfior/hackzurich/bmp_pupil/'

img1 = plt.imread(filedir+'left_pupil30.bmp')[:]
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)

gray = gray[:,:,1]*gray[:,:,2]
#print gray.shape

pupilBlobs,pupilEllipses = detectPupilKMeans(gray,4,2,(40,40))