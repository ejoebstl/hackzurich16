# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

filepath = '/Users/gabrielfior/Dropbox/Hackzurich16/pupils_cutout_3/right_plot.txt'

with open(filepath) as f:
    x = f.readlines()
    
posx=[]
posy=[]
outer_radius=[]
inner_radius=[]

for i in x:
    a=i.split(' ')
    
    if np.double(a[7].replace(',','').replace(']','')) != 0:    
    
        posx.append(np.double(a[4].replace(',','').replace('[','')))
        posy.append(np.double(a[5].replace(',','')))
        outer_radius.append(np.double(a[6].replace(',','')))
        inner_radius.append(np.double(a[7].replace(',','').replace(']','')))
    

ar_inner_radius = np.array(inner_radius)
ar_outer_radius = np.array(outer_radius)
    
plt.figure(1)
plt.clf()
plt.plot(inner_radius,label='pupil')
plt.plot(outer_radius,label='eye')
plt.plot(ar_inner_radius/ar_outer_radius,label='ratio')

#plt.plot(inner_radius/outer_radius,label='ratio')

plt.legend()

plt.figure(2)
plt.clf()
plt.plot(inner_radius,label='pupil')
plt.plot(np.gradient(inner_radius),label='grad pupil')
plt.legend()    