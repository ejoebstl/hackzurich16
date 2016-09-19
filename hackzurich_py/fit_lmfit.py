# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 00:04:43 2016

@author: gabrielfior
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import StepModel

plotting=False

filedir = os.getcwd()+'/images/'
img1 = plt.imread(filedir+'img00005.jpeg')

r = img1[int(img1.shape[0]/2),:,0]
g=img1[int(img1.shape[0]/2),:,1]
b=img1[int(img1.shape[0]/2),:,2]

diff= (abs(r-g) + abs(r-b) + abs(g-b)) 

mod = StepModel(form='erf', prefix='step_')
pars =  mod.guess(y, x=x, center=len(y)/2.)

x=range(len(diff))
y=diff
pars = mod.guess(diff,x=x)
out = mod.fit(y,pars,x=x)
print(out.fit_report(min_correl=0.25))

if plotting:
    plt.plot(x, y)
    plt.plot(x, out.init_fit, 'k--')
    plt.plot(x, out.best_fit, 'r-')
    plt.show()

#######
# Plot derivative
