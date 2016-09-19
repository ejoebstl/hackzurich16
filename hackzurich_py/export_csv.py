# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 18:02:02 2016

@author: gabrielfior
"""

import os

num1 = 10
filename='/Users/gabrielfior/OneDrive/Master Thesis/Share_Ubuntu/out'+str(num1)+'.txt'
filename='/Users/gabrielfior/OneDrive/Master Thesis/Share_Ubuntu/test'+str(num1)+'.txt'
with open(filename) as f:
    x=f.readlines()
    
#Read all lines, export right pupil
lines=[]
for i in x:
    if 'right_pupil' in i:
        lines.append(i)
      
count=1
with open('/Users/gabrielfior/Dropbox/Hackzurich16/pupils_cutout_6/test_wb'+filename[-5]+'.txt', 'w') as f:
    
    for item in lines:
      f.write("%i,%s\n" % (count,item))
      count +=1
      
#Read all lines, export right pupil
lines=[]
for i in x:
    if 'left_pupil' in i:
        lines.append(i)
      
count=1
with open('/Users/gabrielfior/Dropbox/Hackzurich16/pupils_cutout_6/test_left_wb'+filename[-5]+'.txt', 'w') as f:
    for item in lines:
      f.write("%i,%s\n" % (count,item))
      count +=1      