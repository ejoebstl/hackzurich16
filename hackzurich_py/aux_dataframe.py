# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 16:13:43 2016

@author: gabrielfior
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import pandas as pd


def convert_datetime(row,t0):

    print 'entered outside func'

    converted_datetime = datetime.datetime.fromtimestamp(np.double(row['ms'])/1000.).strftime('%Y-%m-%d %H:%M:%S.%f')
    converted_datetime=datetime.datetime.strptime(converted_datetime, '%Y-%m-%d %H:%M:%S.%f')

    a=(converted_datetime-t0)
    print a
    print a.seconds
    print a.microseconds, type(a.microseconds)
    return ((np.double(a.seconds)*1000.) + (np.double(a.microseconds)/1000.))

def get_date_with_mili(row):
    

    print 'extract date with mili'
    a = datetime.datetime.fromtimestamp(np.double(row.ms)/1000.).strftime('%Y-%m-%d %H:%M:%S.%f')
    a=datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S.%f')
    
    return a

def extract_second(row):

    print 'extract second'
    a = datetime.datetime.fromtimestamp(np.double(row.ms)/1000.).strftime('%Y-%m-%d %H:%M:%S')
    a=datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S')
    
    return a