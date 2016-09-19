# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 23:24:11 2016

@author: gabrielfior
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import pandas as pd
import aux_dataframe as aux

#filepath = '/Users/gabrielfior/Dropbox/Hackzurich16/pupils_cutout_6/out_wb7.txt'

#names1=['index','window_name','ms','bestLargex','bestLargey','bestLarger1',
#                        'BestLarger2','bestLargematches','bestLargerejects',
#                        'bestLargeavg','bestSmallmatches','bestSmallrejects',
#                        'bestSmallavg']
#names2=['index','window_name','bestLargex','bestLargey','bestLarger1',
#                        'BestLarger2','bestLargematches','bestLargerejects',
#                        'bestLargeavg','bestSmallmatches','bestSmallrejects',
#                        'bestSmallavg']


#Give me a data frame and I will give you white time and black time back!
def give_responses(df):
        
    t0 = df.ms.min()
    t0 = datetime.datetime.fromtimestamp(float(t0)/1000.).strftime('%Y-%m-%d %H:%M:%S.%f')
    t0=datetime.datetime.strptime(t0, '%Y-%m-%d %H:%M:%S.%f')
    
    #print 'here'
    df['datetime1']=df.apply(aux.convert_datetime,args=(t0,),axis=1)
    df['second']=df.apply(aux.extract_second,axis=1)
    df['date2']=df.apply(aux.get_date_with_mili,axis=1)
    #print 'here2'
    
    
    df = df[df.bestLargematches != 0]
    df['radius_ratio'] = df.bestLarger1/df.BestLarger2
    
    df['avg_ratio']= df.bestSmallavg/df.bestLargeavg
    df['matches_ratio']= df.bestLargematches/df.bestSmallmatches
        
    
    ############
    #Group by second
    df_group_median = df.groupby('second').median()
    df_group_sum = df.groupby('second').sum()
    df_group_avg = df.groupby('second').mean()
        
    #plt.subplot(334)
    #plt.plot(np.array(df_group_sum.matches_ratio),label='sum, matches_rat')
    
    #plt.subplot(339)
    #plt.plot(np.array(df_group_avg.avg_ratio),label='avg, avg_rat')
    
    
    ##########################
    #white response
    max_white = np.array(df_group_sum.matches_ratio).max()
    #a=df_group_sum[df_group_sum.matches_ratio==max_white].datetime1
    for key,value in df_group_sum[df_group_sum.matches_ratio==max_white].datetime1.iteritems():
        time_white= key 
    #get second, get max from this second
    b1 = df[df.second==time_white].matches_ratio
    max_white_value = 0.
    white_key=""
    for key,value in b1.iteritems():
        #get max
        if value>max_white_value:
            max_white_value=value
            white_key=key
        
    white_response_time=df[df.index==white_key].second
    white_response_time = white_response_time[white_response_time.keys()[0]]
    #print white_response_time
    #white_response_time.strftime('%Y-%m-%d %H:%M:%S')
    #get time of that, t1
    #t1-t0 - set white timedelta
    white_delta = (datetime.datetime.strptime(
                    white_response_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    '%Y-%m-%d %H:%M:%S.%f')-t0).total_seconds()#time in sec
    #print white_delta
    
    ###########
    #black response
    max_black = np.array(df_group_avg.avg_ratio).max()
    #a=df_group_avg[df_group_avg.avg_ratio==max_black].datetime1
    for key,value in df_group_avg[df_group_avg.avg_ratio==max_black].datetime1.iteritems():
        time_black= key 
    #get second, get max from this second
    b2 = df[df.second==time_black].avg_ratio
    max_black_value = 0.
    black_key=""
    for key,value in b2.iteritems():
        #get max
        if value>max_black_value:
            max_black_value=value
            black_key=key
        
    black_response_time=df[df.index==black_key].second
    black_response_time = black_response_time[black_response_time.keys()[0]]
    #print white_response_time
    #white_response_time.strftime('%Y-%m-%d %H:%M:%S')
    #get time of that, t1
    #t1-t0 - set white timedelta
    black_delta = (datetime.datetime.strptime(
                    black_response_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    '%Y-%m-%d %H:%M:%S.%f')-t0).total_seconds()#time in sec
    
    #print black_delta

    return {'white_response_time':white_delta, 'black_response_time':black_delta}
