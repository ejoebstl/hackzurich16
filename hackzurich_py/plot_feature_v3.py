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

calculate_time=False

filepath = '/Users/gabrielfior/Dropbox/Hackzurich16/pupils_cutout_6/out_left_wb7.txt'
filepath2 = '/Users/gabrielfior/Dropbox/Hackzurich16/pupils_cutout_6/out_wb7.txt'

names1=['index','window_name','ms','bestLargex','bestLargey','bestLarger1',
                        'BestLarger2','bestLargematches','bestLargerejects',
                        'bestLargeavg','bestSmallmatches','bestSmallrejects',
                        'bestSmallavg']
names2=['index','window_name','bestLargex','bestLargey','bestLarger1',
                        'BestLarger2','bestLargematches','bestLargerejects',
                        'bestLargeavg','bestSmallmatches','bestSmallrejects',
                        'bestSmallavg']


df = pd.read_csv(filepath,
                 names=names1)
df2 = pd.read_csv(filepath2,
                 names=names1)


t0 = df.ms[0]
t0 = datetime.datetime.fromtimestamp(float(t0)/1000.).strftime('%Y-%m-%d %H:%M:%S.%f')
t0=datetime.datetime.strptime(t0, '%Y-%m-%d %H:%M:%S.%f')

t02 = df2.ms[0]
t02 = datetime.datetime.fromtimestamp(float(t02)/1000.).strftime('%Y-%m-%d %H:%M:%S.%f')
t02=datetime.datetime.strptime(t02, '%Y-%m-%d %H:%M:%S.%f')


print 'here'
df['datetime1']=df.apply(aux.convert_datetime,args=(t0,),axis=1)
df['second']=df.apply(aux.extract_second,axis=1)
df['date2']=df.apply(aux.get_date_with_mili,axis=1)

df2['datetime1']=df2.apply(aux.convert_datetime,args=(t02,),axis=1)
df2['second']=df2.apply(aux.extract_second,axis=1)
df2['date2']=df2.apply(aux.get_date_with_mili,axis=1)


print 'here2'
#df = df[df.index not in list_final]


#Delete from both files where best large matches =0
first_list= list(df[df.bestLargematches == 0].index.values)
second_list = list(df2[df2.bestLargematches == 0].index.values)
list_final = first_list + list(set(second_list) - set(first_list))
df.drop(df.index[list_final])
df2.drop(df2.index[list_final])

df['radius_ratio'] = df.bestLarger1/df.BestLarger2
df['avg_ratio']= df.bestSmallavg/df.bestLargeavg
df['matches_ratio']= df.bestLargematches/df.bestSmallmatches

df2['radius_ratio'] = df2.bestLarger1/df2.BestLarger2
df2['avg_ratio']= df2.bestSmallavg/df2.bestLargeavg
df2['matches_ratio']= df2.bestLargematches/df2.bestSmallmatches


#Drop lines where matched points=0
#plt.figure(1)
#plt.clf()
#plt.subplot(311)
#plt.title('avg ratio')
#df.avg_ratio.plot()
#plt.subplot(312)
#plt.title('matches ratio')
#df.matches_ratio.plot()
#plt.subplot(313)
#plt.title('radius ratio')
#df.radius_ratio.plot()
#plt.show()

#plt.figure(2)
#plt.clf()
#plt.title('matches ratio')
#df.plot(kind='scatter',x='ms',y='matches_ratio')

############
#Group by second
df_group_median = df.groupby('second').median()
df_group_sum = df.groupby('second').sum()
df_group_avg = df.groupby('second').mean()

df2_group_median = df2.groupby('second').median()
df2_group_sum = df2.groupby('second').sum()
df2_group_avg = df2.groupby('second').mean()

'''
plt.figure(55)
plt.subplot(331)
plt.plot(np.array(df_group_median.matches_ratio),label='median, matches_rat')
plt.plot(np.array(df2_group_median.matches_ratio),label='median, matches_rat')

plt.subplot(332)
plt.plot(np.array(df_group_median.radius_ratio),label='median,radius_rat')
plt.plot(np.array(df2_group_median.radius_ratio),label='median,radius_rat')

plt.subplot(333)
plt.title('33')
plt.plot(np.array(df_group_median.avg_ratio),label='median,avg_rat')
plt.plot(np.array(df2_group_median.avg_ratio),label='median,avg_rat')

plt.subplot(334)
plt.plot(np.array(df_group_sum.matches_ratio),label='sum, matches_rat')
plt.plot(np.array(df2_group_sum.matches_ratio),label='sum, matches_rat')

plt.subplot(335)
plt.plot(np.array(df_group_sum.radius_ratio),label='sum, radius_rat')
plt.plot(np.array(df2_group_sum.radius_ratio),label='sum, radius_rat')

plt.subplot(336)
plt.plot(np.array(df_group_sum.avg_ratio),label='sum, avg_rat')
plt.plot(np.array(df2_group_sum.avg_ratio),label='sum, avg_rat')

plt.subplot(337)
plt.title('7')
plt.plot(np.array(df_group_avg.matches_ratio),label='avg, matches_rat')
plt.plot(np.array(df2_group_avg.matches_ratio),label='avg, matches_rat')

plt.subplot(338)
plt.title('8')
plt.plot(np.array(df_group_avg.radius_ratio),label='avg, radius_rat')
plt.plot(np.array(df2_group_avg.radius_ratio),label='avg, radius_rat')

plt.subplot(339)
plt.plot(np.array(df_group_avg.avg_ratio),label='avg, avg_rat')
plt.plot(np.array(df2_group_avg.avg_ratio),label='avg, avg_rat')
'''
plt.show()

#################################################
plt.figure(62)
plt.clf()

plt.subplot(311)
plt.title('Average intensity pupil')
plt.plot(np.array(df_group_median.avg_ratio),label='median,avg_rat')
plt.plot(np.array(df2_group_median.avg_ratio),label='median,avg_rat')

plt.subplot(312)
plt.title('Number of pixels matched')
plt.plot(np.array(df_group_avg.matches_ratio),label='avg, matches_rat')
plt.plot(np.array(df2_group_avg.matches_ratio),label='avg, matches_rat')

plt.subplot(313)
plt.title('Pupil radius')
plt.plot(np.array(df_group_avg.radius_ratio),label='avg, radius_rat')
plt.plot(np.array(df2_group_avg.radius_ratio),label='avg, radius_rat')

###########################################################
#Obtain diffs, 
#exclude where diff >2,
# get average
df3_avg_matches = abs(df_group_avg.matches_ratio-df2_group_avg.matches_ratio)
df3_avg_matches = df3_avg_matches[df3_avg_matches<2]
print str(df3_avg_matches.mean())
#plt.plot(df3,'r',label='diff')
df3_median_matches=abs(df_group_median.matches_ratio-df2_group_median.matches_ratio)
df3_median_matches = df3_median_matches[df3_median_matches<2]
print str(df3_median_matches.mean())

df3_sum_matches=abs(df_group_sum.matches_ratio-df2_group_sum.matches_ratio)
df3_sum_matches = df3_sum_matches[df3_sum_matches<2]
print  str(df3_sum_matches.mean())

df3_avg_avg=abs(df_group_avg.avg_ratio-df2_group_avg.avg_ratio)
df3_avg_avg = df3_avg_avg[df3_avg_avg<2]
print  str(df3_avg_avg.mean())

df3_median_avg=abs(df_group_median.avg_ratio-df2_group_median.avg_ratio)
df3_median_avg = df3_median_avg[df3_median_avg<2]
print  str(df3_median_avg.mean())

df3_sum_avg=abs(df_group_sum.avg_ratio-df2_group_sum.avg_ratio)
df3_sum_avg = df3_sum_avg[df3_sum_avg<2]
print  str(df3_sum_avg.mean())

df3_avg_rad=abs(df_group_avg.radius_ratio-df2_group_avg.radius_ratio)
df3_avg_rad = df3_avg_rad[df3_avg_rad<2]
print  str(df3_avg_rad.mean())

df3_median_rad=abs(df_group_median.radius_ratio-df2_group_median.radius_ratio)
df3_median_rad = df3_median_rad[df3_median_rad<2]
print  str(df3_median_rad.mean())

df3_sum_rad=abs(df_group_sum.radius_ratio-df2_group_sum.radius_ratio)
df3_sum_rad = df3_sum_rad[df3_sum_rad<2]
print  str(df3_sum_rad.mean())







if calculate_time:
    ######################################################
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
    print white_delta
    
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
    
    print black_delta
    #########
