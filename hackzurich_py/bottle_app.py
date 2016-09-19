# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 23:10:31 2016

@author: gabrielfior
"""

from bottle import route, run, template, response
from json import dumps
from plot_feature_for_api import give_responses
import datetime
import pandas as pd


@route('/pupai/<name>')
def index(name):

    url_list = [{'target': 'http://10.58.48.103:5000/', 'clicks': '1'}, 
            {'target': 'http://slash.org', 'clicks': '4'},
            {'target': 'http://10.58.48.58:5000/', 'clicks': '1'},
            {'target': 'http://de.com/a', 'clicks': '0'}]
    rv = [{ "id": 1, "name": "Test Item 1" }, { "id": 2, "name": "Test Item 2" }]
    response.content_type = 'application/json'
    #return template('<b>Hello {{name}}</b>!', name=name)
    
    filepath = '/Users/gabrielfior/Dropbox/Hackzurich16/pupils_cutout_6/out_wb7.txt'
    names1=['index','window_name','ms','bestLargex','bestLargey','bestLarger1',
                        'BestLarger2','bestLargematches','bestLargerejects',
                        'bestLargeavg','bestSmallmatches','bestSmallrejects',
                        'bestSmallavg']
    df = pd.read_csv(filepath,names=names1)
    
    return dumps(give_responses(df))
    
run(host='localhost', port=8080)



