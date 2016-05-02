'''
Created on 02/mag/2016

@author: giovanni
'''
import pandas as pd
import numpy as np
import urllib2
import datetime as dt
import matplotlib.pyplot as plt
 
def get_google_data(symbol, period, window):
    url_root = 'http://www.google.com/finance/getprices?i='
    url_root += str(period) + '&p=' + str(window)
    url_root += 'd&f=d,o,h,l,c,v&df=cpct&q=' + symbol
    response = urllib2.urlopen(url_root)
    data = response.read().split('\n')
    #actual data starts at index = 7
    #first line contains full timestamp,
    #every other line is offset of period from timestamp
    parsed_data = []
    anchor_stamp = ''
    end = len(data)
    for i in range(7, end):
        cdata = data[i].split(',')
        if 'a' in cdata[0]:
            #first one record anchor timestamp
            anchor_stamp = cdata[0].replace('a', '')
            cts = int(anchor_stamp)
        else:
            try:
                coffset = int(cdata[0])
                cts = int(anchor_stamp) + (coffset * period)
                parsed_data.append((dt.datetime.fromtimestamp(float(cts)), float(cdata[1]), float(cdata[2]), float(cdata[3]), float(cdata[4]), float(cdata[5])))
            except:
                pass # for time zone offsets thrown into data
    df = pd.DataFrame(parsed_data)
    df.columns = ['ts', 'o', 'h', 'l', 'c', 'v']
    df.index = df.ts
    del df['ts']
    return df

def get_spread(base, hedge, ratio, period, window):
    b = get_google_data(base, period, window)
    h = get_google_data(hedge, period, window)
    combo = pd.merge(pd.DataFrame(b.c), pd.DataFrame(h.c), left_index = True, right_index = True, how = 'outer')
    combo = combo.fillna(method = 'ffill')
    combo['spread'] = combo.ix[:,0] + ratio * combo.ix[:,1]
    return(combo)

