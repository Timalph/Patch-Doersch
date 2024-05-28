#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 14:25:15 2021

@author: timalph
"""
import numpy as np
import pandas as pd
### This Image_object class can hold all the subfunctions we are using such as
### Patch canonical size and what not. 
def ind2sub(siz, ndx):
    # returns -> [I,J] = IND2SUB(SIZ,IND)
    vi = np.remainder(ndx-1, siz[0]) + 1
    v2 = (ndx-vi)/siz[0]
    return vi.astype('int'), v2.astype('int')


### Use this to turn the output of detections that goes into simplifydets into a pandas structure.
### Possibly could be used for more, but not right now
def struct2pandas(s, topndets_variation = False, all_imgs_variation=False, 
                  metadata_variation=False, no_print=False):
    
    s, l = splitstruct(s)
    d = {}
    
    if topndets_variation:
        s = s[0]
    
    
    for idx, i in enumerate(s):
        row = []
        
        if not metadata_variation:
        
            for j in i[0]:
                if len(str(j.dtype)) > 15:
    
                    _, b = splitstruct(j)
                    
                    dt = {}
                    for m in range(len(b)):
                        dt[b[m]] = int(j[0][0][m])
                    row.append(dt)
                    
                else:
                    if j[0].dtype == '<U111' or j[0].dtype == '<U110' or str(j.dtype)[:2] == '<U':
                        row.append(str(j[0]))
                    elif j[0].dtype == 'float64':
                        if len(j[0]) != 1:
                            row.append(np.array(j[0]))
                        else:
                            row.append(float(j[0]))
                    elif len(j[0]) > 1:
                        row.append(j[0].astype('int'))
                    elif j[0].dtype == 'uint8' or 'uint16':
                        row.append(int(j[0]))
        
        else:
            for j in i:
    
                if len(str(j.dtype)) > 15:

                    _, b = splitstruct(j)
                    
                    dt = {}
                    for m in range(len(b)):
                        dt[b[m]] = int(j[0][0][m])
                    row.append(dt)
                    
                else:
                    if j.dtype == '<U111' or j.dtype == '<U110' or str(j.dtype)[:2] == '<U':
                        row.append(str(j))
                    elif j.dtype == 'float64':
                        try:
                            row.append(float(j))
                        except TypeError:
                            if not no_print:
                                print('Assuming empty image is passed. NaN value is stored.')
                            row.append(np.nan)
                    elif len(j) > 1:
                        row.append(j.astype('int'))
                    elif j.dtype == 'uint8' or 'uint16':
                        try:
                            row.append(int(j))
                        except TypeError:
    
                            row.append(list(j[0]))
                
        
        d[idx] = row
    
    
    df = pd.DataFrame.from_dict(d, orient='index', columns = l)
    return df

### Specific instance used for parsing topdetsmap
def unpack_topdetsmap(s):
    
    s, l = splitstruct(s)
    d = {}
    s = s[0][0]
    
    d['scores'] = s[0][0]
    d['imgIds'] = s[1][0]
    d['meta'] = struct2pandas(s[2], 
                              topndets_variation=True, 
                              metadata_variation=True).to_dict('index')

    return d
    
def unpack_matlabTOPN(s):
    
    s, l = splitstruct(s)
    d = {}
    s = s[0][0]
    
    d['scores'] = s[1].flatten()
    d['imgIds'] = s[2].flatten()
    d['meta'] = struct2pandas(s[0]).to_dict('index')

    return d

def matdouble(s):

    l = []
    
    storage = buffer()
    
    for c in s:
        print(c[0])
        print(type(c[0]))
        try:
            i = int(c)
            storage.add(i)
            
        except ValueError:
            if len(storage.content) > 0:
                l.append(storage.empty())
        
    return l 

class buffer:
    
    def __init__(self):
        self.content = []
    
    def add(self, a):
        self.content.append(a)
        
    def empty(self):
        
        self.content.reverse()
        
        num = 0
        
        for idx, i in enumerate(self.content):
            num += (10**idx)*i
        self.content = []
        
        return num

def splitstruct(s):
    details = str(s[0].dtype)
    details = details[1:].split(' ')
    l=[]
    for idx, i in enumerate(details):
        if idx%2 ==0:
            l.append(i[2:-2])
    return s, l


def sumlistofdf(x):
    summation = 0
    for df in x:
        for patch in df['pos']:
            summation += sum(np.array(list(patch.values()))+1)

        print(summation)
    return summation


def matsort(x, descending=False):
    
    if not descending:
        raise ValueError("Check if it's implemented right for ascending order")
        
    return (len(x) - np.argsort(x[::-1]))[::-1]-1
    
    