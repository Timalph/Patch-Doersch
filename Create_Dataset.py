#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:20:11 2022

@author: timalph
"""

## Build dataframe takes a list of positive neighbourhoods and a list of 
## negative neighbourhoods and builds a dataframe with the city column as 
## 'positivecity' or 'negativecity'.
import pandas as pd
import os
from PIL import Image
import numpy as np
import time
### dirname is the name of the directory containing the cutouts.

def build_dataframe(poslist, neglist, dirname,
                    sampling = {},
                    base_dir = '/Users/timalph/Documents/Panoramas/processed'):
    np.random.seed(20)
    start = time.time()
    print("building dataframe for ", poslist, neglist)
    df = pd.DataFrame(columns = ['id', 'bytes', 'city', 'fullname', 'imsize'])
    
    if sampling:
        print("Randomly sampling {} positive and {} negative images".format(sampling['pos'],
                                                                            sampling['neg']))
    
    
    for pos in poslist:
        path = os.path.join(base_dir, dirname, 'cutouts', pos)
        list_of_images = os.listdir(path)
        
        ## if the sampling flag has a dict which specifies the amount of pos
        ## and neg images then randomly sample them here.
        if sampling:
            no_of_pos_imgs = sampling['pos']
            
            index = np.random.permutation(np.arange(len(list_of_images)))
            
            
            posindex = index[:no_of_pos_imgs]
            
            list_of_images = list(np.array(list_of_images)[posindex])
        
        for p in list_of_images:
            fullpath = os.path.join(path, p)
            img = Image.open(fullpath)
            
            tmp_df = pd.DataFrame({'id' : p, 
                       'bytes' : os.stat(fullpath).st_size,
                       'city' : 'positivecity',
                       'fullname' : os.path.join(pos, p),
                       'imsize' : [list((img.size[1], img.size[0]))]})
            
            df = df.append(tmp_df)
    print('We have {} pos images'.format(len(df)))
    for neg in neglist:
        path = os.path.join(base_dir, dirname, 'cutouts', neg)
        list_of_images = os.listdir(path)
        
        
            
        
        if sampling:
            
            no_of_neg_imgs = sampling['neg']
            if neg == neglist[-1]:
                amount = (no_of_neg_imgs*len(neglist)) - len(df.loc[df['city']=='negativecity'])
            else:
                amount = no_of_neg_imgs
            index = np.random.permutation(np.arange(len(list_of_images)))
            negindex = index[:amount]
            list_of_images = list(np.array(list_of_images)[negindex])
            
        for p in list_of_images:
            fullpath = os.path.join(path, p)
            img = Image.open(fullpath)
            
            tmp_df = pd.DataFrame({'id' : p, 
                       'bytes' : os.stat(fullpath).st_size,
                       'city' : 'negativecity',
                       'fullname' : os.path.join(neg, p),
                       'imsize' : [list((img.size[1], img.size[0]))]})
            
            df = df.append(tmp_df)
    
    print('Images sampled in {}'.format(time.time() - start))
    
    return df.reset_index(drop=True)


def script_setup(out_dir, path_out, dataframe, root_dir):
    
    
    ds = {}
    ds['raw_outdir'] = out_dir
    ds['prediction_model'] = 'SVC'
    ds['sys'] = {'outdir' : path_out}
    ds['imgs'] = {}
    pos = len(ds['imgs']) +1
    ds['imgs'][pos] = dataframe
    ds['conf'] = {}
    ds['conf']['gbz'] = {}
    ds['conf']['gbz'][pos] = {'cutoutdir': root_dir, 'imgsurl' : ''}
    ds['conf']['currimset'] = pos
    num_train_its=5
    
    np.random.seed(20)
    
    ds['conf']['params'] = {'imageCanonicalSize': 400, 'patchCanonicalSize': np.array([80,80]),
                            'scaleIntervals': 8, 'sBins': 8, 'useColor': 1,
                            'patchOverlapThreshold': 0.6}
    
    ds['conf']['params']['svmflags']= '-s 0 -t 0 -c 0.1'  
    
    ds['conf']['detectionParams'] = {'selectTopN': False, 'useDecisionThresh': True,
                                   'overlap': 0.4, 'fixedDecisionThresh': -1.002}
    
    imgs = dataframe
    ds['mycity'] = 'positivecity'
        
    
    
    return ds

