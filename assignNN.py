#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:25:53 2021

@author: timalph
"""

#As told by Carl
# Author: Carl Doersch (cdoersch at cs dot cmu dot edu)
#
# For each element in ds.centers, find the nearest neighbor in
# the image specified by ds.myiminds(dsidx).  Just store the
# index for each element of centers to avoid communication. 
# The root node will aggregate across images and extract
# descriptors later.




import numpy as np
import Utils
from skimage import io
import pandas as pd
import math
import PedroPascal
import pickle
from itertools import compress
from numba import jit, prange
from time import time
import train
import numba
import faiss
import torch
#iminds = np.array([16, 13, 10,  4,  1, 14, 17, 12, 19,  5,  8, 18,  0,  2,  7,  6, 11,
#        9, 15,  3, 90, 94, 22, 64, 76, 68, 32, 56, 20, 69, 96, 29, 54, 40,
#       25, 97, 43, 83, 60, 58, 57, 71, 55, 85, 51, 41, 21, 48, 89, 39, 59,
#       84, 92, 80, 66, 28, 91, 47, 24, 67, 42, 61, 73, 70, 86, 50, 62, 72,
#       53, 75, 82, 44, 34, 88, 65, 49, 74, 31, 27, 79, 98, 35, 87, 36, 63,
#       37, 93, 38, 99, 30, 52, 77, 81, 23, 45, 78, 95, 33, 46, 26])
#@jit(nopython=True)
def main(ds, Faiss=False):
    
    gpu = torch.cuda.is_available()
    
    if Faiss:
        d = ds['centers'].shape[1]

        index = faiss.IndexFlatL2(d)
        #nlist= int(min(ds['centers'].shape[0]/40, 1000))
        #index = faiss.IndexIVFFlat(index, d, nlist)
        #print(nlist)
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("Running on GPU!")
            
        except AttributeError:
            print("Running on CPU!")
        
        #print("Training Faiss Index with {} Voronoi Cells...".format(nlist))
        start = time()
        #index.train(np.single(ds['centers']))
        print("Trained in", time()-start)
    
    
    for idx, i in enumerate(ds['myiminds']):
        #start = time()
        print("{}/{}".format(idx, len(ds['myiminds'])))
        im = Utils.getimg(ds, i)
        start = time()
        if Faiss:
            ds = NN_for_single_image(im, ds, idx, gpu=gpu, index=index)
            
        if not Faiss:
            ds = NN_for_single_image(im, ds, idx, gpu=gpu)
        print("step {} took {} seconds!".format(idx, time()-start))
    return ds

def NN_for_single_image(im, ds, idx, testing = False, index=False, gpu=False, k=1):
    
    pyramid = Utils.constructFeaturePyramidForImg(im, ds['conf']['params'])
        
    features, levels, indexes, gradsums = Utils.unentanglePyramid(pyramid, ds['conf']['params'])
    
    thresh = gradsums>=9
    toss = len(gradsums) - sum(thresh)
    
    features = features[thresh]
    levels = levels[thresh]
    indexes = indexes[thresh]
    gradsums = gradsums[thresh]
    
    print('threw out ', toss, 'patches!' )
    
    features = Utils.getcenters(features)
    
    if index:
        dist, assignedidx = index.search(np.single(features), k)
    
    if not index:
        if gpu:
            assignedidx, dist = assigntoclosest_gpu(np.single(ds['centers']), np.single(features))
        else:    
            assignedidx, dist = assigntoclosest(np.single(ds['centers']), np.single(features))
    
    try:
        ds['assignednn'][idx]=dist
    except:
        ds['assignednn'] = {}
        ds['assignednn'][idx]=dist
    
    lev = levels[assignedidx]
    index = indexes[assignedidx].squeeze(axis=1)
    
    ass_idx = np.concatenate((lev,index), axis=1)
    
    try:
        ds['assignedidx'][idx] = ass_idx.astype('int')
    except:
        ds['assignedidx'] = {}
        ds['assignedidx'][idx] = ass_idx.astype('int')
    
    try:
        ds['pyrscales'][idx] = pyramid['scales']
    except:
        ds['pyrscales'] = {}
        ds['pyrscales'][idx] = pyramid['scales']
        
    try:
        ds['pyrcanosz'][idx] = pyramid['canonicalScale']   
    except: 
        ds['pyrcanosz'] = {}
        ds['pyrcanosz'][idx] = pyramid['canonicalScale'] 
    
    if testing:
        return features, levels, indexes, gradsums, assignedidx
    
    return ds

def assigntoclosest_gpu(toassign, targets, step = 800, gpu=True):
    
    targets = torch.tensor(targets)
    toassign = torch.tensor(toassign)
    
    closest = torch.zeros((toassign.shape[0],1), dtype=torch.int64)
    outdist = torch.zeros((toassign.shape[0],1))
    
    if gpu:
        print('Running on GPU!')
        gpu = torch.cuda.current_device()
        print(gpu)
        targets.to(gpu)
        closest.to(gpu)
        outdist.to(gpu)
        toassign.to(gpu)
    center_size = toassign.shape[0]
    
    targsq = torch.sum(targets**2, axis=1, keepdims=True)
    
    steps = np.arange(0, center_size, step)
    
    start = time()
    print("Calculating inproduct...")
    
    for i in steps:
        batch = toassign[i:i+step, :]
        batchsq = torch.sum(batch**2, axis=1)
        inprod = torch.mm(targets, batch.T)
        dist = (batchsq.T - (inprod*2)) + targsq
        
        values, indices = torch.min(dist, axis=0)
        
        closest[i:i+step] = indices.reshape(indices.shape[0],1)
        outdist[i:i+step] = values.reshape(values.shape[0],1)
    
    print(time() - start)
    return closest.numpy(), outdist.numpy()


@jit(nopython=True)
def assigntoclosest_jit(toassign, targets, step = 800):
    
    center_size = toassign.shape[0]

    targsq = np.expand_dims(np.sum(targets**2, axis=1), axis=1)
    
    steps = np.arange(0, center_size, step)
    closest = np.zeros((toassign.shape[0],1), dtype=numba.int32)
    outdist = np.zeros((toassign.shape[0],1))

    for i in steps:
        inds = np.arange(min(i+step, center_size))
        batch = toassign[inds, :]
        batchsq = np.sum(batch**2, axis=1)
        inprod = np.dot(targets, batch.T)
        
        dist = (batchsq.T - (inprod*2)) + targsq
        ###train.quicksave(dist)
        closest[inds] = np.expand_dims(numba_argmin_0(dist), axis=1)
        outdist[inds] = np.expand_dims(numba_min_0(dist), axis=1)
    return closest, outdist

def assigntoclosest(toassign, targets, step = 800):
    
    center_size = toassign.shape[0]

    targsq = np.expand_dims(np.sum(targets**2, axis=1), axis=1)
    
    steps = np.arange(0, center_size, step)
    closest = np.zeros((toassign.shape[0],1), dtype=np.int32)
    outdist = np.zeros((toassign.shape[0],1))

    for i in steps:
        print(i)
        inds = np.arange(min(i+step, center_size))
        batch = toassign[inds, :]
        batchsq = np.sum(batch**2, axis=1)
        inprod = np.dot(targets, batch.T)
        
        dist = (batchsq.T - (inprod*2)) + targsq
        #train.quicksave(dist)
        closest[inds] = np.expand_dims(numba_argmin_0(dist), axis=1)
        outdist[inds] = np.expand_dims(numba_min_0(dist), axis=1)
    return closest, outdist


@jit(nopython=True)
def numba_argmin_0(x):
    res = np.zeros(x.shape[1], dtype=numba.int32)
    for i, row in enumerate(x.T):
        res[i] = np.argmin(row)
    return res
@jit(nopython=True)
def numba_min_0(x):
    res = np.zeros(x.shape[1])
    for i, row in enumerate(x.T):
        res[i] = np.min(row)
    return res
def simplifydets(detections, imidx, assignedClust=False, configuration=False, feature_array=False):
    
    ### If configuration is True, run the first configuration of this function.
    ### As used in line 155 of autoclust_main.
    if configuration == 'First':
        
        decision = []
        pos = []
        mydetector = []
        currimidx = []
        
        
        for idx, (clust, iidx) in enumerate(zip(assignedClust, imidx)):
            row = detections.iloc[idx]
            p = {}
            p['x1'] = row['x1'] 
            p['x2'] = row['x2']
            p['y1'] = row['y1']
            p['y2'] = row['y2']
            
            decision.append(0)
            mydetector.append(clust)
            currimidx.append(iidx)
            pos.append(p)
        
        res = pd.DataFrame({'decision' : decision, 'pos' : pos, 
                          'imidx' : imidx, 'detector' : mydetector})
    
    elif configuration == 'Non-empty decisions':
        decision = []
        pos = []
        mydetector = []
        currimidx = []
        
        
        for idx, (clust, iidx) in enumerate(zip(assignedClust, imidx)):
            row = detections.iloc[idx]
            p = {}
            p['x1'] = row['x1'] 
            p['x2'] = row['x2']
            p['y1'] = row['y1']
            p['y2'] = row['y2']
            
            decision.append(row['detScore'])
            mydetector.append(clust)
            currimidx.append(iidx)
            pos.append(p)
        
        res = pd.DataFrame({'decision' : decision, 'pos' : pos, 
                          'imidx' : imidx, 'detector' : mydetector})
    
    else:
        
        decisions = []
        posses = []
        mydetectors = []
        currimidxes = []
        features = []
        
        md = detections['firstLevel']['detections']['metadata']
        for i in range(len(md.keys())):
            for j in range(len(md[i].keys())):
                pos = {'x1':md[i][j]['x1'],
                       'y1':md[i][j]['y1'],
                       'x2':md[i][j]['x2'],
                       'y2':md[i][j]['y2']}
                ## Line that asks if imidx is even a var. When are we going to need that?
                
                if Utils.numel(imidx) == 1:
                    curimidx=imidx
                else:
                    print(imidx)
                    raise ValueError
                    curimidx=imidx[curridx]
                if 'decision' in detections['firstLevel']['detections']:
                    decision = detections['firstLevel']['detections']['decision'][i][j]
                elif 'detScore' in detections['firstLevel']['detections']:
                    raise ValueError('not implemented yet!')
                elif 'scores' in detections['firstLevel']['detections']:
                    raise ValueError('not implemented yet!')
                else:
                    decision = 0
                
                if 'clust' in md[i][j]:
                    raise ValueError('not implemented yet!')
                    
                try:
                    assignedClust[curridx]
                except TypeError:
                    mydetector=i
                
                if ('features' in detections['firstLevel']['detections']):
                    if len((detections['firstLevel']['detections']['features'].keys())) > 0:
                        
                        decisions.append(decision)
                        mydetectors.append(mydetector)
                        currimidxes.append(imidx)
                        posses.append(pos)
                        features.append(detections['firstLevel']['detections']['features'][i][j])
                        
                        
                        #d = {'decision' : decision, 'pos' : pos, 
                        #      'imidx' : imidx, 'detector' : mydetector,
                        #      'features' : detections['firstLevel']['detections']['features'][i][j]}
                        
                        #with open('temp.pickle', 'wb') as handle:
                        #    pickle.dump(d, handle)
                        
                        #res = pd.DataFrame.from_dict({'decision' : decision, 'pos' : pos, 
                        #      'imidx' : imidx, 'detector' : mydetector,
                        #      'features' : detections['firstLevel']['detections']['features'][i][j]})
                    else:
                        res = pd.DataFrame.from_dict({'decision' : decision, 'pos' : pos, 
                              'imidx' : imidx, 'detector' : mydetector})
                        
                        decisions.append(decision)
                        mydetectors.append(mydetector)
                        currimidxes.append(imidx)
                        posses.append(pos)
                        
                else:
                    #res = pd.DataFrame.from_dict({'decision' : decision, 'pos' : pos, 
                    #          'imidx' : imidx, 'detector' : mydetector})
                    
                    decisions.append(decision)
                    mydetectors.append(mydetector)
                    currimidxes.append(imidx)
                    posses.append(pos)
                    
                    
        try:
            res = pd.DataFrame({'decision' : decisions, 'pos' : posses, 
                      'imidx' : currimidxes, 'detector' : mydetectors, 
                      'features' : features})
        except:
            res = pd.DataFrame({'decision' : decisions, 'pos' : posses, 
                      'imidx' : currimidxes, 'detector' : mydetectors})
    if not feature_array:
        return res
    elif feature_array:
        return res, np.array(features)



def create_topn(ds, npatches, assignednn, nneighbors=100, test = False, assignedidx = False):
    
    shape1 = len(ds['assignednn'][0])
    
    topndist = np.zeros((shape1, nneighbors))
    topnlab = np.zeros((shape1, nneighbors))
    topnidx = np.zeros((shape1, nneighbors, 4))

    
    for j in range(npatches-1, -1, -1):
        dists=[]    
        order = np.argsort(assignednn[j])[:nneighbors]
        topndist[j] = assignednn[j][order]
        
        for i in range(len(order)-1, -1, -1):
            topnlab[j,i] = ds['ispos'][ds['myiminds'][order[i]]]
            topnidx[j,i, 0] = order[i] 
            if test:
                topnidx[j,i, 1:] = assignedidx[order[i]][j] -1  
            else:
                topnidx[j,i, 1:] = ds['assignedidx'][order[i]][j]
                #Code below was to check whether non square patches already existed previous to this function
# =============================================================================
#             imgidx = order[i]
#             _ ,pos = pyridx2pos(topnidx[j,i,2:], ds['pyrcanosz'][imgidx],
#                 ds['pyrscales'][imgidx][int(topnidx[j,i,1])],8,8,8,[537,936])
#             if not is_square(pos):
#                 print(abs((pos['x2']-pos['x1'])-(pos['y2']-pos['y1'])))
# =============================================================================
    
    
    return topndist, topnlab, topnidx.astype('int')


curridx=1
selClustIdx=1
mainflag=1
topndets={}
topndetshalf={}
topndetstrain={}
topnorig=[]
newselclust=[]

## Create topndetshalf

def create_topndetshalf(ds, postord, topnidx, topndist, mainflag, nneighbors=100, test=False):
    
    pyridx_adjust = False
    curridx=0
    selClustIdx=0
    mainflag=1
    topndets={}
    topndetshalf=[]
    topndetstrain=[]
    newselclust=[]
    topnorig = pd.DataFrame()
    if test:
        
        postord, topnidx, topndist, dsmyiminds, dspyrcanosz, dspyrscales, dsselectedClust, dsperclustpost, dsinitPatches = test
        
        postord -=1 
        
        topnidx -=1
        
        dsselectedClust -= 1
        
        pyridx_adjust = True
        
        
    for i in postord:
        if mainflag:
            curdet_decision = np.zeros(nneighbors)
            curdet_pos = []
            curdet_imidx = []
            curdet_detector = [] 
            curdet_pos_clustertest = []
            
            patch_size_x = ds['conf']['params']['patchCanonicalSize'][0]/ds['conf']['params']['sBins']-2
            patch_size_y = ds['conf']['params']['patchCanonicalSize'][1]/ds['conf']['params']['sBins']-2
            bin_size = ds['conf']['params']['sBins']
            for j in range(nneighbors):
                imgidx = topnidx[i, j, 0]
 
                pos_df, pos = pyridx2pos(topnidx[i,j,2:], ds['pyrcanosz'][imgidx],
                ds['pyrscales'][imgidx][topnidx[i,j,1]],
                patch_size_x,
                patch_size_y,
                bin_size, ds['imgs'][ds['conf']['currimset']].iloc[ds['myiminds'][imgidx]]['imsize'], test=pyridx_adjust)
                
                
                #if pos['x1'] == 1 and pos['x2'] == 177 and pos['y1'] == 197:
                #    print(pos)
                
# =============================================================================
#                     with open('notsquaretest_candelete.pickle', 'wb') as h:
#                         pickle.dump((topnidx[i,j,2:], ds['pyrcanosz'][imgidx],
#                     ds['pyrscales'][imgidx][topnidx[i,j,1]],
#                     patch_size_x,
#                     patch_size_y,
#                     bin_size, ds['imgs'][ds['conf']['currimset']].iloc[ds['myiminds'][imgidx]]['imsize']), h)
# =============================================================================
                
                
                
                #try:
                #    curdet['decision'].append(-topndist[i,j])
                #except:
                #    curdet['decision'] = -topndist[i,j]
                #print('square?')
                #print(pos['x2'] - pos['x1'])
                #print(pos['y2'] - pos['y1'])
                
                
                curdet_decision[j] = -topndist[i,j]
                curdet_pos.append(pos)
                curdet_pos_clustertest.append(pos_df)
                
                
                if not test:
                    curdet_detector.append(ds['selectedClust'][selClustIdx])
                    curdet_imidx.append(ds['myiminds'][imgidx])
                else:
                    curdet_detector.append(dsselectedClust[selClustIdx])
                    curdet_imidx.append(int(dsmyiminds[imgidx]))
                curridx += 1
                
                #try:
                #    curdet['pos'].append(pos, ignore_index=True)
                
                #except:
                #    pass
            #curdet_pos = pd.concat(curdet_pos).reset_index(drop=True)
            ####### Reject near-duplicate patches: when spatial overlap of more than 30% between any 5 of their top 50 NN.
            ####### We use curdet[:25] because we only have 35 current detections

            if mainflag:
                
# =============================================================================
#                 if selClustIdx == 0:
#                     with open('safetodelete0.pickle', 'wb') as handle:
#                         pickle.dump((topndetshalf, 
#                                                      curdet_decision[:25],
#                                                      curdet_pos_clustertest[:25],
#                                                      curdet_imidx[:25],
#                                                      curdet_detector[:25]), handle)
#                 elif selClustIdx == 2:   
#                     with open('safetodelete2.pickle', 'wb') as handle:
#                         pickle.dump((topndetshalf, 
#                                                      curdet_decision[:25],
#                                                      curdet_pos_clustertest[:25],
#                                                      curdet_imidx[:25],
#                                                      curdet_detector[:25]), handle)
# =============================================================================
                    
                tmpmainflag = testclusteroverlap(topndetshalf, 
                                                 curdet_decision[:50],
                                                 curdet_pos_clustertest[:50],
                                                 curdet_imidx[:50],
                                                 curdet_detector[:50])
                
            if not test:
                
                origpatind=np.where(ds['assignedClust']==ds['selectedClust'][selClustIdx])[0]
                origdet = ds['initPatches'].iloc[origpatind].copy()

                origdet['decision'] = 0
                detector = ds['selectedClust'][selClustIdx]
                origdet['detector'] = detector
                origdet['count'] = ds['perclustpost'][selClustIdx]
                #origdet['pos'] = [pos.to_dict('index')[0]]
                origdet['pos'] = [{'x1' : int(origdet['x1']),
                                  'x2' : int(origdet['x2']),
                                  'y1' : int(origdet['y1']),
                                  'y2' : int(origdet['y2'])}]
            else:
                origpatind=np.where(ds['assignedClust']==dsselectedClust[selClustIdx])[0]
                origdet = dsinitPatches.iloc[origpatind].copy()

                origdet['decision'] = 0
                detector = dsselectedClust[selClustIdx]
                origdet['detector'] = detector
                origdet['count'] = dsperclustpost[selClustIdx]
                #origdet['pos'] = [pos.to_dict('index')[0]]
                origdet['pos'] = [{'x1' : int(origdet['x1']),
                                  'x2' : int(origdet['x2']),
                                  'y1' : int(origdet['y1']),
                                  'y2' : int(origdet['y2'])}]
                
                
   
            if tmpmainflag:
                if len(topnorig)<1200:
                    curdet_t = pd.DataFrame(list(zip(curdet_decision, 
                                                     curdet_pos, curdet_imidx, 
                                                     curdet_detector, curdet_pos_clustertest)), 
                                            columns=['decision', 
                                                     'pos', 'imidx', 
                                                     'detector', 'pos_df'])
                    #display
                    #topndets=[topndets;{curdet([1:10 15:7:35])}];%for display
                    

                    topndetshalf.append(curdet_t[:50])
                    topndetstrain.append(curdet_t[:5])
                    

                    topnorig = topnorig.append(origdet)
                    

                print('we now have ', len(newselclust), ' topnorig.')
                newselclust.append(detector)
                if len(newselclust)>=1200:
                    mainflag=0
                tmpmainflag=0
            
        selClustIdx += 1
        print(selClustIdx,  '/' ,len(postord))
    
    topnorig = topnorig[['decision', 'pos', 'imidx', 'detector', 'count']].reset_index(drop=True)
    return topndets, drop_posdf(topndetshalf), drop_posdf(topndetstrain), topnorig, newselclust

def drop_posdf(x):
    return [y.drop('pos_df', axis=1) for y in x]


def testclusteroverlap(topN, decision, pos, imidx, detector):
    
    if len(topN) == 0:
        return True
    
    else:

        coccurOverlap = np.zeros((len(topN), 10))
        keep = True
        
        for i in range(len(topN)):
            imgIds = topN[i]['imidx']
            qimgIds = imidx
            inter = np.intersect1d(imgIds, qimgIds)
            
            for x in inter:
                memi = np.where(imgIds == x, 1, 0)
                memj = np.where(np.array(qimgIds) == x, 1, 0)
                boxesi = getBoxesForPedro(topN[i].iloc[np.where(memi)[0]]['pos_df'])
                
                #temp = [memj, pos]
                
                #with open('tmp.p', 'wb') as f:
                #    pickle.dump(temp, f)
                
                boxesj = getBoxesForPedro(list(compress(pos,memj)))
                
                overl = PedroPascal.computePascalOverlap(boxesj, boxesi)

                overl = np.max(overl, axis=1)
                overlInd = np.ceil(overl * 10)
                overlInd = np.where(overlInd==0, 1, overlInd)
                ## bins from 1 to 10
                h, _ = np.histogram(overlInd, bins=np.arange(11)+1)
    

                coccurOverlap[i,:] += h
                
                if sum(coccurOverlap[i, 2:]) > (.1*len(decision)):
                    keep = False
                    return keep
                
                
        
    return keep



def pyridx2pos(idx, pyrcanosz, pyrsc, prSize, pcSize, sBins, imsize, printing=False, test=False):
    metadata = {}
    levSc = pyrsc
    canoSc = pyrcanosz
    levelPatch = np.array([0, round((pcSize + 2) * sBins * levSc / canoSc) - 1,
                  0, round((prSize + 2) * sBins * levSc / canoSc) - 1])
    y1, x1 = idx
    xoffset = math.floor((x1) * sBins * levSc / canoSc)
    yoffset = math.floor((y1) * sBins * levSc / canoSc)
    thisPatch = levelPatch + np.array([xoffset, xoffset, yoffset, yoffset])
    
    if test:
        thisPatch +=1
    
    
    if printing:
        print('levelPatch: ', levelPatch)
        print(idx)
        print(xoffset, yoffset)
        print(thisPatch)
        
    metadata['x1'] = max(0, thisPatch[0])
    metadata['x2'] = min(thisPatch[1], imsize[1])
    metadata['y1'] = max(0, thisPatch[2])
    metadata['y2'] = min(thisPatch[3], imsize[0])   
    

    return pd.DataFrame(metadata, index=[0]), metadata
    #return metadata
    
def getBoxesForPedro(data, decisionScore=False):
    
    try:
        if not decisionScore:
            decisionScore = np.zeros(len(data))
    except:
        pass
    boxes = np.zeros((len(data), 5))
    
    ## Running into issues because the MATLAB code doesn't specify when functions
    ## are used for different datatypes. Using a try except to catch the 
    ## typeerror for now. Might fix this later.
    

    ### Data is a dataframe, or perhaps even a     
    try:
        for k, row in enumerate(data):
            r = [row['x1'], row['y1'], row['x2'], row['y2']]
            row = list(map(int, r))
            boxes[k, :4] = row
            boxes[k, 4] = decisionScore[k]
    ## TypeError is raised when dat is a dict.
        
    except TypeError:
        for k in data.keys():
            row = data[k]
            r = [row['x1'], row['y1'], row['x2'], row['y2']]
            row = list(map(int, r))
            boxes[k, :4] = row
            boxes[k, 4] = decisionScore[k]
            
        
    ### Boxes must be a numpy array
    return boxes
    
    
def is_square(pos):
    
    if pos['x2'] - pos['x1'] == pos['y2'] - pos['y1']:
        return True
    else:
        return False
    
    
