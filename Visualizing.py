#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:23:57 2022

@author: timalph
"""
import numpy as np
from libsvm.svmutil import svm_train
from scipy.io import loadmat
from sklearn import svm
import pandas as pd
from scipy import sparse
import pickle
from skimage import io
from train import VisualEntityDetectors
import Utils
import assignNN
import PedroPascal
import train

def collateAllDetectors(dets):
    
    numDets = np.zeros(len(dets)).astype(int)
    for i in range(len(dets)):
        det = dets[i]
        numDets[i] = det.firstLevModels['w'].shape[0]
    
    totalNumDets = sum(numDets)
    featLength = dets[0].firstLevModels['w'].shape[1]
    params = dets[0].params
    w = np.zeros((totalNumDets, featLength))
    rho = np.zeros(totalNumDets)
    firstLabel = np.zeros(totalNumDets)
    info = [{} for x in range(totalNumDets)]
    threshold = np.zeros(totalNumDets)
    firstInd = 0
    allParams = []
    
    for i in range(len(dets)):
        det = dets[i]
        num = det.firstLevModels['w'].shape[0]
        w[firstInd:firstInd + num] = det.firstLevModels['w']
        rho[firstInd:firstInd + num] = det.firstLevModels['rho']
        firstLabel[firstInd:firstInd + num] = det.firstLevModels['firstLabel']
        for j in range(num):
            info[j] = det.firstLevModels['info'][j]
        threshold[firstInd:firstInd + num] = det.firstLevModels['threshold']
        allParams.append(det.params)
        firstInd = firstInd + num
        
    params['category'] = ''
    collatedDetector = VisualEntityDetectors([], params)
    collatedDetector.firstLevModels['w'] = w
    collatedDetector.firstLevModels['rho'] = rho
    collatedDetector.firstLevModels['firstLabel'] = firstLabel
    collatedDetector.firstLevModels['info'] = info
    collatedDetector.firstLevModels['threshold'] = threshold
    
    return collatedDetector, numDets, allParams

##:)))
## Unreadable function that I think will take less time to copy than to understand
def readdetsimple(maxdetector, ds, ctthresh=0, conf = {'oneperim' : 0,
                                                         'issingle' : 0,
                                                         'nperdet'  : 100}):
    topn = {} 
    posCount = {}
    posCtIds = {0:{}, 1:[]}
    
    uncell = 0
    nouts = 2
    
    issingle = conf['issingle']
    oneperim = conf['oneperim']
    nperdet = conf['nperdet']
    
    posCount = {}
    negCount = []
    
    for i in range(nouts):
        topn[i] = [pd.DataFrame()] * maxdetector
        posCount[i] = []
        posCtIds[i] = []
        
    detsbydetector = {0:{}, 1:{}}
    locstouched = {}
    imgs = ds['imgs'][ds['conf']['currimset']]
    
    for i in range(len(ds['myiminds'])):
        mydets = ds['detsimple'][i]
        
        if len(mydets) != 0:
            
            curCount = np.zeros(maxdetector)
            loc = ds['ispos'][ds['myiminds'][i]]
        
            if issingle:
                loc = min(loc,1)
                
            if not loc:
                loc = nouts
            
            
            loc -= 1
            
            locstouched[loc] = 1

            if oneperim:
                mydets = mydets.sort_values(by=['decision'], ascending=False)
                mydets = mydets.iloc[np.unique(mydets['detector'].to_numpy(),
                                               return_index=True)[1]]
            #print(mydets)
            for j in range(len(mydets)):
                if mydets.iloc[j]['decision'] > ctthresh:
                    curCount[mydets.iloc[j]['detector']]+=1                                      
                try:
                    detsbydetector[loc][mydets.iloc[j]['detector']] = detsbydetector[loc][mydets.iloc[j]['detector']].append(pd.DataFrame(mydets.iloc[j]).transpose())
                except KeyError:
                    detsbydetector[loc][mydets.iloc[j]['detector']] = pd.DataFrame(mydets.iloc[j]).transpose()

            if ds['ispos'][ds['myiminds'][i]]:
                posCount[loc].append(curCount)
                posCtIds[loc].append(i)
            else:
                negCount.append(curCount)
                posCount[loc].append(curCount)
                posCtIds[loc].append(i)
        else:
            
            mydets = pd.DataFrame()
            curCount = np.zeros(maxdetector)
            loc = ds['ispos'][ds['myiminds'][i]]
        
            if issingle:
                loc = min(loc,1)
                
            if not loc:
                loc = nouts
            
            
            loc -= 1
            
            if ds['ispos'][ds['myiminds'][i]]:
                posCount[loc].append(curCount)
                posCtIds[loc].append(i)
            else:
                negCount.append(curCount)
                posCount[loc].append(curCount)
                posCtIds[loc].append(i)
            
        #if i%100 == 0:
        #    for m in locstouched.keys():
        #        topn[m] = maxkcombine(topn[m], detsbydetector[m], nperdet)
        #    detsbydetector = {0:{}, 1:{}}
        #    locstouched = {}
        if i%500 == 0:
            print(i)
        
        
    for m in locstouched.keys():
        topn[m] = maxkcombine(topn[m], detsbydetector[m], nperdet)
    
    minthresh = {}
    #print(topn)
    for j in range(maxdetector):
        currdecs = np.array([])
        
        for i in range(len(topn)):
            if len(topn[i][j]) != 0:
                tmpdecs = topn[i][j]['decision'].to_numpy()
                currdecs = np.concatenate((currdecs, tmpdecs))
        if len(currdecs) > 0:
            currdecs[::-1].sort()
            minthresh[j] = currdecs[-1]
        else:
            minthresh[j] = -1
    
    for k,v in topn.items():
        topn[k] = pd.concat(v)
        
    return topn, posCount, negCount, posCtIds


def maxkcombine(x, y, nperdet):
    
    for i in range(len(x)):      
        x[i] = x[i].append(y[i]).sort_values(['detector']).iloc[:nperdet]
        
    
    return x

def findOverlapping(detsByDetr, conf={}, nargout = 3):
    
    percentForOverlap = 0.3
    maxOverlapping = 5
    numOtherDetrs = 1
    findNonOverlap = False
    maxToFind = np.inf
    mustload = isinstance(detsByDetr, str)
    currgroup = 1
    groups = np.zeros(len(detsByDetr))
    detrIdForPos = [int(det['detector'].unique()) for det in detsByDetr]
    
    if nargout == 3:
        ovlmat = np.zeros((len(detsByDetr), len(detsByDetr)))
    else:
        raise ValueError("not implemented")
    
    if conf:
        if 'maxToFind' in conf:
            maxToFind = conf['maxToFind']
        if 'findNonOverlap' in conf:
            findNonOverlap = conf['findNonOverlap']
        if 'range' in conf:
            range_ = conf['range']
            
        else:
            range_ = list(range(len(detsByDetr)))   
        
        
    resPrevDets = []
    resPrevDetsAll = []
    res = []
    
    totaloverlapByDetr = []
    
    for k, i in enumerate(range_):
   #for k, i in enumerate(range_[:5]):

        if mustload:
            raise ValueError("check if implemented right")
            currDets = Utils.dsload('detsByDetr', i)
        else:
            currDets = detsByDetr[i]
        
        if not 'imidx' in currDets:
            currDets = assignNN.simplifydets(currDets)
            
        totaloverlapByDetr = np.zeros(i+1)
        totaloverlapByDetrAll = np.zeros(i+1)
        overlapFlag = 1
        for j in range(len(currDets)):
       #for j in range(25,len(currDets)):
            #print(currDets['imidx'].iloc[j] + 1 > len(resPrevDets))
            #print(k, i)
            #print(j)
            #print('max', currDets['imidx'].iloc[j])
            #print('len', len(resPrevDetsAll))

            if currDets['imidx'].iloc[j] + 1 > len(resPrevDets):
                
                resPrevDets += [pd.DataFrame()] * (currDets['imidx'].iloc[j] + 1 - len(resPrevDets))
                resPrevDetsAll += [pd.DataFrame()] * (currDets['imidx'].iloc[j] + 1 - len(resPrevDetsAll))
            
            compDets = resPrevDets[currDets['imidx'].iloc[j]]
            
            if len(compDets) == 0:
                continue

            boxesi = assignNN.getBoxesForPedro([currDets['pos'].iloc[j]])
            boxesj = assignNN.getBoxesForPedro(compDets['pos'])
            overl = PedroPascal.computePascalOverlap(boxesj, boxesi)
            detinds = compDets.detector.to_numpy()
            
            if len(detinds) == 1:
                toup = np.unique(detinds[overl.squeeze()>percentForOverlap]).astype(int)
            else:                
                detinds = compDets.detector.to_numpy()
                toup = np.unique(detinds[np.where(overl>percentForOverlap)[0]]).astype(int)
            
            if not len(toup) == 0:
                if len(totaloverlapByDetr) - 1 < max(toup):
                    totaloverlapByDetr = np.concatenate((totaloverlapByDetr, np.zeros(1 + max(toup)-len(totaloverlapByDetr))))

                totaloverlapByDetr[toup] +=1

                    
            if nargout==3:
                compDetsAll = resPrevDetsAll[currDets['imidx'].iloc[j]]
                boxesj = assignNN.getBoxesForPedro(compDetsAll['pos'])
                overlAll = PedroPascal.computePascalOverlap(boxesj, boxesi)
                detindsAll = compDetsAll['detector'].to_numpy()
                toupAll = np.unique(detindsAll[np.where(overlAll>percentForOverlap)[0]]).astype(int)
                
                if len(toupAll) > 0:
                    if len(totaloverlapByDetrAll)-1 < max(toupAll):
                        totaloverlapByDetrAll = np.concatenate((totaloverlapByDetrAll, np.zeros(1 + max(toupAll)-len(totaloverlapByDetrAll))))

                    totaloverlapByDetrAll[toupAll] += 1
            
            
            if sum(totaloverlapByDetr[toup] > maxOverlapping) >=numOtherDetrs:
                overlapFlag = 0
                
                if nargout < 3:
                    break
            
        for j in range(len(currDets)):   
            resPrevDetsAll[currDets['imidx'].iloc[j]] = resPrevDetsAll[currDets['imidx'].iloc[j]].append(currDets.iloc[j])
        
        if overlapFlag:
            for j in range(len(currDets)):
                resPrevDets[currDets['imidx'].iloc[j]] = resPrevDets[currDets['imidx'].iloc[j]].append(currDets.iloc[j])
            
            if findNonOverlap:
                res.append(i)
                print("Found {} nonoverlap".format(i+1))
            
                if len(res) >= maxToFind:
                    nextClust = k+1
                    print('searched {} detectors, found {} nonoverlap'.format(k,len(res)))
                    raise ValueError("delete this line")
                    #return res, groups, ovlmat
            
            groups[i] = currgroup
            currgroup +=1
                    
        else:
            if not findNonOverlap:
                raise NotImplementedError
            tmp = np.where(totaloverlapByDetr>maxOverlapping)[0]
            groups[i] = groups[np.where(tmp[0]==detrIdForPos)[0]]
            
        if nargout == 3:
            ## This is not used in the original script.
            ovlmat[i, :i] =  totaloverlapByDetrAll[:i]
            ovlmat[:i, i] = totaloverlapByDetrAll[:i].T
        
        if k%10 == 0:
            print('findOverlapping: {} / {}'.format(k, len(range_)))
    return np.array(res), groups, ovlmat


def selectDetectors(dets, inds):
    
    flm = dets.firstLevModels
    
    detscpy = train.VisualEntityDetectors({}, dets.params)
    
    
    detscpy.firstLevModels['w'] = flm['w'][inds]
    detscpy.firstLevModels['rho'] = flm['rho'][inds]
    
    detscpy.firstLevModels['firstLabel'] = flm['firstLabel'][inds]
    
    detscpy.firstLevModels['info'] = np.array(flm['info'])[inds]
    detscpy.firstLevModels['threshold'] = flm['threshold'][inds]
    
    
    return detscpy
    
def dispres_discpatch(ds):
    
    html='<table>'
    
    relpath='.'
    drdp_split=0
    
    if 'splitflag' in ds['bestbin']:
       drdp_split=1 
       relpath = '../'
      
    if 'alldisclabel' in ds['bestbin']:
        ds['bestbin']['alldisclabelcat'] = ds['bestbin']['alldisclabel']
    
    if len(ds['bestbin']['alldisclabelcat']) == 0:
        ds['bestbin']['alldisclabelcat'] = np.zeros(5)
        
    mytosave = ds['bestbin']['tosave'][np.where(ds['bestbin']['isgeneral'])[0]]
    htmlall = {}
    drdp_ct=-1
    for drdp_j in mytosave:
        drdp_ct = drdp_ct+1
        html += htmlpatchrow(ds, drdp_j, relpath)
        
        if drdp_split and drdp_ct%20 == 0:
            raise NotImplementedError
            
    html += '</table>'
    
    if drdp_split:
        raise NotImplementedError
    
    else:
        ds['bestbin']['bbhtml'] = html
        
    return ds
    
        
def htmlpatchrow(ds, binid, patchpath, style=''):
    html = '<tr>'       
    
    if not 'lsh' in ds:
        ds['lsh'] = {}
    if not 'kmn' in ds:
        ds['kmn'] = {}
        
    if not 'isgeneral' in ds['bestbin']:
        ds['bestbin']['isgeneral'] = np.ones(len(ds['bestbin']['tosave']))
        
        ds['bestbin'][np.where(ds['bestbin']['isgeneral'])[0]]

    pos = np.where(ds['bestbin']['tosave'][np.where(ds['bestbin']['isgeneral'])[0]]==binid)[0]
    
    if ds['lsh']:
        raise NotImplementedError
    elif ds['kmn']:
        raise NotImplementedError
    elif 'counts' in ds['bestbin']:
         counts = ds['bestbin']['counts'][pos].squeeze()
    else:
        counts = np.array([])
    
    if 'imgcounts' in ds['bestbin']:
        raise NotImplementedError
        
    else:
        imgcountstr = ''
        
    if 'gain' in ds['bestbin']:
        raise NotImplementedError
    
    else:
        entstr = ''
    
    if 'misclabel' in ds['bestbin']:
        raise NotImplementedError
        
    else:
        miscstr = ''
        
        
    if 'detweight' in ds['bestbin']:
        raise NotImplementedError
        
    else:
        weightstr = ''
    
    corresplink = ''
    
    if 'corresphtml' in ds['bestbin']:
        raise NotImplementedError
        
    if 'svmweight' in ds['bestbin']:
        raise NotImplementedError
    
    else:
        svmweightstr = ''
        
    if len(counts) > 0:
        countstr = ' paris:{} non:{}'.format(counts[0], counts[1]) 
    else:
        countstr = ''
        
        
    pos = np.where(ds['bestbin']['tosave'][np.where(ds['bestbin']['isgeneral'])[0]]==binid)[0]
    
    html+=  '<td class="postd" style="' + style + '" ><img src="" style="width:201px;height:0px"/> ' + str(int(pos)) + ':' + str(binid) + '<br/> ' + countstr + imgcountstr + entstr + ' ' + weightstr + ' ' + svmweightstr + ' ' + miscstr + '</td>\n'
    
    if 'coredetection' in ds['bestbin']:
        raise NotImplementedError
    
    patches = np.where(ds['bestbin']['alldisclabelcat'][:,1] == binid)[0]
    
    if 'decision' in ds['bestbin']:
        order = np.argsort(ds['bestbin']['decision'][patches])[::-1]
        patches = patches[order]
        
    if 'group' in ds['bestbin']:
        raise NotImplementedError
        
        
    if len(patches) > 20 and 'group' not in ds['bestbin']:
        patches = patches[:20]
        
    for patch in patches:
        label=''
        
        if 'decision' in ds['bestbin']:
            label = label + 'score: ' + str(ds['bestbin']['decision'][patch])
        
        if 'rank' in ds['bestbin']:
            raise NotImplementedError
            
        link = ''
            
        if 'detectvishtml' in ds['bestbin']:
            raise NotImplementedError
            
        if 'imgsurl' in ds['bestbin']:
            raise NotImplementedError
                    
        style = ''
        
        if 'iscorrect' in ds['bestbin']:
            raise NotImplementedError
            
        if 'group' in ds['bestbin']:
            raise NotImplementedError
        
        imstyle = 'width:80px;height:80px'
        fname = patchpath + '/alldiscpatchimg[]/' + str(patch)
        html += imgtd(fname, label, link, style, imstyle) + '\n'

        if 'alldiscpatchlabimg' in ds['bestbin']:
            raise NotImplementedError
    html += '</tr>\n'
    return html

def imgtd(fname, label, link='', style='', imstyle=''):
    if not dshassuffix(fname.lower(), '.jpg'):   
        fname += '.jpg'
    
    if label:
        labelstr = ' title= {}'.format(label)
    
    else:
        labelstr=''
        
    link1 = ''
    link2 = ''
    
    if link:
        link1 = '<a style="border:solid 0px #000" href="' + link + '">'
        link2 = '</a>'
        
    return '<td class="imgtd" style={}>{}<img style="{}" src="{}"{}/>{}</td>'.format(style,link1,imstyle,fname,labelstr,link2)

def dshassuffix(string, suffix):
    
    if len(suffix) > len(string):
        return False
    
    return string[-len(suffix):] == suffix

    
    
    