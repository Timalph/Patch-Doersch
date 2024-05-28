#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 11:25:37 2022

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
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import time
import Utils
import assignNN
import PedroPascal

from numba import jit

def initialize(ds, modeltype = 'SVM'):
    """
    

    Parameters
    ----------
    ds : Global Datastruct
    modeltype : TYPE, optional
        DESCRIPTION. The default is 'SVM'.
        Other possibility is 'SVC' which is scikit implementation of linear SVC. 
    Returns
    -------
    ds : Datastruct with saved models.

    """
    
    for dsidx, clustId in enumerate(ds['batch']['round']['selectedClust']):
        print(dsidx)
        posInds = ds['batch']['round']['assignedClust'] == clustId

        labels = np.concatenate(((np.ones(sum(posInds))), np.ones(ds['initFeatsNeg'].shape[0])*-1))
        features = np.concatenate((ds['batch']['round']['posFeatures'][posInds], ds['initFeatsNeg']), axis=0)
        #workspace = loadmat('svm_input')
        
        #labels = workspace['labels'].squeeze()
        #features = workspace['features']
        
        if modeltype == 'SVM':
            print('Training LIBSVM ... ')
            model = mySvmTrain(labels, features, ds['conf']['params']['svmflags'], False)
            predictedLabels, accuracy, decision = mySvmPredict(labels, features, model)
        
        elif modeltype == 'SVC':
            print('Training SVC ...')
            model = mySvmTrain(labels, features, ds['conf']['params']['svmflags'], False, modeltype='SVC')
            predictedLabels, accuracy, decision = mySvmPredict(labels, features, model)
        
            
            
            #model = LinearSVC(C=0.1, verbose=1)
            #model.fit(features, labels)
            
            #predictedLabels, accuracy, decision = model_predict(labels, features, model)
            
            
        elif modeltype == 'SGD':
            print('Training SGD ...')
            model = SGDClassifier()
            model.fit(features, labels)
            predictedLabels, accuracy, decision = model_predict(labels, features, model)
           
        try:
            ds['batch']['round']['firstDet'][dsidx] = model
        except:
            ds['batch']['round']['firstDet'] = {}
            ds['batch']['round']['firstDet'][dsidx] = model
        try:    
            ds['batch']['round']['firstResult'][dsidx] = {'predictedLabels' : predictedLabels,
                                                      'accuracy' : accuracy,
                                                      'decision' : decision}
        except:
            ds['batch']['round']['firstResult'] = {}
            ds['batch']['round']['firstResult'][dsidx] = {'predictedLabels' : predictedLabels,
                                                      'accuracy' : accuracy,
                                                      'decision' : decision}
            
    return ds

def model_predict(labels, features, model):
    
    predictedlabels = model.predict(features)
    
    accuracy = len(np.where(labels==predictedlabels)[0])/len(labels) * 100
    decision = model.decision_function(features)
    return predictedlabels, accuracy, decision


def mySvmTrain(labels, features, flags, fullModel, modeltype='SVM'):
    
    if modeltype == 'SVM':
        orgModel = svm_train(labels, features, flags)
    
    elif modeltype == 'SVC':
        orgModel = LinearSVC(C=0.1, verbose=1)
        orgModel.fit(features, labels)
        
    model = {}
    model['model'] = orgModel
    
    model = getMinimalModel(model, modeltype)
    model['info'] = {}
    model['info']['numPositives'] = sum(labels==1)
    model['info']['numNegatives'] = len(labels) - model['info']['numPositives']
    #model['info']['nSV'] = orgModel.nSV
    model['threshold'] = 0
    model.pop('model')
    if fullModel:
        model['info']['model'] = orgModel
    
    return model

def mySvmPredict(labels, features, model, decision_only=True):
    
    if 'model_svc' in model:
        return 0, 0, predict_composite_svc(labels, features, model, decision_only)
    
    elif isinstance(model, type(LinearSVC())):
        return 0, 0
    
    elif 'type' in model:
        
        if model['type'] == 'composite':
            predictedLabels, accuracy, decision = doMyPredictComposite(labels, features, model)
            
        else:
            raise ValueError("Not implemented yet")
    
    elif not 'firstLabel' in model:
        raise ValueError("Not implemented yet")
    
    else:
        predictedLabels, accuracy, decision  = doMyPredictMinimal(labels,
                                                           features,
                                                           model)        
        
    
    return predictedLabels, accuracy, decision


def predict_composite_svc(labels, features, model, decision_only=True):
    
    numfeat = features.shape[0]
    nummodels = len(model['model_svc'])    
    
    all_decisions = np.zeros((nummodels, numfeat))
    
    if decision_only:
        for i in range(nummodels):
            all_decisions[i] = model['model_svc'][i].decision_function(features)
        
    else:
        raise NotImplementedError
    
    
    return all_decisions


def doMyPredictMinimal(labels, features, model):
    
    b = np.tile(model['rho'], (1, features.shape[0]))
    w = model['w']    
    
    predictedLabels, accuracy, decision = doPrediction(w, b, 
                                                       features, labels, 
                                                       model['firstLabel'])
    
    return predictedLabels, accuracy, decision

def doMyPredictComposite(labels, features, model, fast=True):
    
    if fast:
        return doPredictionComposite(np.single(model['w']), 
                                     np.single(model['rho']), 
                                     np.single(features), 
                                 labels, model['firstLabel'])
    
    return doPredictionComposite(model['w'], model['rho'], features, 
                                 labels, model['firstLabel'])
@jit(nopython=True)
def doPrediction(w, b, features, labels, modelFirstLabel):

    decision = features @ w - b
    decision = decision * modelFirstLabel
    predictedLabels = np.sign(decision)
    
    predictedLabels = np.where(predictedLabels == 0, -1, predictedLabels)
    
    numCorrect = np.sum(predictedLabels == labels)
    
    accuracy = numCorrect / len(labels)
    accuracy *= 100
    
    return predictedLabels, accuracy, decision

@jit(nopython=True)
def doPredictionComposite(W, B, features, labels, modelFirstLabel):
    
    
    #### I think that W is coef_ and B is -intercept when running linearSVC
    
    ## Is accuracy supposed to be a vector of length 5 also?
    numFeats = features.shape[0]
    decision = np.dot(features, W.T) - B
    decision *= modelFirstLabel
    predictedLabels = np.sign(decision)
    
    predictedLabels = np.where(predictedLabels == 0, -1, predictedLabels)
    
    numCorrect = np.sum(np.equal(predictedLabels,labels))

    accuracy = numCorrect / numFeats
    accuracy *= 100
    
    return predictedLabels, accuracy, decision


    
    

def getMinimalModel(model, modeltype):
    
    
    minModel = {}

    if modeltype == 'SVM':
    
        suppVec = model['model'].get_SV()
        
        if len(suppVec) != 0:
            suppVec = np.nan_to_num(unpack_SV(suppVec))
            coeff = model['model'].get_sv_coef()
            coeff = np.array(coeff)
            
            coeff = np.tile(coeff, (1, suppVec.shape[1]))
            
            
            minModel['coeff'] = coeff
            minModel['rho'] = model['model'].rho[0]
            # this w does not seem to be the same as the matlab w
            wpre = coeff * suppVec
            minModel['wpre'] = wpre
            minModel['w'] = np.sum(wpre, axis=0)
            minModel['firstLabel'] = model['model'].get_labels()[0]
            minModel['model'] = model['model']
        else:
            suppVec = np.zeros((0,2112))
            coeff = model['model'].get_sv_coef()
            coeff = np.array(coeff)
            coeff = np.tile(coeff, (1, suppVec.shape[1]))
            minModel['coeff'] = coeff
            minModel['rho'] = model['model'].rho[0]
        # this w does not seem to be the same as the matlab w
            minModel['w'] = np.zeros((0,2112))
            minModel['firstLabel'] = model['model'].get_labels()[0]
            minModel['model'] = model['model']
    
    if modeltype == 'SVC':
        
        minModel['w'] = model['model'].coef_.squeeze()
        minModel['rho'] = -model['model'].intercept_
        minModel['firstLabel'] = model['model'].classes_[1]
        minModel['model'] = model['model']
    # model info and model threshold not implemented yet
    
    if 'threshold' in model:
        minModel['threshold'] = model['threshold']
    if 'info' in model:
        minModel['info'] = model['info']

    return minModel

# x is a list of dicts
def unpack_SV(x):
    
    # Find shape
    
    ## Can we just turn this second part to 2112?? I'm too tired to evaluate that now
    s = np.zeros((len(x),2112))
    for ind, d in enumerate(x):
        for key in d.keys():
            s[ind,key-1] = d[key]
    
    return s

def constructCompositeDetectors(models):

    if not models:
        newDets = getInitializedNewDets()
        return newDets
    
    if isinstance(models[0], type(LinearSVC())):
        newDets = {}
        newDets['model_svc'] = {}
        for i in range(len(models)):
            newDets['model_svc'][i] = models[i]
        newDets['type'] = 'composite'
    else:
       
        wt = np.zeros((len(models), models[0]['w'].shape[0]))
        rh = np.zeros(len(models))
        fl = np.zeros(len(models))
        info = {}
        th = np.zeros(len(models))
        
        for i in models.keys():
            md = models[i]
            wt[i] = md['w']
            rh[i] = md['rho']
            fl[i] = md['firstLabel']
            info[i] = md['info']
            th[i] = md['threshold']
        
        newDets = {}
        newDets['w'] = wt
        newDets['rho'] = rh
        newDets['firstLabel'] = fl
        newDets['info'] = info
        newDets['threshold'] = th
        newDets['type'] = 'composite'
    
    return newDets

def getInitializedNewDets():
    
    return {'w' : None, 'rho' : None, 'firstLabel' : None,
            'info' : [], 'threshold' : None, 'type' : 'composite'}

class VisualEntityDetectors:
    
    def __init__(self, models, params):
        
        self.firstLevModels = constructCompositeDetectors(models)
        self.voteWeights = np.array(len(models))
        
        self.firstLevMmhtwts = []
        self.secondLevModels = {}
        self.params = params
        self.mixModel = []
        self.logisParams = []
        self.modelWeightFn = []
        
        
    def setMixModel(self, mixModel):
        self.mixModel = mixModel
    
    def setSuccessRatio(self, successRatio):
        self.successRatio = successRatio
    
    def setSecondLevModels(self, models):
        self.secondLevModels = models
    
    def setLogisParams(self, params):
        self.logisParams = params
    
    def setVoteWeights(self, voteWeights):
        self.voteWeights = voteWeights
    
    def setModelWeightFn(self, fn):
        self.modelWeightFn = fn
    
    def getNumDetectors(self):
        
        ## MATLAB implementation contains statement: if iscell(obj.firstLevModels)
        ## Unsure why firstLevModels would ever be a cell, so disregarding that line.
        
        return len(self.firstLevModels['w'])
    
    def detectPresenceInImg(self, im, ds, detectionParams=False):
        
        detectors = {}
        detectors['firstLevel'] = self.firstLevModels
        detectors['secondLevel'] = self.secondLevModels
        
        if 'detectionParams' in ds['conf']:
            detectionParams = ds['conf']['detectionParams']
        

        if not detectionParams:
            detectionParams = {'selectTopN' : False,
                               'useDecisionThresh' : True,
                               'overlap' : 5,
                               'fixedDecisionThresh' : -0.7,
                               'removeFeatures' : 1}
            
        if not 'removeFeatures' in detectionParams:
            detectionParams['removeFeatures'] = 0
        
        results = detectPresenceUsingEntDet(im, detectors, detectionParams, ds)
        
        return results

    def VisualEntityDetectors(model, params):
        
        obj = 3
        
        return obj

    
    
def detectPresenceUsingEntDet(im, detectors, detectionParams, ds):
    
    params = ds['conf']['params']
    
    if params['imageCanonicalSize'] != 400:
        raise ValueError('There is an issue with the canonical size. Figure out where the issue is because this is a bug an not supposed to occur')
    
    pyra = Utils.constructFeaturePyramidForImg(im, params, [])

    detections = getDetectionsForEntDets(detectors['firstLevel'], pyra, params, detectionParams, im, ds)
    
    results = {}
    
    results['firstLevel'] = constructResults(detections, detectionParams['removeFeatures'])
    
    return results

def constructResults(detections, removeFeatures):
    
    if removeFeatures:
        raise ValueError("not implemented yet")
        
    numDet = 0

    for key, val in detections['metadata'].items():
        numDet += len(val)
        
    results = {'numDetections' : numDet, 'detections' : detections,
               'totalProcessed' : detections['totalProcessed'][0]}
    
    
    return results


def getDetectionsForEntDets(detectors, pyramid, params, detectionParams, im, ds):
    
    
    patchCanonicalSize = params['patchCanonicalSize']
    
    prSize = int(np.round(patchCanonicalSize[0] / pyramid['sbins']) - 2)
    pcSize = int(np.round(patchCanonicalSize[1] / pyramid['sbins']) - 2)
    
    features, levels, indexes, gradsums = Utils.unentanglePyramid(pyramid, params)
    
    thresh = gradsums>=9
    toss = len(gradsums) - sum(thresh)
    
    features = features[thresh]
    levels = levels[thresh].astype(int)
    indexes = indexes[thresh]
    gradsums = gradsums[thresh]
    
    print('threw out ', toss, 'patches!' )
    
    totalProcessed = len(features)
    
    if 'w' in detectors:
        labels = np.ones((len(features), len(detectors['w'])))
    elif 'model_svc' in detectors:
        labels = np.ones((len(features), len(detectors['model_svc'])))
    else:
        raise NotImplementedError
    
    
    # Qualitative testing showed less detections than in the MATLAB script
    # 
    _, _, decision = mySvmPredict(labels, features, detectors)
    
    
    selected = doSelectionForParams(detectionParams, decision)
    detections = constructResultStruct(pyramid, prSize, pcSize, totalProcessed,
                                       features, decision, levels, indexes, 
                                       selected, detectionParams, im, ds)
    
    return detections
def constructResultStruct(pyramid, prSize, pcSize, totalProcessed, features, 
                          decision, levels, indexes, selected, 
                          detectionParams, im, ds)  :
    
    detections = {'features': {}, 'metadata' : {}, 'decision' : {}, 
                  'totalProcessed' : {}, 'thresh' : []}
    
    numDets = decision.shape[1]
    j = 0
    for i in range(numDets):
        
        detections['totalProcessed'][i] = totalProcessed
        selInds = np.where(selected[:, i] == True)[0]
        if len(selInds) < 1:
            detections['features'][j] = np.array([])
            detections['metadata'][j] = {}
            detections['decision'][j] = np.array([])
            
        else:
            metadata = Utils.getMetadataForPositives(selInds, levels, indexes, 
                                               prSize, pcSize, pyramid, im, ds, suppress_warning=True)
        
            picks = doNmsForImg(metadata, decision[selInds, i], detectionParams['overlap'])
    
            detections['features'][j] = features[selInds[picks], :]
            detections['metadata'][j] = {}
            for pick in picks:
                detections['metadata'][j][pick] = metadata[pick]
            detections['decision'][j] = decision[selInds[picks], i]
        j += 1
    #detections['metadata'] = {i : v for i, v in enumerate(detections['metadata'].values())}
    for key, dic in detections['metadata'].items():
    
        detections['metadata'][key] = {i : v for i, v in enumerate(dic.values())}
    
    
    
    
    return detections

## workspace = loadmat(matPATH + 'tmpForTestingNms')
## datam = workspace['data']
## datam = struct2pandas(datam, topndets_variation=True, metadata_variation=True)
## datam = datam.to_dict('index')
## decisionScorem = workspace['decisionScore'].squeeze()

def doNmsForImg(data, decisionScore, overlap):
    
    boxes = assignNN.getBoxesForPedro(data, decisionScore)
    picks = myNms(boxes, overlap)
    return picks


## workspace = loadmat(matPATH + 'tmpForTestingNms')
## includes boxes and picks
def myNms(boxes, overlap):
    
    ## Description from the original function:
    ## Non-maximum suppresion
    ## Greedily select high-scoring detections and skip detections that are 
    ## significantly covered by a previously selected detection.
    ## NOTE: This is adapted from Pedro Felzenszwalb's version (nms.m),
    ## but an inner loop has been eliminated to significantly speed it up in 
    ## the case of a large number of boxes.
    
    if boxes.size == 0:
        return []
    
    x1, y1, x2, y2, s = boxes.T
    
    ## Do we need this +1? Im not sure
    area = (x2 - x1+1) * (y2-y1+1)
    I = np.argsort(s)
    vals = s[I]
    
    pick = s*0
    counter = 0
    
    ## Lots of +1 and -1 in this loop. Unsure what it's used for so unsure 
    ## whether to change this or not
    
    while I.size != 0:
        
        i = I[-1]
        pick[counter] = i
        counter += 1

        xx1 = np.where(x1[I[:-1]] < x1[i], x1[i], x1[I[:-1]])
        yy1 = np.where(y1[I[:-1]] < y1[i], y1[i], y1[I[:-1]])
        xx2 = np.where(x2[I[:-1]] > x2[i], x2[i], x2[I[:-1]])
        yy2 = np.where(y2[I[:-1]] > y2[i], y2[i], y2[I[:-1]])
        
        unf_w = xx2-xx1+1
        unf_h = yy2-yy1+1
        
        w = np.where(unf_w > 0, unf_w, 0)        
        h = np.where(unf_h > 0, unf_h, 0)
        
        o = w*h/area[I[:-1]]
        
        del_vals = (o>overlap).flatten()
        del_vals = np.append(del_vals, True)
        
        I = np.delete(I, del_vals)
    
    pick = pick[:counter]
        
    return pick.astype(int)
    
def doSelectionForParams(params, decision):
    
    if params['useDecisionThresh']:
        if 'fixedDecisionThresh' in params:
            thresh = params['fixedDecisionThresh']
        else:
            raise ValueError('Fixed decision thresh needs to be specified')
        
        selected = decision >= thresh
        
    else:
        thresh = 0
        selected = decision >= thresh
        
    return selected




    
def autoclust_mine_negs(ds):
    
    ## dsidx just looping over negmin.iminds
    
    if 'imgs' not in ds:
        raise NameError("dsload not implented")
        
    #im = Utils.getimg(ds, iminds[dsidx])
    
    ### testing with identical image to matlab
    for dsidx, imind in enumerate(ds['batch']['round']['negmin']['iminds']):
        print("mine_negs: {}/{}".format(dsidx+1, len(ds['batch']['round']['negmin']['iminds'])))
        im = Utils.getimg(ds, imind)
        start = time.time()
        dets = ds['batch']['round']['detectors'].detectPresenceInImg(im, ds)
        print("Detected presence in {}".format(time.time()-start))
        selectedClust = ds['batch']['round']['selectedClust']
        
        #print('imind', imind)
        #quicksave(dets, 'temp')
        tmpdets=assignNN.simplifydets(dets, imind)
        
        indstosave=[]
        try:
            ds['batch']['round']['negmin']['detections'][dsidx] = {}
        except KeyError:
            ds['batch']['round']['negmin']['detections'] = {}
            ds['batch']['round']['negmin']['detections'][dsidx] = {}
        for k, clustId in enumerate(selectedClust):
            alldets = pd.DataFrame([])
            if tmpdets.size > 0:
                alldets = tmpdets.loc[tmpdets['detector'] == k]
            if not alldets.empty:
                ds['batch']['round']['negmin']['detections'][dsidx][k] = alldets
                indstosave.append(k)
            
        if len(indstosave) > 0:
            tosave = ds['batch']['round']['negmin']['detections'][dsidx]
            tosave = {k : v for k, v in tosave.items() if k in indstosave}
        
        ##dssave dsbatchroundnegmindetections
            Utils.dssave(ds['raw_outdir'] + '.' + "ds.batch.round.negmin.detections", str(dsidx),  tosave)
        
        try: 
            ds['batch']['round']['negmin']['imageflags'][dsidx] = 1
        except KeyError:
            ds['batch']['round']['negmin']['imageflags'] = {}
            ds['batch']['round']['negmin']['imageflags'][dsidx] = 1
    
    return ds

def autoclust_mine_negs_speed(ds):
    
    ## dsidx just looping over negmin.iminds
    
    if 'imgs' not in ds:
        raise NameError("dsload not implented")
        
    #im = Utils.getimg(ds, iminds[dsidx])
    
    ### testing with identical image to matlab
    neg_array = np.zeros((len(ds['batch']['round']['negmin']['iminds']), len(ds['batch']['round']['selectedClust'])))
    horizontal_stash = {}
    for dsidx, imind in enumerate(ds['batch']['round']['negmin']['iminds']):
        print("mine_negs: {}/{}".format(dsidx+1, len(ds['batch']['round']['negmin']['iminds'])))
        im = Utils.getimg(ds, imind)
        start = time.time()
        dets = ds['batch']['round']['detectors'].detectPresenceInImg(im, ds)
        print("Detected presence in {}".format(time.time()-start))
        selectedClust = ds['batch']['round']['selectedClust']
        
        #print('imind', imind)
        #quicksave(dets, 'temp')
        tmpdets, feat_array = assignNN.simplifydets(dets, imind, feature_array=True)
        
        indstosave=[]

        try:
            ds['batch']['round']['negmin']['detections'][dsidx] = {}
        except KeyError:
            ds['batch']['round']['negmin']['detections'] = {}
            ds['batch']['round']['negmin']['detections'][dsidx] = {}
            
        
        for k, clustId in enumerate(selectedClust):
            alldets = pd.DataFrame([])
            if tmpdets.size > 0:
                
                new_inds = tmpdets['detector'] == k
                
                alldets = tmpdets.loc[new_inds]
                #feat_array = feat_array[np.array([tmpdets['detector'] == k]).squeeze()]
                
            if not alldets.empty:
                ds['batch']['round']['negmin']['detections'][dsidx][k] = alldets
                indstosave.append(k)
                try:
                    horizontal_stash[k] = np.vstack((horizontal_stash[k],
                                                    feat_array[new_inds.to_numpy()]))
                except KeyError:
                    horizontal_stash[k] = np.zeros((0,2112))
                    horizontal_stash[k] = np.vstack((horizontal_stash[k],
                                                    feat_array[new_inds.to_numpy()]))

            
        #if len(indstosave) > 0:
        #    tosave = ds['batch']['round']['negmin']['detections'][dsidx]
        #    tosave = {k : v for k, v in tosave.items() if k in indstosave}
        
        ##dssave dsbatchroundnegmindetections
            #Utils.dssave(ds['raw_outdir'] + '.' + "ds.batch.round.negmin.detections", str(dsidx),  tosave)
        
        #for key in range(ds['batch']['round']['negmin']['detections'].keys())
            
    for key in horizontal_stash.keys():
        tosave = horizontal_stash[key]
        Utils.dssave(ds['raw_outdir'] + '.' + "ds.batch.round.negmin.hstash", str(key),  tosave)
    
    
        try: 
            ds['batch']['round']['negmin']['imageflags'][dsidx] = 1
        except KeyError:
            ds['batch']['round']['negmin']['imageflags'] = {}
            ds['batch']['round']['negmin']['imageflags'][dsidx] = 1
    
    return ds



def autoclust_train_negs(ds):

    
    
    for dsidx, clustId in enumerate(ds['batch']['round']['selectedClust']):
        print("train_negs: {}/{}".format(dsidx+1, len(ds['batch']['round']['selectedClust'])))
        #dsload('ds.batch.round.negmin.iminds')
        
        if not 'iminds' in ds['batch']['round']['negmin']:
            raise KeyError
            
        if not 'assignedClust' in ds['batch']['round']:
            raise KeyError
            
        if not 'posFeatures' in ds['batch']['round']:
            raise KeyError
            
        posInds = ds['batch']['round']['assignedClust'] == clustId
        posFeatures = ds['batch']['round']['posFeatures'][posInds]
        
        iminds = ds['batch']['round']['negmin']['iminds']
    
        alldets = pd.DataFrame()
        
        #for k in range(len(iminds)):
            #print('printing:', dsidx, k)
        #    obj = Utils.dsload(ds['raw_outdir'] + '.' + 'ds.batch.round.negmin.detections', detections = [dsidx, k])
            
            ## MATLAB script has a line of code to identify whether the further 
            ## images contain anything.
            
    
        #    alldets = alldets.append(obj)
        
        alldetfeats = Utils.dsload(ds['raw_outdir'] + '.' + 'ds.batch.round.negmin.hstash.' + str(dsidx))
        #print('totest')
        #assert np.allclose(totest,np.vstack(alldets.features.tolist()))
        
        #prevfeats not encountered yet
        try:
            #prevFeats = Utils.dsload('ds.batch.round.negmin.prevfeats' + str(dsidx))

            prevfeats = ds['batch']['round']['negmin']['prevfeats'][dsidx]
            print('prevfeats_correctly loaded and of length: {}'.format(len(prevfeats)))
            #ds['batch']['round']['negmin']['prevfeats'] = Utils.dsload(ds['raw_outdir'] + '.' + 'ds.batch.round.negmin.prevfeats' + str(dsidx))
        
            if not (len(ds['batch']['round']['negmin']['prevfeats']) >= dsidx-1):
                prevfeats = ds['initFeatsNeg']
            elif len(ds['batch']['round']['negmin']['prevfeats']) == 0:
                prevfeats = ds['initFeatsNeg']    
        except:
            prevfeats = ds['initFeatsNeg']
        
        if len(alldetfeats)>0:

            allnegs = np.vstack((prevfeats, alldetfeats))
        else:
            allnegs = prevfeats
            
         
        features = np.vstack((posFeatures, allnegs))
        
        poslabels = np.ones(len(posFeatures))
        neglabels = np.ones(len(allnegs)) * -1
        
        labels = np.concatenate((poslabels, neglabels))
        
        print('Training SVM...')
        
        model = mySvmTrain(labels, features, 
                           ds['conf']['params']['svmflags'], False, 
                           modeltype=ds['prediction_model'])
        
            
        
        selectedNegs = cullNegatives(allnegs, model, -1.02)
        
        allnegs = allnegs[selectedNegs]
        
        ds['batch']['round']['nextnegmin']['prevfeats'][dsidx] = allnegs
        
        ## MATLAB SAVES THIS BUT I DONT THINK ITS NECESSARY FOR OUR IMPLEMENTATION
        #Utils.dssave(ds['raw_outdir'] + '.' + 'ds.batch.round.nextnegmin.prevfeats', str(dsidx) , allnegs)
        
        try:
            ds['batch']['round']['nextnegmin']['traineddetectors'][dsidx] = model
        except KeyError:
            if 'nextnegmin' in ds['batch']['round']:
                ds['batch']['round']['nextnegmin']['traineddetectors'] = {dsidx:model}
            else:
                raise KeyError
    return ds

def cullNegatives(negFeatures, detectors, cullingThreshold, detInd = False):
    
    if isinstance(detectors, type(LinearSVC())):
        
        _, _, decision = model_predict(np.ones((len(negFeatures), 
                                      1)) * -1, 
                                      negFeatures, detectors)
      
    else:
    
        if 'firstLevModels' in detectors:
            detectors = detectors['firstLevModels']
        
        try:
            w_shape = detectors['w'].shape[1]
        except IndexError:
            w_shape = 1
            
        if w_shape == 2112:
            raise ValueError("highly likely you're taking the wrong dimension")
            
            
        _, _, decision = mySvmPredict(np.ones((len(negFeatures), 
                                      w_shape)) * -1, 
                                      negFeatures, detectors)
    
    if detInd:
        raise ValueError("not implemented yet")
        
    numNegs = len(negFeatures)
    maxSel1 = np.sum(decision >= cullingThreshold)
    maxSel2 = np.sum(decision >= -1) * 3
    
    maxSel = min(maxSel1, maxSel2)
    
    if maxSel > numNegs:
        maxSel = numNegs
    
    sortedInds = np.argsort(decision)[0]
    sortedInds = np.flip(sortedInds)
    
    selectedNegs = np.zeros((negFeatures.shape[0], 1))
    selectedNegs[sortedInds[:maxSel]] = 1
    
    selectedNegs = np.array(selectedNegs, dtype=bool).squeeze()
    
    return selectedNegs

def autoclust_detect(ds, checkpoint=False):

    imgs = ds['imgs']
    
    cdir = ds['conf']['gbz'][ds['conf']['currimset']]['cutoutdir']
    imgframe = imgs[ds['conf']['currimset']]    
    for dsidx, imind in enumerate(ds['batch']['round']['iminds']):
            
        if checkpoint:
            if dsidx < checkpoint:
                continue
        
        start = time.time()
        print("detect: {}/{}".format(dsidx+1, len(ds['batch']['round']['iminds'])))
        im = Utils.getimg(ds, imind)
        #print('detecting...')
        dets = ds['batch']['round']['detectors'].detectPresenceInImg(im, ds)
        #print('adding image metadata...')
        print(time.time()-start)
        for key, val in dets['firstLevel']['detections']['metadata'].items():
            for k, v in val.items():
                v['im'] = cdir + imgframe.iloc[imind]['fullname']

        ds['batch']['round']['tmpdsidx'] = dsidx
        numTopN = 20
        maxOverlap = 0.1
        
        #print('getResultData...')
        indstosave = []

        for i in reversed(range(len(ds['batch']['round']['selectedClust']))):

            clusti = ds['batch']['round']['selectedClust'][i]

            thisScores, imgMeta = getResultData(dets, i, maxOverlap)

            res = {}
            if len(thisScores) > 0:
                res['scores'] = thisScores
                res['imgIds'] = np.ones(thisScores.shape) * imind
                res['meta'] = imgMeta
                
                if 'topdetsmap' not in ds['batch']['round']:
                    ds['batch']['round']['topdetsmap'] = {}
                try:
                    ds['batch']['round']['topdetsmap'][i][dsidx] = res
                except KeyError:
                    ds['batch']['round']['topdetsmap'][i] = {}
                    ds['batch']['round']['topdetsmap'][i][dsidx] = res
                    
                indstosave.append(i)

        #print('saving...')
        if len(indstosave) > 0 and dsidx%100 == 0:
            
            dir2save = {k : v for k, v in ds['batch']['round']['topdetsmap'].items()}
            Utils.dssave(ds['raw_outdir'] + '.' + 'ds.batch.round.topdetsmap', str(0), dir2save)
        try:
            ds['batch']['round']['detectorflags'][dsidx] = 1
        except KeyError:
            ds['batch']['round']['detectorflags'] = {}
            ds['batch']['round']['detectorflags'][dsidx] = 1
            
        end = time.time()-start
        if end > 15:
            print("{} dsidx is hella slow".format(dsidx))
        print(end)
    return ds    





def getResultData(result, clusti, overlap):
    scores = np.array([])
    meta = {}
    
    if len(result['firstLevel']['detections']['features']) <= clusti:
        return scores, meta
    
    thisScores = result['firstLevel']['detections']['decision'][clusti]
    if len(thisScores) == 0:
        return scores, meta
    imgMeta = result['firstLevel']['detections']['metadata'][clusti]
    
    # Do NMS for image
    picks = doNmsForImg(imgMeta, thisScores, overlap)
    
    scores = thisScores[picks]
    
    ind = 0
    for p in picks:
        meta[ind] = imgMeta[p]
        ind +=1

    
    return scores, meta

def autoclust_topn(ds, dsidx, matlab=False):
    
    print('dsidx is:', dsidx)
    
    imgs = ds['imgs']
    ispos = ds['ispos']
    
    numTopN = 20
    maxOverlap = 0.1

    detObj = PresenceDetectionResults2([dsidx], 
                                       len(ds['batch']['round']['selectedClust']), 
                                       ds['batch']['round']['iminds'])
    
    traintopN = getTopNDetsPerCluster2(detObj, 
                                       maxOverlap, 
                                       ds['batch']['round']['totrainon'][ispos[ds['batch']['round']['totrainon']]==1],
                                       numTopN, ds, matlab)
    
    valtopN = getTopNDetsPerCluster2(detObj,
                                    maxOverlap,
                                    ds['batch']['round']['tovalon'][ispos[ds['batch']['round']['tovalon']]==1],
                                    numTopN, ds, matlab)
    
    ### continuing code for autoclustTopN
    for val in valtopN.values():
        if len(np.unique(val['imgIds'])) != len(val['imgIds']):
            raise ValueError('valtopN contains repeats!')
    
    alltopN = getTopNDetsPerCluster2(detObj, maxOverlap, ds['batch']['round']['tovalon'], 100, ds, matlab)
    
    ds['batch']['round']['traintopN'][dsidx] = traintopN[0]
    ds['batch']['round']['validtopN'][dsidx] = valtopN[0]
    ds['batch']['round']['alltopN'][dsidx] = alltopN[0]
    
    return ds
        

class PresenceDetectionResults2():
    
    def __init__(self, idx, numclusters, iminds):

        self.myidx = idx
        self.myiminds = iminds        
        self.mynumclusters = numclusters
        
    def getPosResult(self, idd, ds):
        idx= int(np.where(self.myiminds==idd)[0])
        result = {}
        #dsload(topdetsmap[self.myidx, idx])
        #if size(ds.batch.round.topdetsmap,2) >= idx
        
        for i, myidx in enumerate(self.myidx):
            if (len(ds['batch']['round']['topdetsmap'].keys())>=myidx):
                
                try:
                    if len(ds['batch']['round']['topdetsmap'][myidx][idx]['scores']) > 0:
                        result[i] = ds['batch']['round']['topdetsmap'][myidx][idx]
                except KeyError:
                    result[i] = {'scores' : np.array([]), 
                             'imgIds' : np.array([]),
                             'meta' : {}}  
    
            else:
                result[i] = {'scores' : np.array([]), 
                             'imgIds' : np.array([]),
                             'meta' : {}}  
            
        return result
        
    def getPosResult_testenv(self, idd, ds):
        myidx = int(np.where(self.myiminds==idd)[0])
        result = {}
        #dsload(topdetsmap[self.myidx, idx])
        #if size(ds.batch.round.topdetsmap,2) >= idx
        for i, idx in enumerate(self.myidx):
            if (len(ds['batch']['round']['topdetsmatlab'].keys())>=idx):
                
                try:
                    if len(ds['batch']['round']['topdetsmatlab'][idx][myidx]['scores']) > 0:
                        result[i] = ds['batch']['round']['topdetsmatlab'][idx][myidx]
                except KeyError:
                    result[i] = {'scores' : np.array([]), 
                             'imgIds' : np.array([]),
                             'meta' : {}}  
    
            else:
                result[i] = {'scores' : np.array([]), 
                             'imgIds' : np.array([]),
                             'meta' : {}}  
            
        return result
        
    def getNegResult(self, idd, ds):
    
        return self.getPosResult(idd, ds)
    
    def getNumClusters(self):
        return len(self.myidx)
    
    
def getTopNDetsPerCluster2(detectionResults, overlap, posIds, N, ds, matlab=False):
    
    numClusters = detectionResults.getNumClusters()
    scores = {k : [] for k in range(numClusters)}
    imgIds = {k : [] for k in range(numClusters)}
    meta = {k : [] for k in range(numClusters)}
    maxCacheSize = max(N, 200)
    maxToShow = N
    
    
    idsToUse = np.sort(posIds)
    nresults = 0
    total = 0
    #Doersch creates a progress bar here
    
    for idx in idsToUse:
        if not matlab:
            res = detectionResults.getPosResult(idx, ds)
        else:
            res = detectionResults.getPosResult_testenv(idx, ds)
        
        if len(res[0]['scores']) == 0:
            continue
        for clusti in range(numClusters):
            if clusti == 0:
                total += Utils.numel(res[clusti]['scores'])
            scores[clusti]+=(list(res[clusti]['scores']))
            imgIds[clusti]+=(list(res[clusti]['imgIds']))
            meta[clusti]+=list(res[clusti]['meta'].values())
            
            if len(scores[clusti]) > maxCacheSize:
                meta[clusti], scores[clusti], imgIds[clusti] = pickTopN(scores[clusti], imgIds[clusti], meta[clusti], maxToShow, overlap)
                imgIds[clusti] = list(imgIds[clusti])
    print('got', total, 'results for first detector')
    
    topN = {}
    for i in range(numClusters):
        topN[i] = {}
        m, s, imid = pickTopN(scores[i], imgIds[i], meta[i], maxToShow, overlap)
        topN[i]['meta'] = m
        topN[i]['scores'] = s
        topN[i]['imgIds'] = imid
    return topN

def pickTopN(scores, imgIds, meta, numToPick, maxOverlap):
    
    imgIds = np.array(imgIds)
    
    _, ordered = cleanUpOverlapping(meta, scores, imgIds, maxOverlap)
    
    toSelect = min(len(ordered), numToPick)
    _, uniqueim = np.unique(imgIds[ordered], return_index=True)
    ordered=ordered[np.sort(uniqueim)]
    selected = ordered[:min(len(ordered), toSelect)]
    
    meta = [meta[k] for k in selected]
    scores = [scores[k] for k in selected]
    imgIds = imgIds[selected]
    
    return meta, scores, imgIds
    
    return

def cleanUpOverlapping(patches, scores, correspImgs, maxOverlap):

    correspImgs = np.array(correspImgs)
    
    ascend = np.argsort(scores)
    inds = ascend[::-1] ## descend
    
    selected = np.zeros(len(patches))
    
    sortedPats = [patches[k] for k in inds]
    bbx = assignNN.getBoxesForPedro(sortedPats)
    uniqueImIds, m, n = np.unique(correspImgs, return_inverse=True, return_index=True)
    
    unids = np.argsort(m)
    uniqueImIds = uniqueImIds[unids]
    
    for i, imId in enumerate(uniqueImIds):
        sameImgPatInds = np.where(correspImgs[inds] == imId)[0]
        bx = bbx[sameImgPatInds]
        overlaps = PedroPascal.computePascalOverlap(bx, bx)
        nEl = overlaps.size
        interval = overlaps.shape[0]+1
        
        ## This is all to set the diagonal to zero I think.
        ## Wouldn't it be cleaner to just multiply by a array of ones with 0 on the diag?
        overlaps_flat = overlaps.flatten()
        overlaps_flat[list(range(0, nEl, interval))] = 0
        overlaps = overlaps_flat.reshape(overlaps.shape)
        
        aboveThresh = overlaps > maxOverlap
        p = np.max(overlaps, axis = 1)
        q = np.argmax(overlaps, axis = 1)
        isActive = np.ones(p.size, dtype=bool)
        
        for j in range(len(q)):
            if not isActive[j]:
                continue
            selected[inds[sameImgPatInds[j]]] = True
            isActive[aboveThresh[:,j]] = False
            
    selected = selected.astype(bool)
    scores = np.array(scores)
    
    order = np.argsort(scores[selected])
    order = order[::-1]
    
    selectedInds = np.where(selected)[0]
    
    order = selectedInds[order]
    return selected, order

def prepareDetectedPatchClusters(topN, nVote, nTop, params, trainSetPos, selectedClusters, ds, matlab=False):
    

    
    topN = selectN(topN, nVote)
    
    patsPerImg = [[] for x in range(len(trainSetPos))]
    for i in range(len(topN.keys())):
        if not matlab:
            inds = ismember(topN[i]['imgIds'], trainSetPos)        
        elif matlab:
            inds = ismember(topN[i]['imgIds'] - 1, trainSetPos)
        
        for j, ind in enumerate(inds):
            topN[i]['meta'][j]['clust'] = selectedClusters[i]
            topN[i]['meta'][j]['detScore'] = topN[i]['scores'][j]
            patsPerImg[ind].append(topN[i]['meta'][j])
            #print("patsPerImg")
            #for k in patsPerImg:
            #    print("1 x",len(k), "struct")
            
            
    allFeatures, allPatches, posCorrespImgs = calculateFeaturesFromPyramid(patsPerImg, params, trainSetPos, ds, matlab)
    
    posCorrespInds = ismember(posCorrespImgs, trainSetPos)
    assignedClustVote, assignedClustTrain = getAssignedClust(allPatches, nTop)
    selectedClust = getSelectedClusts(assignedClustVote, selectedClusters)
    print('Done features')
    return allFeatures, allPatches, posCorrespInds, posCorrespImgs, assignedClustVote, assignedClustTrain, selectedClust

def getSelectedClusts(assignedClust, selectedClusters):
    
    selectClust = []
    for clust in selectedClusters:
        if sum(assignedClust==clust) > 2:
            selectClust.append(clust)
    
    return selectClust

def getAssignedClust(allPatches, nTop):
    
    assignedClustVote = [x['clust'] for x in allPatches]
    assignedClustTrain = np.array(assignedClustVote)
    clusts = np.unique(assignedClustVote)    
    
    for clust in clusts:
        inds = np.where(assignedClustTrain == clust)[0]
        scores = [x['detScore'] for i, x in enumerate(allPatches) if i in inds]
        sortScore = np.argsort(scores)
        numToDiscard = max(0, len(inds) - nTop)
        selected = sortScore[:numToDiscard]
        assignedClustTrain[inds[selected]] = 0
        
    return assignedClustVote, assignedClustTrain


def calculateFeaturesFromPyramid(patches, params, imgIds, ds, matlab):
    
    totalPatches = len(patches)
    
    try:
        ds['cffp']['patches'] = patches
    except KeyError:
        ds['cffp'] = {}
        ds['cffp']['patches'] = patches
    
    for dsidx in range(totalPatches):
        print('autoclust_calcFeatsFromPyr', dsidx+1, '/', totalPatches)
        ds = autoclust_calcFeatsFromPyr(ds, dsidx, matlab)
    
    allFeatures = ds['cffp']['allFeatures']
    
    del ds['cffp']
    
    indexes = []
    
    print('Collecting all in one array')
    
    posPatches = [d for sublist in patches for d in sublist]
    allFeat = np.zeros((0,2112))
    ### What if all feat is empty? This gives us a keyerorr but also needs us to adjust
    ### The code below. 
    for key in range(len(allFeatures.keys())):
        if len(allFeatures[key]) != 0:
            allFeat = np.vstack((allFeat, allFeatures[key]))
            
    ##Matlab hints at a possibility for feat to be empty. I can't really see how. 
    for i in range(totalPatches):
        
        feat = allFeatures[i]
        
        if feat.size == 0:
            continue
        
        inds = np.ones(feat.shape[0]) * imgIds[i]
        inds = inds.astype(int)
        if len(inds) != 0:
            indexes += list(inds)
    print('Done')
    
    collatedPatches = posPatches
    features = allFeat    
    
    return features, collatedPatches, indexes

def autoclust_calcFeatsFromPyr(ds, dsidx, matlab):
    
    pPat = ds['cffp']['patches'][dsidx]
    
    if len(pPat) > 0:
        imPath = pPat[0]['im']
        ### if statement below because we're testing with matlab data
        if imPath[0] == '[':
            imPath = imPath[2:-2]
            imPath = ds['conf']['gbz'][ds['conf']['currimset']]['cutoutdir'] + imPath[51:]
        
        pyra = Utils.constructFeaturePyramidForImg(io.imread(imPath)/255, ds['conf']['params'])
        feats = getPatchFeaturesFromPyramid(pPat, pyra, ds['conf']['params'], matlab)
        try:
            ds['cffp']['allFeatures'][dsidx] = feats
        except KeyError:
            ds['cffp']['allFeatures'] = {}
            ds['cffp']['allFeatures'][dsidx] = feats
    else:
        try:
            ds['cffp']['allFeatures'][dsidx] = np.array([])
        except KeyError:
            ds['cffp']['allFeatures'] = {}
            ds['cffp']['allFeatures'][dsidx] = np.array([])
    return ds


def getPatchFeaturesFromPyramid(patches, pyramid, params, matlab=False):
    
    nrows, ncols, nzee, nextra = Utils.getCanonicalPatchHOGSize(params)
    numElement = nrows * ncols * nzee + nextra
    feats = np.zeros((len(patches), numElement))
    
    for i in range(len(patches)):
        pyramidInfo = patches[i]['pyramid']
        
        if matlab:
            pyramidInfo = list(np.array(pyramidInfo) - 1)
        
        pyraLevel, r, c = pyramidInfo.astype(int)
        
        ### pyramidfeats from matlab, this was to test whether we were to use r+nrows or r+nrows-1
        #workspace = loadmat(matPATH + 'PyramidFeaturesForTestPullingPatFeats')
        #pyramidfeats = workspace['pfeats'][0][1]
        #matFeat = pyramidfeats[r : r+nrows, c:c+ncols, :]
        
        patFeat = pyramid['features'][pyraLevel][r : r+nrows, c:c+ncols, :]
        if 'useColorHists' in params:
            raise ValueError("Not implemented yet!")
        else:
            feats[i] = patFeat.flatten('F')
        
    return feats



def selectN(topN, N):
    
    ## possible that topN isnt always a dict but sometimes a list
    for key in topN.keys():
        ids = topN[key]['imgIds']
        selInd = np.where(ids>-1)[0]
        toSel = min(N, len(selInd))
        topN[key]['meta'] = topN[key]['meta'][:toSel]
        topN[key]['scores'] = topN[key]['scores'][:toSel]
        topN[key]['imgIds'] = topN[key]['imgIds'][:toSel]
        
    return topN

def selectDetectors(dets, inds):
    
    flm = dets.firstLevModels
    detscpy = VisualEntityDetectors({}, dets.params)
    detscpy.firstLevModels['w'] = flm['w'][inds,:]
    detscpy.firstLevModels['rho'] = flm['rho'][inds]
    detscpy.firstLevModels['firstLabel'] = flm['firstLabel'][inds]
    detscpy.firstLevModels['info'] =  {k : v for k, v in flm['info'].items() if k in inds}
    detscpy.firstLevModels['threshold'] = flm['threshold'][inds]
    
    return detscpy
    


def ismember(a,b):
    
    return [int(np.where(b==x)[0]) for x in a if x in b]

def ismember2(a, b):
    
    l = []
    
    for i, x in enumerate(a):
        if x in b:
            l.append(i)
            
    return np.array(l)
        

def quicksave(val, string='temp'):
    with open(string + '.pickle', 'wb') as handle:
        pickle.dump(val, handle)

def quickload(string='temp'):
    data = open(string + '.pickle', 'rb')
    return pickle.load(data)







    
