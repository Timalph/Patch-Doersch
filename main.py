#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:13:38 2021

@author: timalph
@environment: parisclone
"""


import os
import pickle
import numpy as np
import time

import Utils
import Sample_Random_Patches as srp
import assignNN
import PedroPascal
import train
import mat_funcs
import math
import pandas as pd
from itertools import compress
import Visualizing
from scipy.io import loadmat

import cv2 as cv
### Set up dir of datastruct
### Set up dir with dataset
### set up root_dir


no_images = 100
cut = 12
#out_dir = "bigparis_test_10k"
out_dir = "tiny_paris"
prediction_model = 'SVC'

## Moeten alle /255 -> /256 worden?
##### 

matlabinds=False

def main1(matlabinds=False):
    # Step 1 build directory structure
     
    ds = {}
    ds['savedir'] = NotImplemented
    
    out_dir = 'out/'
    Utils.setup_dir(out_dir)
    path_out = os.path.abspath(os.getcwd()) + '/' + 'out_dir'
    
    
    ds['sys'] = {'outdir' : path_out}
    
    
    
    
    
    #### Load the struct
    #ams_struct = load('Users/timalph/Documents/Paris/release/dataset_new_tiny.mat')
    
    #ams_dataset with lines name (filename), date, bytes, isdir, datenum, 
    # city(dir), istrain([1,0]), fullname (city + filename), imsize, arrond( unsure what arrond is)
    #ams_dataset = ams_struct.dataset_test
    
    #Set root_dir that contains the city dirs and images. 
    #Dataframe is a simple pandas represenation of the struct they use themselves
    root_dir = '/Users/timalph/Documents/Panoramas/processed/dataset_tiny/cutouts/'
    with open('/Users/timalph/Documents/Paris/release/pyth/dataset.p', "rb") as input_file:
        dataframe = pickle.load(input_file)
    
    
    
    
    ds['imgs'] = {}
    pos = len(ds['imgs']) +1
    ds['imgs'][pos] = dataframe
    ds['conf'] = {}
    ds['conf']['gbz'] = {}
    ds['conf']['gbz'][pos] = {'cutoutdir': root_dir, 'imgsurl' : ''}
    ds['conf']['currimset'] = pos
    
    
    #define the number of training iterations used.  The paper uses 3; sometimes
    #using as many as 5 can result in minor improvements.
    #numtrainits is redefined somewhere else because this entire piece of code is in a function
    num_train_its=3
    
    np.random.seed(20)
    
    ## SVM Flags are here to make sure we know what the flags are when we run in to 
    ## Code that uses them. 
    ds['conf']['params'] = {'imageCanonicalSize': 400, 'patchCanonicalSize': np.array([80,80]),
                            'scaleIntervals': 8, 'sBins': 8, 'useColor': 1,
                            'patchOverlapThreshold': 0.6}
    
    ds['conf']['params']['svmflags']= '-s 0 -t 0 -c 0.1'  
    
    ds['conf']['detectionParams'] = {'selectTopN': False, 'useDecisionThresh': True,
                                   'overlap': 0.4, 'fixedDecisionThresh': -1.002}
    
    ## This seems like an elaborate way to just specify imgs is the dataframe
    #imgs = ds['imgs'][ds['conf']['currimset']]
    imgs = dataframe
    ds['mycity'] = 'positivecity'
    
    
    # Does parimgs mean: Paris images?
    parimgs = dataframe.loc[dataframe['city'] == ds['mycity']]
    otherimgs = dataframe.loc[dataframe['city'] != ds['mycity']]
    
    pos_ids = pd.Index.to_numpy(parimgs.index)
    neg_ids = pd.Index.to_numpy(otherimgs.index)
    
    ds['ispos'] = np.concatenate((np.ones(len(pos_ids)), np.zeros(len(neg_ids)))).astype('int')
    
    if not matlabinds:
        np.random.shuffle(pos_ids)
        np.random.shuffle(neg_ids)
        ds['myiminds'] = np.concatenate((pos_ids, neg_ids))
    elif matlabinds:
        matPATH = '/Users/timalph/Documents/Paris/release/'
        workspace = loadmat(matPATH + 'matlab_random_shuffle')
        ds['myiminds'] = workspace['matinds'].squeeze() -1

    ds['parimgs'] = parimgs
    
    
    #sample random positive "candidate" patches
    
    
    step = 2
    
    ### He randomly samples the indices of half of the positive images
    
    ds['isinit'] = Utils.makemarks(ds['myiminds'][list(range(0, len(ds['myiminds']), 2))], len(imgs))
    
    initInds = np.where(np.logical_and(ds['ispos'], ds['isinit']))[0]
    ## Then he starts sampling positive patches
    ds['sample'] = {}
    ds['sample']['initInds'] = initInds
    ds['sample']['patches'] = {}
    ds['sample']['features'] = {}
    
    
    
    print('Sampling positive patches')
    
    for ind in ds['sample']['initInds']:
        print(ind)
        ## srp.main()
        ds['sample']['patches'][ind], ds['sample']['features'][ind], _ = srp.main(ind, ds, 25, matlabinds=matlabinds)
        
    no_of_pos_images = len(initInds)
    
    initPatches = [pd.DataFrame.from_dict(d, orient='index') for d in ds['sample']['patches'].values()]              
    initPatches = pd.concat(initPatches, ignore_index=True)
    ds['initPatches'] = initPatches
    
    initFeats = np.zeros((0, 2112))
    for d in initInds:
        initFeats = np.vstack((initFeats, ds['sample']['features'][d]))

    ds['initFeats'] = initFeats
    
    ds['initImgInds']=initInds
    
    ds['sample']['patches'].clear()
    ds['sample']['features'].clear()
    
    ## Sample negative indices
    workspace = loadmat(matPATH + 'negnums_for_python')
    initInds = np.where(np.logical_and(1-ds['ispos'], ds['isinit']))[0]
    
    order = workspace['ord'].squeeze()-1
    myinds = order[:min(len(order),30)]
    
    if not matlabinds:
        raise ValueError("Mapreduce is continuing its count of dsidx I presume. I don't know what the implications of that are for this implementation.")
    print('Sampling negative patches')
    ds['sample']['initInds'] = myinds
    ## no_of_pos_images makes sure this script continues with dsidx where the 
    ## matlabscript starts again after sampling positive patches

    for ind in ds['sample']['initInds'][no_of_pos_images:]:
        print(ind)
        if matlabinds:
            matlab_neg_inds = 'neg'
        else:
            matlab_neg_inds = False
        ds['sample']['patches'][ind], ds['sample']['features'][ind], _ = srp.main(ind, ds, matlabinds = matlab_neg_inds)
        with open('11april_backup.pickle', 'wb') as handle:
            pickle.dump(ds, handle)
    initPatches = [pd.DataFrame.from_dict(d, orient='index') for d in ds['sample']['patches'].values()]              
    initPatches = pd.concat(initPatches, ignore_index=True)
    ds['initPatchesNeg'] = initPatches  
    
    initFeats = [d for d in ds['sample']['features'].values()] 
    initFeats = np.concatenate(initFeats)
    ds['initFeatsNeg'] = initFeats
    ds['initImgIndsNeg']=initInds
    
    
    ### Centers only computed for positive set
    ### The mean operator in matlab outputs something ever so slightly different
    ds['centers'] = Utils.getcenters(ds['initFeats'])
    ds['selectedClust'] = np.arange(ds['initFeats'].shape[0])
    ds['assignedClust'] = ds['selectedClust']
    
    ## compute nearest neighbors for each candidate patch
    #ds['centers'] = []
    ds = assignNN.main(ds)
    #ds['centers'] = []
    with open ('15april_dsbackup_afterassignNN.pickle', 'wb') as f: 
        pickle.dump(ds, f)
    #### Didn't fill empty rows with infites because why would I do that Carl Doersch?

## ===== Load Checkpoint ===== ##
#f = open('ds_checkpoint_nearest_neighbours.pickle', 'rb')
#f = open('ds_checkpoint_nearest_neighbours_23_03_2022.pickle', 'rb')
f = open('15april_dsbackup_afterassignNN.pickle', 'rb')
ds = pickle.load(f)
ds['conf']['params']['svmflags'] = '-s 0 -t 0 -c 0.1'
ds['conf']['params']['imageCanonicalSize'] = 400
ds['raw_outdir'] = out_dir
ds['prediction_model'] = prediction_model
## =========================== ##

#Turn assignednn into an array
shape0 = len(ds['assignednn'].values())
shape1 = len(ds['assignednn'][0])

assignednn = np.zeros((shape0,shape1))
for k, v in ds['assignednn'].items():
    assignednn[k] = v.squeeze()

assignednn = assignednn.T
npatches = len(ds['centers'])


### Pyth assignednn sum = 9.111550764995627
### Matlab assignednn sum= 9.106705182094096
if matlabinds:
    matPATH = '/Users/timalph/Documents/Paris/release/'
    workspace = loadmat(matPATH + 'create_topn_in_out_13apr')
    assignednn2 = workspace['assignednn']



### THE ORIGINAL USES NNEIGHBORS =100, Delete the 35 when you run it for real.
topndist, topnlab, topnidx = assignNN.create_topn(ds, npatches, assignednn, nneighbors=35)

### We get different dist, lab, and idx because somehow its impossible to take the mean of an array
### in the same manner in python and matlab.

if matlabinds:
    topndist = workspace['topndist']
    topnlab = workspace['topnlab']
    topnidx = workspace['topnidx']-1
    





perclustpost = np.sum(topnlab[:, :20], axis=1)

#### We need to do some sorting acrobatics to get the sort by last 
#### convention used in matlab.
#### We don't need to worry about complexity as this array is relatively short.
#### We want the list to be sorted, descending, but also that the indices 
#### corresponding to the same value are ascending. 


postord=[]
for v in sorted(np.unique(perclustpost))[::-1]:
    listofindices = list(np.where(perclustpost==v)[0])
    postord+= listofindices
    
ds['perclustpost'] = perclustpost[postord]
ds['selectedClust'] = ds['selectedClust'][postord]


## cut = 12 because we have 100 images.
disppats = np.intersect1d(ds['assignedClust'], ds['selectedClust'][:cut])
correspimg = np.array(ds['initPatches']['imidx'])
#a, b, c= srp.main(18, ds, 25)



detections = ds['initPatches'].iloc[disppats]
imidx = correspimg[disppats]
assignedClust = ds['assignedClust'][disppats]

## Configuration argument set to 'First'
currdets = assignNN.simplifydets(detections, imidx, assignedClust, 'First')
### here we can start multiprocessing
ds = Utils.prepbatchwisebestbin(ds, currdets, 0, 1)

## If necessary
ds['bestbin0'] = ds['bestbin'].copy()
ds['bestbin'] = {}

###===================
### They dssave here!=
###===================

curridx=0
selClustIdx=0
mainflag=1
topndets={}
gitopndetshalf={}
topndetstrain={}
topnorig=[]
newselclust=[]

## Create topndetshalfpickle for testing
#with open('create_topndetshalf_ds.pickle', 'wb') as handle:
#    pickle.dump(ds, handle)
    
#with open('26april_before_create_topndetshalf.pickle', 'wb') as handle:
#    pickle.dump(ds, handle)
topndets, topndetshalf, topndetstrain, topnorig, newselclust = assignNN.create_topndetshalf(ds, postord, 
                                                                               topnidx, topndist, 
                                                                               mainflag, nneighbors=35)


#newselclust

#delete topndetshalf

ds['selectedClust'] = newselclust
ds['topnidx'] = topnidx
ds['topnlab'] = topnlab
ds['topndist'] = topndist

# =============================================================================
# topndets=cell2mat(topndets);
# if(dsfield(ds,'dispoutpath')),dssymlink(['ds.bestbin_topn'],ds.dispoutpath);end
# =============================================================================

#### SKIPPING EVERYTHING AROUND BESTBIN (287 - 293)
#### Highly probable this is only used for displaying

#ds = Utils.prepbatchwisebestbin(ds, topnorig, 0, 1, 1)

#prepbatchwisebestbin(topnorig, 0, 1, 1)




# =============================================================================
# Extract feature for the top 5 of each cluster
# =============================================================================

topndetstrain = pd.concat(topndetstrain).reset_index(drop=True)

#with open ('extractpatches_ds_26april.pickle', 'wb') as f:
#    pickle.dump(ds, f)

#Extract the features for the top 5 for each cluster
#Qualitatively checked
trpatches = Utils.extractpatches(ds, topndetstrain, ds['imgs'][ds['conf']['currimset']])

### Python version of dsmv
ds['initFeatsNeg'] = np.vstack((ds['initFeatsNeg'], ds['initFeats']))
ds['initFeatsOrig'] = ds.pop('initFeats')
ds['assignedClustOrig'] = ds.pop('assignedClust')


ds['initFeats'] = np.zeros((len(trpatches), ds['initFeatsOrig'].shape[1]))
ds['initFeatsOrig'] = []
 

extrparams=ds['conf']['params'].copy()
extrparams['imageCanonicalSize'] = min(ds['conf']['params']['patchCanonicalSize'])
for i in range(len(trpatches)):
    #print(i)
    tmp = Utils.constructFeaturePyramidForImg(trpatches[i], extrparams, levels=1)
    #print(tmp['features'][0].shape)
    ds['initFeats'][i,:] = tmp['features'][0].flatten(order='F')
    if(i%10)==0:
        print(i)

ds['assignedClust'] = topndetstrain['detector'].to_numpy()
ds['posPatches'] = topndetstrain

del trpatches
del topnidx
del topnlab
del topndist
del topnorig

ds['conf']['processingBatchSize'] = 600
pbs = ds['conf']['processingBatchSize']
batchidx=0
starti=1

if 'batch' in ds:
    if 'curriter' in ds['batch']:
        starti = 1 + (pbs*(ds['batch']['curriter']-1))
        batchidx=ds['batch']['curriter']-1

maintic = time.time()
# =============================================================================
# if(isparallel&&(~dsmapredisopen()))
#   dsmapredopen(nprocs,1,~isdistributed);
#   pause(10);
# end
# =============================================================================
j=0

ds['batch'] = {}
ds['batch']['round'] = {}
ds['batch']['round']['assignedClust'] = np.array([])
ds['batch']['round']['posFeatures'] = np.array([])
ds['batch']['round']['selectedClust'] = []
ds['batch']['round']['selClustIts'] = []
ds['batch']['nextClust'] = 0
ds['batch']['finishedDets'] = []
ds['batch']['nFinishedDets'] = 0

ds['batch']['round']['traintopN'] = {}
ds['batch']['round']['validtopN'] = {}
ds['batch']['round']['alltopN']   = {}

ds['findetectors'] = {}
ds['finSelectedClust'] = {}
ds['batch']['round']['negmin'] = {}
num_train_its=3
### we want the y dim of len batch round posfeatures, may need to be an array instead of a list

conf2 = {}

while((ds['batch']['nextClust'] <= len(ds['selectedClust'])) or     
      len(ds['batch']['round']['posFeatures']) > 0):
    
    if ds['batch']['nextClust'] <= len(ds['selectedClust']):
        print("nextClust: {} <= no of selectedClust: {}".format(ds['batch']['nextClust'],
                                                                len(ds['selectedClust'])))
    
    if len(ds['batch']['round']['posFeatures']) > 0:
        print("no of posFeatures > 0 : {}".format(len(ds['batch']['round']['posFeatures'])))
    
    ds['batch']['round']['curriter'] = j
    
    
    # code below used to pause the script during training, i'm not sure
    # stopfile=[ds.prevnm '_stop'];
    # if(exist(stopfile,'file'))
    #   %lets you stop training and just output the results so far
    #   break;
    # end
    # pausefile=[ds.prevnm '_pause'];
    # if(exist(pausefile,'file'))
    #   keyboard;
    # end
    
    # Choose which candidate clusters to start working on

    ntoadd = ds['conf']['processingBatchSize'] - len(ds['batch']['round']['selectedClust'])
    rngend=min((ds['batch']['nextClust'] + ntoadd), len(ds['selectedClust']))
    newselclust=[ds['selectedClust'][i] for i in np.arange(ds['batch']['nextClust'], rngend)]
    
    ismem = np.in1d(ds['assignedClust'], newselclust)
    newfeats = np.arange(len(ismem))[ismem]
    
    try:
        ds['batch']['round']['posFeatures']  = np.vstack((ds['batch']['round']['posFeatures'], ds['initFeats'][newfeats]))
    except ValueError:
        ds['batch']['round']['posFeatures'] = ds['initFeats'][newfeats]
    try:
        ds['batch']['round']['assignedClust'] = np.append(ds['batch']['round']['assignedClust'], ds['assignedClust'][newfeats])
    except ValueError:
        ds['batch']['round']['assignedClust'] = ds['assignedClust'][newfeats]
        
    if len(newselclust) != 0:
        ds['batch']['round']['selectedClust'] += newselclust
    ds['batch']['round']['selClustIts'] += [0] * len(newselclust)
    ds['batch']['nextClust'] += ntoadd
    
    
    # Choose the training/validation sets for the current round
    nsets = 3
    jidx = np.mod(j, nsets)
    jidxp1 = np.mod(j+1, nsets)
    
    p_train_indices = np.arange(jidx,len(ds['parimgs']),nsets)
    n_train_indices = np.arange(len(ds['parimgs'])+j,len(ds['myiminds']),7)
    
    training_indices = np.concatenate((p_train_indices, n_train_indices))
    
    p_val_indices = np.arange(jidxp1,len(ds['parimgs']),nsets)
    n_val_indices = np.arange(len(ds['parimgs'])+j+1,len(ds['myiminds']),7)
    
    validation_indices = np.concatenate((p_val_indices, n_val_indices))
    
    currtrainset = ds['myiminds'][training_indices]
    currvalset = ds['myiminds'][validation_indices]
    
    ds['batch']['round']['totrainon'] = currtrainset
    ds['batch']['round']['tovalon'] = currvalset
    
    #with open('ds_for_svmtesting.pickle', 'wb') as handle:
    #    pickle.dump(ds, handle)
    
    # ===================================================== #
    # Initialize the SVMs using the random negative patches
    # ===================================================== #
    
    ds = train.initialize(ds, modeltype=ds['prediction_model'])
    dets = train.VisualEntityDetectors(ds['batch']['round']['firstDet'],
                                       ds['conf']['params'])
    ds['batch']['round']['detectors'] = dets
    
    # Use the hard negative mining technique to train on negatives from the current negative set
    
    istrain = np.zeros(len(ds['imgs'][ds['conf']['currimset']]))
    istrain[ds['batch']['round']['totrainon']] = 1
     
    allnegs = np.where(np.logical_and(ds['ispos']^1, istrain))[0]
    currentInd = 0
    maxElements = len(allnegs)
    iteration = 0
    startImgsPerIter = 15
    alpha = 0.71
    
    #with open('beforeNotMineddetectors.pickle', 'wb') as handle:
    #    pickle.dump(ds, handle)
    
    if not 'mineddetectors' in ds['batch']['round']:
        
# =============================================================================
#         try:
#             ds['batch']['round'].pop('negmin')
#         except:
#             pass
# =============================================================================
        ds['batch']['round']['negmin'] = {}
        while currentInd <= maxElements:
            imgsPerIter = math.floor(startImgsPerIter * 2**(iteration * alpha))
            finInd = min(currentInd + imgsPerIter, maxElements)

            ds['batch']['round']['negmin']['iminds'] = allnegs[currentInd:finInd]
            
            try:
                conf['noloadresults'] = 1
            except NameError:
                conf = {'noloadresults' : 1}
            ## Train autoclust_mine_negs works perfectly (qualitatively)
            ds['batch']['round']['nextnegmin']= {}
            ds['batch']['round']['nextnegmin']['prevfeats'] = {}
            ds = train.autoclust_mine_negs_speed(ds)
            ds = train.autoclust_train_negs(ds)
            
            ## Not necessary as its already loaded in the ds
            #Utils.dsload('ds.batch.round.nextnegmin', 'traineddetectors')
            
            dets = train.VisualEntityDetectors(ds['batch']['round']['nextnegmin']['traineddetectors'], ds['conf']['params'])
             
            ds['batch']['round']['detectors']  = dets
            ds['batch']['round']['negmin'] = ds['batch']['round'].pop('nextnegmin')
            
            iteration +=1
            currentInd += imgsPerIter
            try:
                with open('aftermined.pickle', 'wb') as handle:
                    pickle.dump((ds, iteration), handle)
            except:
                pass
        ds['batch']['round'].pop('negmin')
        
    ds['batch']['round']['iminds'] = np.concatenate((ds['batch']['round']['totrainon'],ds['batch']['round']['tovalon']))
    ds['batch']['round']['mineddetectors'] = dets  
        
        
    # run detection on both the training and validation sets
    ds = train.autoclust_detect(ds)
    conf2['allatonce']=True
    matlab = False
    ### This is where we make traintopN
    for dsidx in range(len(ds['batch']['round']['selectedClust'])):
        print("clust_topn: {}/{}".format(dsidx+1, len(ds['batch']['round']['selectedClust'])))
        ds = train.autoclust_topn(ds, dsidx, matlab)
    
    validtopN = ds['batch']['round']['validtopN']
    traintopN = ds['batch']['round']['traintopN']
    
    # extract the top 5 form the validation set for the next round

    indices_pdpc = ds['batch']['round']['tovalon'][np.where(ds['ispos'][ds['batch']['round']['tovalon']])[0]]
    out_tuple = train.prepareDetectedPatchClusters(validtopN, 10, 5, ds['conf']['params'], indices_pdpc, ds['batch']['round']['selectedClust'], ds, matlab)

    posFeatures, positivePatches, posCorrespInds, posCorrespImgs, assignedClustVote,  assignedClustTrain, selectedClusters = out_tuple
    
    #extract the top 100 and display them
    out = train.prepareDetectedPatchClusters(ds['batch']['round']['alltopN'],
                                        100, 100, ds['conf']['params'],
                                        ds['batch']['round']['tovalon'],
                                        ds['batch']['round']['selectedClust'],
                                        ds, matlab)
    
    _, positivePatches2, _, posCorrespImgs2, _, assignedClustTrain2, _ = out
    
    dispdets = assignNN.simplifydets(pd.DataFrame(positivePatches2), posCorrespImgs2, assignedClustTrain2, configuration='Non-empty decisions')
    
    dispdetscell = pd.DataFrame([])
    dispdetscellv2 = np.array([])
    
            
    ## It says here that code has been chaged from for to end and dsipdets=cell2mat etc
    ## I assume this is my past self telling me that something has been changed for it to work
    ## Adjust this when you run the full script accordingly. 
    
    for i, clust in enumerate(ds['batch']['round']['selectedClust']):

        mydispdets=dispdets.iloc[np.where(dispdets['detector'].to_numpy()==clust)[0]]
        if len(mydispdets) == 0:
            continue
        ord5 = np.argsort(mydispdets['decision'].to_numpy())[::-1]
        dispdetscell = dispdetscell.append(mydispdets.iloc[ord5[list(range(min(10, len(ord5)))) + list(range(14, min(len(ord5), 100), 7))]], ignore_index=True)

    ## Note from Doersch in the Matlab code
    ## up until this point in the while loop, if the program crashes (e.g. due
    ## to disk write failers you can just restart it at line 286 and the right 
    ## thing should happen. After this point, however, the program starts performing updates that shouldn't happen twice')
    
    ## prepbatchwisebestbin is used here, unsure if I need this. 
    
    
    ## assuming this is all just for displaying
    #dispres_discpatch (create html display from ds.bestbin)
    #ds['bestbin_topn']['alldiscpatchimg'] = np.zeros(len(ds['bestbin_topn']['alldiscpatchimg']))
    
    tooOldClusts = np.array(ds['batch']['round']['selectedClust'])[np.array(ds['batch']['round']['selClustIts']) >= num_train_its-1]
    try:
        ds['sys']['savestate']['thresh'] = []
    except KeyError:
        ds['sys']['savestate'] = {}
        ds['sys']['savestate']['thresh'] = []
        try:
            ds['sys']['savestate']['thresh'] = []
        except KeyError:
            ds['sys'] = {}
            ds['sys']['savestate'] = {}
            ds['sys']['savestate']['thresh'] = []
            
    finished = train.ismember(ds['batch']['round']['selectedClust'], np.intersect1d(selectedClusters, tooOldClusts))
    finished = np.array(finished).astype(int)
    
    ds['findetectors'][j] = train.selectDetectors(ds['batch']['round']['detectors'], finished)
    ds['finSelectedClust'][j] = np.array(ds['batch']['round']['selectedClust'])[finished]
    
    #Store stuff (finished detectors, top detections etc.) for next round.
    
    
    ds['batch']['nFinishedDets'] += ds['findetectors'][j].firstLevModels['w'].shape[0]
    selectedClusters = np.setdiff1d(selectedClusters, tooOldClusts)
    #markedAssiClust = ismember(ds['batch']['round']['assignedClust'], selectedClusters)
    markedAssiClust = train.ismember2(assignedClustTrain, selectedClusters)
    
    #### Error when markedAssiClust is sempty
    if len(markedAssiClust) > 0:
        assignedClustTrain = assignedClustTrain[markedAssiClust]
        
        posFeatures = posFeatures[markedAssiClust]
    elif len(markedAssiClust) == 0:
        assignedClustTrain = np.array([])
        posFeatures = np.array([])
    indstokeep = train.ismember(selectedClusters, ds['batch']['round']['selectedClust'])
    ## indstokeep(indstokeep==0) = [];
    selClustIts = [x+1 for i, x in enumerate(ds['batch']['round']['selClustIts']) if i in indstokeep]
    del ds['batch']['round']['topdetsmap']
    #creates a backup of ds.batch.round

    ds['batch']['round']['posFeatures'] = posFeatures
    #print(len(ds['batch']['round']['assignedClust']))
    ds['batch']['round']['assignedClust'] = assignedClustTrain
    ds['batch']['round']['selectedClust'] = selectedClusters
    ds['batch']['round']['selClustIts'] = selClustIts
    ### save batch round??
    ds['batch']['round'].pop('mineddetectors')
    
    j+=1
    try:
        with open('finpickle.pickle', 'wb') as handle:
            pickle.dump(ds, handle)
    except:
        pass
            

## can't pickle cpointers
dets = Visualizing.collateAllDetectors(ds['findetectors'])
ds['selectedClust'] = np.concatenate([ds['finSelectedClust'][k] for k in range(len(ds['finSelectedClust']))])
dps = {'selectTopN' : False,
       'useDecisionThresh' : True,
       'overlap' : .5,
       'fixedDecisionThresh' : -.85,
       'removeFeatures' : 1,}

ds['dets'] = dets
ds['detsimple'] = {}

test_with_matlabsvms=False
if test_with_matlabsvms:
    workspace = loadmat(matPATH + 'ForTestingreaddetsimplewithSVMs')
    wm = workspace['model_w']
    rhom = workspace['model_rho'].squeeze()
    ds['dets'][0].firstLevModels['w'] = wm
    ds['dets'][0].firstLevModels['rho'] = rhom

ds['conf']['detectionParams']['overlap'] = 0.5
ds['conf']['detectionParams']['fixedDecisionThresh'] = -.85


for i, imind in enumerate(ds['myiminds']):
    img = Utils.getimg(ds, imind)
    detections = ds['dets'][0].detectPresenceInImg(img, ds)
    ds['detsimple'][i] = assignNN.simplifydets(detections, imind)

maxdet = ds['dets'][0].firstLevModels['w'].shape[0]
imgs = ds['imgs'][ds['conf']['currimset']]

disptypes = ['overallcounts', 'posterior']
 
topn, posCounts, negCounts, _ = Visualizing.readdetsimple(maxdet,ds, -.2, {'oneperim' : 1,
                                                         'issingle' : 1,
                                                         'nperdet'  : 250})
alldetections = pd.concat([topn[0],topn[1]])

for k, v in posCounts.items():
    posCounts[k] = np.array(v)
    
negCounts = np.array(negCounts)

if matlabinds:
    workspace = loadmat(matPATH + 'tmp_alldetections')
    alldetectionsm = workspace['alldetections']
    alldetections = mat_funcs.struct2pandas(alldetectionsm, 
                                            topndets_variation=True, 
                                            metadata_variation=True)
    alldetections['imidx'] -=1
    alldetections['detector']-=1

topNall = {}
alldetections = alldetections.sort_values(['detector', 'decision'], ascending= [True, False]).reset_index(drop=True)
for disptype in disptypes:
    detsimpletmp = pd.DataFrame()
    tmpdetector = alldetections['detector']
    tmpdecision = alldetections['decision']
    
    counts = np.zeros((len(tmpdetector.unique()), 2)).astype(int)
    for i in tmpdetector.unique():
        alltmpsfordetr = alldetections[alldetections['detector'] == i]
        tmpdetsfordetr = alltmpsfordetr[:30]
        if disptype == 'overallcounts':
            topNall[i] = alltmpsfordetr[:250] 
        else:
            topNall[i] = alltmpsfordetr[:50]
        
        detsimpletmp = detsimpletmp.append(tmpdetsfordetr)
        
        if disptype == 'overallcounts':
            counts[i, 0] = sum(ds['ispos'][tmpdetsfordetr['imidx'].to_numpy().astype(int)])
            counts[i, 1] = len(tmpdetsfordetr) - counts[i,0]
        elif disptype == 'posterior':
            counts[i, 0] = sum(posCounts[0][:,0])
            counts[i, 1] = sum(negCounts[0, :])
            
        print(i)
        
    if disptype == 'overallcounts':
        detord = mat_funcs.matsort(counts[:, 0], descending=True)
    else:
        post = (counts[:,0]+1)/(np.sum(counts, axis=1)+2)
        detord = mat_funcs.matsort(post, descending=True)
    
    ## Groups and affinities are all exactly the same as MATLAB
    ## This means they might need -1 indexing
    overl, groups, affinities = Visualizing.findOverlapping([topNall[i] for i in detord], {'findNonOverlap' : 1})
    
    resSelectedClust = np.array([ds['selectedClust'][x] for x in detord[overl]])
    
    detsimple = topn[0]
    
    ### ????? this suddenly didnt work and I cant check why without rerunning the entire matlab script
    try:
        for j in range(len(detsimple)):
            detsimple['detector'].iloc[j] = ds['selectedClust'][detsimple['detector'].iloc[j]]
    except:
        pass
    
    
    if not 'selectedClustDclust' in ds:
        ds['selectedClustDclust'] = resSelectedClust
        mapping = []
        for x in ds['selectedClustDclust']:
            mapping.append(np.where(ds['selectedClust'] == x)[0])
        mapping=np.array(mapping).squeeze()
        
        ds['detsDclust'] = Visualizing.selectDetectors(ds['dets'][0], mapping)
    
    #Generate a display of the final detectors. 
    
    ds['bestbin']['imgs'] = imgs
    nycdets2 = pd.DataFrame()
    mydetectors = []
    mydecisions = []
    nycdets = detsimple
    #for j in range(len(nycdets), -1, -1):
    #    myinds = np.where(mydetectors == j)[0]
    mydetectors=nycdets['detector'].to_numpy()
    mydecisions=nycdets['decision'].to_numpy()
    curridx=1
    for j in np.unique(mydetectors):
        myinds = np.where(mydetectors==j)[0]
        best = np.argsort(mydecisions[myinds])[-20:]
        nycdets2 = nycdets2.append(nycdets.iloc[myinds[best]])
        curridx+=1
        
    
    
    ds['bestbin']['alldisclabelcat'] = np.vstack((nycdets2['imidx'].to_numpy(), nycdets2['detector'].to_numpy())).T
    nycdets2 = nycdets2.reset_index()
    ds['bestbin']['alldiscpatchimg'] = Utils.extractpatches(ds, nycdets2, ds['bestbin']['imgs'])
    ### Save all discpatchimgs to a folder
    try:
        os.makedirs(ds['raw_outdir'] + '/alldiscpatchimg[]')
    except FileExistsError:
        pass
    if not 'savedir' in ds:
        for key, value in ds['bestbin']['alldiscpatchimg'].items():
                cv.imwrite( ds['raw_outdir'] + '/alldiscpatchimg[]/{}.jpg'.format(key), 
                           cv.cvtColor(value, cv.COLOR_BGR2RGB))
            
        
        
    else:
        raise NotImplementedError
    
    ds['bestbin']['decision'] = nycdets2.decision.to_numpy()
    countsIdxOrd = detord[overl[:min(len(overl), 500)]]
    ds['bestbin']['tosave'] = ds['selectedClust'][countsIdxOrd]
    ds['bestbin']['isgeneral'] = np.ones(len(ds['bestbin']['tosave']))
    ds['bestbin']['counts'] = counts[countsIdxOrd]
    
    ## if exists misclabel, didnt exist in tinyparis run
    
    ds = Visualizing.dispres_discpatch(ds)
    with open(ds['raw_outdir'] + '/output.html', 'w') as f:
        f.write(ds['bestbin']['bbhtml'])
   