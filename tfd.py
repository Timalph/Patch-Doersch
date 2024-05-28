#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 12:38:41 2021

@author: timalph
"""
import Utils
import numpy as np
from scipy.io import loadmat
import assignNN
import pickle
import mat_funcs
from sklearn import svm
from libsvm.svmutil import svm_train
import train
import math
import pandas as pd
import sys, os
import PedroPascal

def test(img, params, matPATH):
    
    prSize, pcSize, pzSize, nExtra = Utils.getCanonicalPatchHOGSize(params)
    print('prSize: ', prSize)
    print('pcSize: ', pcSize)
    print('pzSize: ', pzSize)
    print('pExtra', nExtra)
    
    workspace = loadmat(matPATH + 'FeaturesForLevel.mat')
    gimg = workspace['gradimg']
    level = workspace['level']
    
    
    f, i, g = Utils.getFeaturesForLevel(level, params, prSize, pcSize, pzSize, nExtra, gimg)
    
    ## Accidentaly overwrote these workspaces so the matfiles are incorrect
    assert np.allclose(f, workspace['feats'])
    print(np.sum(abs(f-workspace['feats'])))
    assert np.allclose(i+1, workspace['indexes'])
    assert np.allclose(g, workspace['gradsums'].flatten())
    
    
    pmid = Utils.constructFeaturePyramidForImg(img, params)
    f, l, i, g = Utils.unentanglePyramid(pmid, params)
    
    workspace = loadmat(matPATH + 'unentanglePyramid.mat')
    features, levels = workspace['features'], workspace['levels']
    indexes, gradsums = workspace['indexes'], workspace['gradsums']
    
    test_similarity(f, features)
    test_similarity(l+1, levels.flatten())
    test_similarity(i+1, indexes)
    test_similarity(g, gradsums.flatten())
    
    
    return g, gradsums

def test_similarity(a,b):
    
    if a.shape != b.shape:
        print(a.shape, b.shape)
        raise ValueError("Shapes don't match!")
    
    sum_a, sum_b = np.sum(a), np.sum(b)
    var_a, var_b = np.var(a), np.var(b)
    std_a, std_b = np.std(a), np.std(b)
    avg_diff = np.mean(abs(a-b))
    #print('Sums: ', sum_a, sum_b)
    #print('Vars, ', var_a, var_b)
    #print('Stds, ', std_a, std_b)
    
    print('Sum diff: ', sum_a - sum_b)
    print('Var diff: ', var_a - var_b)
    print('Std diff: ', std_a - std_b)
    print('Avg diff: ', avg_diff)
    
    
def try_assert(fun, a, b):
    try:
        c = a == b
        assert c.all()
    except:
        try:
            assert np.allclose(a, b)
        except:
            print(fun + ' similarity: ') 
            test_similarity(a, b)

def compare_df():
    return


    
def run_tests():
    matPATH = '/Users/timalph/Documents/Paris/release/'
    
    ## Test getcenters
    workspace = loadmat(matPATH + 'dscenters.mat')
    IF = workspace['IF']
    DC = workspace['DC']
    
    out = Utils.getcenters(IF)
    
    try_assert('getcenters', out, DC)
    
    ## Test assigntoclosest =============
    workspace = loadmat(matPATH + 'assigntoclosest.mat')
    closest = workspace['closest']
    outdist = workspace['outdist']
    toassign = workspace['toassign']
    targets = workspace['targets']
    
    c, o = assignNN.assigntoclosest(toassign, targets)
    
    try_assert('assigntoclosest', c+1, closest)
    try_assert('assigntoclosest', o, outdist)
    
    workspace = loadmat(matPATH + 'assigntoclosest_50.mat')
    closest = workspace['closest']
    outdist = workspace['outdist']
    toassign = workspace['toassign']
    targets = workspace['targets']
    
    c, o = assignNN.assigntoclosest(toassign, targets, step=50)
    
    try_assert('assigntoclosest', c+1, closest)
    try_assert('assigntoclosest', o, outdist)
    
    ## ===================================
    ## Test assignNN
    ## ===================================
    
    workspace = loadmat(matPATH + 'assignNN.mat')
    im = workspace['im']
    features = workspace['features']
    levels = workspace['levels']
    indexes = workspace['indexes']
    gradsums = workspace['gradsums']
    
    d = open('ds_assignNN.pickle', 'rb')
    ds = pickle.load(d)
    
    f, l, i, g, _ = assignNN.NN_for_single_image(im, ds, 0, testing = True)
    
    try_assert('assignNN features', f, features)
    try_assert('assignNN levels', l, levels.squeeze() -1)
    try_assert('assignNN indexes', i, indexes-1)
    try_assert('assignNN gradsums', g, gradsums.squeeze())
    
    
    f = open('ds_checkpoint_nearest_neighbours.pickle', 'rb')
    ds = pickle.load(f)
    
    workspace = loadmat(matPATH + 'create_topn.mat')
    assignednn = workspace['assignednn']
    assignedidx = workspace['assignedidx'][0]
    
    topndist = workspace['topndist']
    topnidx = workspace['topnidx']
    topnlab = workspace['topnlab']    
    
    tnd, tnl, tni = assignNN.create_topn(ds, 250, assignednn, 
                                         test = True, assignedidx = assignedidx)
    
    #tni[:,:,0] += 1
    
    try_assert('topndist ', tnd, topndist)
    try_assert('topnidx ', tni, topnidx -1)
    try_assert('topnlab ', tnl, topnlab)
    
    
    workspace = loadmat(matPATH + 'simplifydets.mat')
    
    assignedClust = workspace['assignedClust']
    currdets = workspace['currdets']
    detections = workspace['detections']
    imidx = workspace['imidx']
    
    detections = mat_funcs.struct2pandas(detections)
    
    res = assignNN.simplifydets(detections, imidx.squeeze(), 
                                assignedClust.squeeze(), configuration='First')
    
    res_mat = mat_funcs.struct2pandas(currdets)
    
    try_assert('simplifydets ', res.to_numpy(), res_mat.to_numpy())
    
    
    
    ## ===================================
    ## Pyridx2pos
    ## ===================================
    
    
    #metadata = assignNN.pyridx2pos(np.array([5,11]), 0.7449, 1.4142, 8, 8, 8, [537,936])
    
    
    
    ## ===================================
    ## Create_topndetshalf
    ## ===================================
    
    matPATH = '/Users/timalph/Documents/Paris/release/'
    
    workspace = loadmat(matPATH + 'topndets_initpatches')
    dsinitPatches = mat_funcs.struct2pandas(workspace['dsinitpatches'])
    
    workspace = loadmat(matPATH + 'create_topndetshalf')
    
    postord_m = workspace['postord'].squeeze()
    topnidx_m = workspace['topnidx']
    topndist_m = workspace['topndist']
    dsmyiminds = workspace['dsmyiminds']
    dspyrcanosz = workspace['dspyrcanosz']
    dspyrscales = workspace['dspyrscales']
    dsselectedClust = workspace['dsselectedClust'].squeeze()
    dsperclustpost = workspace['dsperclustpost']
    
    
    
    
    test = [postord_m, topnidx_m, topndist_m, 
            dsmyiminds, dspyrcanosz, dspyrscales, dsselectedClust, dsperclustpost, dsinitPatches]
    
    mainflag = True
    
    f = open('create_topndetshalf_ds.pickle', 'rb')
    ds = pickle.load(f)
    

    topndets_out, topndetshalf_out, topndetstrain_out, topnorig_out, _ = assignNN.create_topndetshalf(ds, postord_m, topnidx_m, topndist_m, mainflag, nneighbors=35, test=test)

    workspace = loadmat(matPATH + 'topndets_out')
    
    topndets_out_m = unpack(workspace['topndets'])
    topndetshalf_out_m = unpack(workspace['topndetshalf'])
    topndetstrain_out_m = unpack(workspace['topndetstrain'])
    
    topnorig_out_m = mat_funcs.struct2pandas(workspace['topnorig'])

    print('checking topndetshalf...')
    for idx, (x,y) in enumerate(zip(topndetshalf_out, topndetshalf_out_m)):
        
        x['detector'] +=1
        if x.equals(y):
            print('dataframe ', idx, ' is equal')
        else:
            print('dataframe ', idx, ' is not equal')
    
    print('checking topndetstrain...')
    for idx, (x,y) in enumerate(zip(topndetstrain_out, topndetstrain_out_m)):
        
        x['detector'] +=1
        if x.equals(y):
            print('dataframe ', idx, ' is equal')
        else:
            print('dataframe ', idx, ' is not equal')
            
    print('checking topnorig...')
    topnorig_out_m['detector'] -=1
    topnorig_out_m['detector'] = topnorig_out_m['detector'].astype('uint8')
    topnorig_out_m['count'] = topnorig_out_m['count'].astype('uint8')

    if topnorig_out.equals(topnorig_out_m):
        print('dataframe is equal')
    else:
        print('dataframe is not equal')
    
    
    
    print('checking extractpatches')
    
    
    workspace = loadmat(matPATH + 'extractpatches')
    detsimple = mat_funcs.struct2pandas(workspace['topndetstrain']) 
    detsimple['imidx'] -=1
    trpatches_m = workspace['trpatches'].squeeze()
    order = workspace['ord'].squeeze() - 1
    f = open('extractpatches_ds.pickle', 'rb')
    ds = pickle.load(f)
    trpatches = Utils.extractpatches(ds, detsimple, False, test = order)
    
    for i in range(len(trpatches)):
            try_assert('trpatches ', trpatches[i] * 255, trpatches_m[i])
    
    ## can be asserted manually like this
    ## Output was asserted manually and it worked. 
# =============================================================================
#     for i in range(len(trpatches)):
#         x = trpatches[i]
#         y = trpatches_m[i]
#         f = plt.figure()
#         f.add_subplot(1,2,1)
#         plt.imshow(x)
#         f.add_subplot(1,2,2)
#         
#         plt.imshow(y)
#         plt.show()
#     
# =============================================================================
    

    print('testing getMinimalModel...')    
    
    workspace = loadmat('svm_input.mat')
    
    labels = workspace['labels'].squeeze()
    features = workspace['features']
    
    mw = workspace['w']
    
    flags = '-s 0 -t 0 -c 0.1'
    model = svm_train(labels, features, flags)
    
    suppVec = model.get_SV()
    suppVec = np.nan_to_num(train.unpack_SV(suppVec))

    #Unsure how to achieve the same result with suppVec as in Matlab. Seems that
    #an issue persists between row first or column first sparsity.


    coeff = model.get_sv_coef()
    coeff = np.array(coeff)
    
    coeff = np.tile(coeff, (1, suppVec.shape[1]))
    
    
    
    minModel_rho = model.rho[0]
    # this w does not seem to be the same as the matlab w
    wpre = coeff * suppVec
    minModel_w = np.sum(wpre, axis=0)

    try_assert('svm_w', minModel_w, mw)
    
    
    print('testing myNms')
    workspace = loadmat(matPATH + 'myNms_var.mat')
    
    boxes = workspace['boxes']
    overlap = workspace['overlap']
    pick_mat = workspace['pick']
    
    pick = train.myNms(boxes, overlap)
    
    try_assert('myNms', pick, pick_mat.flatten()-1)
    
    print('test initialize')
    
    workspace = loadmat(matPATH + 'svmspace.mat')
    
    file = open('ds_for_svmtesting.pickle', 'rb')
    ds = pickle.load(file)
    ds['conf']['params']['imageCanonicalSize'] = 400
    
    
    ds['batch']['round']['selectedClust'] = workspace['selectedClust'].flatten() - 1
    ds['batch']['round']['assignedClust'] = workspace['assignedClust'].flatten() - 1
    ds['batch']['round']['posFeatures'] = workspace['posFeatures']
    ds['initFeatsNeg'] = workspace['initFeatsNeg']
    firstDet_mat = workspace['svm_firstDet']
    firstResult_mat = workspace['svm_firstResult']
    
    
    ds = train.initialize(ds)
    rhos = []
    ws = []
    for df in firstDet_mat.flatten():
        df = df[0][0]
        rho, w, firstLabel, info, threshold = df
        rhos.append(rho)
        ws.append(w.todense())
    
    for idx, rho in enumerate(rhos):
        try_assert('rho', ds['batch']['round']['firstDet'][idx]['rho'], rho)
    for idx, w in enumerate(ws):
        try_assert('w', ds['batch']['round']['firstDet'][idx]['w'], w)
    
    dets = train.VisualEntityDetectors(ds['batch']['round']['firstDet'],
                                       ds['conf']['params'])
    ds['batch']['round']['detectors'] = dets
    #im = Utils.getimg(ds, iminds[dsidx])
    
    ### testing with identical image to matlab
    print('testing autoclust_mine_negs')
    im = Utils.getimg(ds, 24)
    
    #dets = ds['batch']['round']['detectors'].detectPresenceInImg(im, ds)
    
    
    print('testing mySvmPredict')
    ##DetectPresenceInImg======================================================
    detectionParams = {'selecTopN': False,
                       'useDecisionThresh': True,
                       'overlap': 0.4,
                       'fixedDecisionThresh': -1.002,
                       'removeFeatures': 0}
    detectors = {}
    detectors['firstLevel'] = dets.firstLevModels
    detectors['secondLevel'] = dets.secondLevModels
    ##=========================================================================
    
    workspaceobj = loadmat(matPATH + 'ObjForTestingmySvmPredict.mat')
    labels = workspaceobj['labels']
    features = workspaceobj['features']
    decisionm = workspaceobj['decision']
    selectedm = workspaceobj['selected']
    
    _, _, decision = train.mySvmPredict(labels, features, detectors['firstLevel'])
    
    try_assert('decision', decision, decisionm)
    
    
    selected = train.doSelectionForParams(detectionParams, decision)
    
    try_assert('selected', selected, selectedm)
    params = ds['conf']['params']
    im = Utils.getimg(ds, 24)
    
    ##==detectPresenceUsingEntDet=============================================
    pyramid = Utils.constructFeaturePyramidForImg(im, params, [])
    ##====getDetectionsForEntDets=============================================
    features, levels, indexes, gradsums = Utils.unentanglePyramid(pyramid, params)
    
    thresh = gradsums>=9
    toss = len(gradsums) - sum(thresh)
    
    features = features[thresh]
    levels = levels[thresh].astype(int)
    indexes = indexes[thresh]
    gradsums = gradsums[thresh]
    
    print('threw out ', toss, 'patches!' )
    ##======ConstructResultStruct=============================================
    detections = train.constructResultStruct(pyramid, 8, 8, len(features),
                                   features, decision, levels, indexes, 
                                   selected, detectionParams, im, ds)
    
    ##====constructResults====================================================
    
    results = {}

    results['firstLevel'] = train.constructResults(detections, detectionParams['removeFeatures'])
    
    workspace = loadmat(matPATH + 'ObjForTestingConstructResults.mat')
    workspace2 = loadmat(matPATH + 'ObjMetadataForTestingMineNegs.mat')
    
    md = results['firstLevel']['detections']['metadata']
    
    ### Construct results
    print('Checking Metadata...')
    for i in range(len(md.keys())):
        df = pd.DataFrame.from_dict(md[i], orient = 'index')
        
        matkey = 'metadata' + str(i+1)
        matmetadata = workspace2[matkey]
        df_mat = mat_funcs.struct2pandas(matmetadata, topndets_variation = True, metadata_variation=True, no_print=True)
        compare_metadata(df, df_mat, issues=True)
    
    print('Checking decisions...')
    dec = results['firstLevel']['detections']['decision']
    for i in range(len(dec.keys())):
        key = 'decide' + str(i+1)
        try_assert('Decisions', dec[i], workspace2[key].squeeze())
        
    #======================
    
    istrain = np.zeros(len(ds['imgs'][ds['conf']['currimset']]))
    istrain[ds['batch']['round']['totrainon']] = 1
    allnegs = np.where(np.logical_and(ds['ispos']^1, istrain))[0]
    currentInd = 0
    maxElements = len(allnegs)
    iteration = 0
    startImgsPerIter = 15
    alpha = 0.71
    imgsPerIter = math.floor(startImgsPerIter * 2**(iteration * alpha))
    finInd = min(currentInd + imgsPerIter, maxElements)
    ds['batch']['round']['negmin'] = {}
    ds['batch']['round']['negmin']['iminds'] = allnegs[currentInd:finInd]
    iminds = ds['batch']['round']['negmin']['iminds']

    #======================
    
    dets = ds['batch']['round']['detectors'].detectPresenceInImg(im, ds)
    selectedClust = ds['batch']['round']['selectedClust']
    
    
    tmpdets=assignNN.simplifydets(dets, iminds[0])
    print('tmpdets', tmpdets.shape)
    
    
    
    ## Test Autoclust mine negs
# =============================================================================
#     
#     dets = train.VisualEntityDetectors(ds['batch']['round']['firstDet'],
#                                    ds['conf']['params'])
#     ds['batch']['round']['detectors'] = dets
# =============================================================================
    

    workspace = loadmat(matPATH + 'ObjForTrainNegs')
    
    ds['batch']['round']['selectedClust'] = workspace['selclust'].T.squeeze()
    ds['batch']['round']['assignedClust'] = workspace['assclust'].T.squeeze()
    ds['batch']['round']['posFeatures'] = workspace['posfeat']
    ds['batch']['round']['negmin']['iminds'] = workspace['iminds'].T.squeeze()
    ds['initFeatsNeg'] = workspace['initFeatsNeg']
    ## Test Autoclust Train Negs
    
    

            
    
    
    
    
    
    
    
    ## Test Autoclust Detect
    ## Test Autoclust TopN
    
    
    
    
    
    
    
    
    print('All tests done!')

## Compare dataframes of metadata for a df and its matlab equivalent
def compare_metadata(df, df_mat, issues=False):
    
    issue = 0
    
    for x, y in df[['x1', 'y2']].to_numpy():
        try:
            r = df.loc[(df['x1'] == x) & (df['y2'] == y)]
            rm = df_mat.loc[(df_mat['x1'] == x + 1) & (df_mat['y2'] == y+1)]
        except ValueError:
            print(df.loc[df['x1'] == x][['x1','y1','x2','y2']],
                  df_mat.loc[df_mat['x1'] == x+1][['x1','y1','x2','y2']])
            df[['x1', 'y2']].to_numpy().next()
            continue
        
        ## Check if the relevant column values are equal
        assert list(df.columns) == list(df_mat.columns)
        
        ccols = ['x1', 'y1', 'x2', 'y2']
        
        val = r[ccols].values == rm[ccols].values-1
        if not val.all():
            print('coords are not equal!:',r, rm)
            issue += 1

        val = list(r['pyramid'].values[0]+1) == rm['pyramid'].values[0]
        if not val:
            print('pyramids are not equal!:', r, rm)
            issue += 1
            
        rest = ['trunc', 'flip', 'size']
        val = r[rest].values == rm[rest].values
        if not val.all():
            print('rest values are not equal!:', r, rm)
            issue += 1

    if issue==0:
        print('Metadata is equal!')
    else:
        if issues:
            print('There are', issues, 'issues.')



def unpack(x, var=True):
    
    l = []
    
    for i in x:
        l.append(mat_funcs.struct2pandas(i, topndets_variation=True))
        
    return l




### test_exc(lusive) is a test function written to test autoclust mine negs, 
### " train negs, " detect, and " topN which I lost track of in the large function.
    

def test_exc():
    
    matPATH = '/Users/timalph/Documents/Paris/release/'
    
    ## Unpacking the workspace for train.initialize
    workspace = loadmat(matPATH + 'svmspace.mat')
    
    file = open('ds_for_svmtesting.pickle', 'rb')
    ds = pickle.load(file)
    ds['conf']['params']['imageCanonicalSize'] = 400
    
    
    ds['batch']['round']['selectedClust'] = workspace['selectedClust'].flatten() - 1
    ds['batch']['round']['assignedClust'] = workspace['assignedClust'].flatten() - 1
    ds['batch']['round']['posFeatures'] = workspace['posFeatures']
    ds['initFeatsNeg'] = workspace['initFeatsNeg']
    
    ds['batch']['round']['traintopN'] = {}
    ds['batch']['round']['validtopN'] = {}
    ds['batch']['round']['alltopN']   = {}
    
    ds = train.initialize(ds)
    
    ## Continuing main.py script
    
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
    
    workspace = loadmat(matPATH + 'ObjForTrainNegs')
    ds['batch']['round']['selectedClust'] = workspace['selclust'].T.squeeze() -1
    ds['batch']['round']['assignedClust'] = workspace['assclust'].T.squeeze() -1
    ds['batch']['round']['posFeatures'] = workspace['posfeat']
    #ds['batch']['round']['negmin']['iminds'] = workspace['iminds'] -1
    allnegs = np.array([25, 30, 44, 53, 63, 65, 66, 71, 76, 77, 80, 86]) -1
    ds['initFeatsNeg'] = workspace['initFeatsNeg']
    
    
    
#=============================================================================
    if not 'mineddetectors' in ds['batch']['round']:
        
        try:
            ds['batch']['round'].pop('negmin')
        except:
            pass
        
        while currentInd <= maxElements:
            imgsPerIter = math.floor(startImgsPerIter * 2**(iteration * alpha))
            finInd = min(currentInd + imgsPerIter, maxElements)
            ds['batch']['round']['negmin'] = {}
            ds['batch']['round']['negmin']['iminds'] = allnegs[currentInd:finInd]
            
            try:
                conf['noloadresults'] = 1
            except NameError:
                conf = {'noloadresults' : 1}
            
            
            ##### You need to make sure what goes into mine negs is the same as in matlab
            ##### That way you can use the same detections
            
            ds = train.autoclust_mine_negs(ds)

            ds = autoclust_train_testenv(ds)
            
            ## Not necessary as its already loaded in the ds
            #Utils.dsload('ds.batch.round.nextnegmin', 'traineddetectors')
            
            dets = train.VisualEntityDetectors(ds['batch']['round']['nextnegmin']['traineddetectors'], ds['conf']['params'])
            
            ds['batch']['round']['detectors']  = dets
            ds['batch']['round']['negmin'] = ds['batch']['round'].pop('nextnegmin')
            
            iteration +=1
            currentInd += imgsPerIter
            
        ds['batch']['round'].pop('negmin')
#=============================================================================
    ### load iminds from MATLAB
    
    
    
    ds['batch']['round']['iminds'] = loadmat(matPATH + 'ObjForTestDetect')['iminds'].T.squeeze()-1
    ds['batch']['round']['mineddetectors'] = dets  
    
    ds = train.autoclust_detect(ds)

    ##### test TopN


    workspace = loadmat(matPATH + 'ObjForTestACTopN')    
    
    ds['batch']['round']['tovalon'] = workspace['tovalon'].T.squeeze() - 1
    ds['batch']['round']['totrainon'] = workspace['totrainon'].T.squeeze()- 1
    ds['batch']['round']['iminds'] = workspace['iminds'].T.squeeze() - 1
    
    ### Entering TopN
    dsidx = 0
    
    imgs = ds['imgs']
    ispos = ds['ispos']
    
    numTopN = 20
    maxOverlap = 0.1
    
    detObj = PresenceDetectionResults2_testenv([dsidx], 
                                       len(ds['batch']['round']['selectedClust']), 
                                       ds['batch']['round']['iminds'])
            
    
    #### load all topdetsmaps from matlab and store them in the same dir struct
    #### as you've done for your pyth objects.
    ds['batch']['round']['topdetsmatlab'] = {}
    for i in range(38):
        workspace = loadmat(matPATH + 'topdetsmap[]/' + str(i+1))
        for j in range(12):
            key = 'data' +str(j+1)
            try:
                d = mat_funcs.unpack_topdetsmap(workspace[key])
                try:
                    ds['batch']['round']['topdetsmatlab'][j][i] = d
                except KeyError:
                    ds['batch']['round']['topdetsmatlab'][j] = {}
                    ds['batch']['round']['topdetsmatlab'][j][i] = d
            except KeyError:
                if key not in workspace.keys():
                    #print(key, 'not in workspace!')
                    pass
    for dsidx in range(len(ds['batch']['round']['selectedClust'])):     
        ds = train.autoclust_topn(ds, dsidx, matlab=True)
    
    ## Pulling the output from a .mat file
    workspace = loadmat(matPATH + 'ObjForTesttopNOutput')   
    
    traintopn_m = {}
    for idx, topn in enumerate(workspace['traintopn'][0]):
        traintopn_m[idx] = mat_funcs.unpack_matlabTOPN(topn)
    validtopn_m = {}
    for idx, topn in enumerate(workspace['validtopn'][0]):
        validtopn_m[idx] = mat_funcs.unpack_matlabTOPN(topn)
    alltopn_m = {}
    for idx, topn in enumerate(workspace['alltopn'][0]):
        alltopn_m[idx] = mat_funcs.unpack_matlabTOPN(topn)
        
    traintopn = ds['batch']['round']['traintopN']
    validtopn = ds['batch']['round']['validtopN']
    alltopn = ds['batch']['round']['alltopN']
        
    for struct, struct_m in zip([traintopn, validtopn, alltopn], 
                                [traintopn_m, validtopn_m, alltopn_m]):
        assert struct.keys() == struct_m.keys()
        for key in struct.keys():
            ## Testing meta is hard because of list array discrepancies
            for k in ['scores', 'imgIds']:
                try_assert(k, struct[key][k], struct_m[key][k])
                ### Metadeta hasn't been assessed yet because it takes time to compare those as I have to hardcode the comparison
            compare_metadata2(struct[key]['meta'], struct_m[key]['meta'])

def compare_metadata2(d, dm):
    
    for i in range(len(d)):
        di = d[i]
        dmi = dm[i]        
        for key in di.keys():
            if key == 'im':
                assert di[key][2:-2] == dmi[key]
            elif key == 'pyramid':
                assert di[key] == list(dmi[key])
            else:
                assert di[key] == dmi[key]




    ## necessary
# =============================================================================
# 
#     dets = train.VisualEntityDetectors(ds['batch']['round']['firstDet'],
#                                        ds['conf']['params'])
#     ds['batch']['round']['detectors'] = dets
#     
#     ds['batch']['round']['negmin']['iminds']
#     
#     ds = train.autoclust_mine_negs(ds)
# =============================================================================
    

#### This function runs exactly the same as the matlab equivalent
## In order to run this it needs to have access to the same detections matlab
## got. This is because our detections are slightly different due to very 
## low level differences in pyth/matlab imageprocessing standards.
def autoclust_train_testenv(ds):
    
    for dsidx, clustId in enumerate(ds['batch']['round']['selectedClust']):
    
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
        
        for k in range(len(iminds)):
            print('printing:', dsidx, k)
            
            detpath = '/Users/timalph/Documents/Paris/release/detections[]/'
            workspace = loadmat(detpath + str(k+1))
            try:
                struct = workspace['data' + str(dsidx+1)]
            except KeyError:
                ## The key doesn't exist, which signifies that the matlab 
                ## obj contained nothing. 
                continue
            
            df = mat_funcs.struct2pandas(struct)
            #obj = Utils.dsload('ds.batch.round.negmin.detections', detections = [dsidx, k+1])
            
            ## MATLAB script has a line of code to identify whether the further 
            ## images contain anything.
    
            alldets = alldets.append(df)
    
        #prevfeats not encountered yet
        try:
            prevfeats = Utils.dsload('ds.batch.round.negmin.prevfeats' + str(dsidx))
        
            if not (len(ds['batch']['round']['negmin']['prevfeats']) >= dsidx-1):
                prevfeats = ds['initFeatsNeg']
            elif len(ds['batch']['round']['negmin']['prevfeats']) == 0:
                prevfeats = ds['initFeatsNeg']    
        except:
            prevfeats = ds['initFeatsNeg']
        
        if len(alldets)>0:
            alldetfeats = np.array(alldets.features.tolist())
            allnegs = np.vstack((prevfeats, alldetfeats))
        else:
            allnegs = prevfeats
            
         
        features = np.vstack((posFeatures, allnegs))
        
        poslabels = np.ones(len(posFeatures))
        neglabels = np.ones(len(allnegs)) * -1
        
        labels = np.concatenate((poslabels, neglabels))
        
        print('Training SVM...')
        
        model = train.mySvmTrain(labels, features, ds['conf']['params']['svmflags'], False)
        
        selectedNegs = train.cullNegatives(allnegs, model, -1.02)
        
        allnegs = allnegs[selectedNegs]
        
        Utils.dssave('batch.round.nextnegmin.prevfeats', str(dsidx) , allnegs)
        
        try:
            ds['batch']['round']['nextnegmin']['traineddetectors'][dsidx] = model
        except KeyError:
            if not 'nextnegmin' in ds['batch']['round']:
                ds['batch']['round']['nextnegmin'] = {'traineddetectors' : {dsidx:model}}
            else:
                raise KeyError
    return ds


def autoclust_detect_testenv(ds):

    imgs = ds['imgs']
    
    cdir = ds['conf']['gbz'][ds['conf']['currimset']]['cutoutdir']
    imgframe = imgs[ds['conf']['currimset']]    
    for dsidx, imind in enumerate(ds['batch']['round']['iminds']):
        im = Utils.getimg(ds, imind)
        print('detecting...')
        dets = ds['batch']['round']['detectors'].detectPresenceInImg(im, ds)
        print('adding image metadata...')
        for key, val in dets['firstLevel']['detections']['metadata'].items():
            for k, v in val.items():
                v['im'] = cdir + imgframe.iloc[imind]['fullname']

        ds['batch']['round']['tmpdsidx'] = dsidx
        numTopN = 20
        maxOverlap = 0.1
        
        print('getResultData...')
        indstosave = []

        for i in reversed(range(len(ds['batch']['round']['selectedClust']))):
            clusti = ds['batch']['round']['selectedClust'][i]
            thisScores, imgMeta = train.getResultData(dets, i, maxOverlap)
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
        
        print('saving...')
        if len(indstosave) > 0:
            
            dir2save = {k : v for k, v in ds['batch']['round']['topdetsmap'].items()}
            Utils.dssave('ds.batch.round.topdetsmap', str(dsidx), dir2save)
        try:
            ds['batch']['round']['detectorflags'][dsidx] = 1
        except KeyError:
            ds['batch']['round']['detectorflags'] = {}
            ds['batch']['round']['detectorflags'][dsidx] = 1
    return ds    


class PresenceDetectionResults2_testenv():
    
    def __init__(self, idx, numclusters, iminds):

        self.myidx = idx
        self.myiminds = iminds        
        self.mynumclusters = numclusters
        
    def getPosResult(self, idd, ds):
        myidx = int(np.where(self.myiminds==idd)[0])
        result = {}
        #dsload(topdetsmap[self.myidx, idx])
        #if size(ds.batch.round.topdetsmap,2) >= idx
        #if len(ds['batch']['round']['topdetsmatlab'][0].keys()) >= myidx:
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
        #else:
        #    result = None
            
        return result
        
        
        
    def getNegResult(self, idd):
    
        return getPosResult(self, idd)
    
    def getNumClusters(self):
        return len(self.myidx)
    
    
def getTopNDetsPerCluster2_testenv(detectionResults, overlap, posIds, N, ds):
    
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
        print('idx:',idx)
        res = detectionResults.getPosResult(idx, ds)
        print(len(res[0]['scores']))
        if res == None:
            continue
        if len(res[0]['scores']) == 0:
            continue
        print(idx)
        for clusti in range(numClusters):
            if clusti == 0:
                total += Utils.numel(res[clusti]['scores'])
            scores[clusti]+=(list(res[clusti]['scores']))
            imgIds[clusti]+=(list(res[clusti]['imgIds']))
            meta[clusti]+=list(res[clusti]['meta'].values())
            
            if len(scores[clusti]) > maxCacheSize:
                raise ValueError("not implemented yet!")
                meta[clusti], scores[clusti], imgIds[clusti] = pickTopN_testenv(scores[clusti], imgIds[clusti], meta[clusti], maxToShow, overlap)
   
    print('got', total, 'results for first detector')
    
    topN = {}
    for i in range(numClusters):
        topN[i] = {}
        m, s, imid = pickTopN_testenv(scores[i], imgIds[i], meta[i], maxToShow, overlap)
        topN[i]['meta'] = m
        topN[i]['scores'] = s
        topN[i]['imgIds'] = imid
    return topN

def pickTopN_testenv(scores, imgIds, meta, numToPick, maxOverlap):
    
    #for the matlab +1
    imgIds = np.array(imgIds)-1
    
    _, ordered = cleanUpOverlapping_testenv(meta, scores, imgIds, maxOverlap)
    
    toSelect = min(len(ordered), numToPick)
    _, uniqueim = np.unique(imgIds[ordered], return_index=True)
    ordered=ordered[np.sort(uniqueim)]
    selected = ordered[:min(len(ordered), toSelect)]
    
    meta = [meta[k] for k in selected]
    scores = [scores[k] for k in selected]
    imgIds = imgIds[selected]
    
    return meta, scores, imgIds

def cleanUpOverlapping_testenv(patches, scores, correspImgs, maxOverlap):
    
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


def test_AC_detect():
    
    
    return

