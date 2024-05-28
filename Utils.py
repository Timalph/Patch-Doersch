#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 12:12:28 2021

@author: timalph
"""
import math
import numpy as np
from skimage import io

import os
from os import makedirs
import pickle
import matplotlib.pyplot as plt

from skimage.transform import rescale, resize
from skimage import img_as_float
import cv2 as cv

import hog
#https://github.com/fatheral/matlab_imresize Credit due here
#import matlab_imresize.imresize
import imresize_like_matlab
import time

#https://github.com/zzd1992/Numpy-Pytorch-Bicubic Credit due here
from pyResize import imresize
from numba import jit
from numba import cuda, float32

#Can't determine NUmba type of <class 'function'>
#@jit(nopython=True)
def constructFeaturePyramidForImg(img, params, levels = 0):
    fast_resize=True
    canonicalSize = params['imageCanonicalSize']
    sBins = params['sBins']
    
    ## this can go faster
    if not fast_resize:
        IS, canoScale = img2canonicalsize(img, canonicalSize)
    else:
        r, c, canoScale = speedimg2canonicalsize(img, canonicalSize)
        IS = cv.resize(img, (c, r))
    
    
    rows, cols, chans = IS.shape
    
    #Get the number of levels of the pyramid
    numLevels = getNumPyramidLevels(rows, cols, params['scaleIntervals'], params['patchCanonicalSize'])
    #Get the scales in intervals according to the levels of the pyramid
    scales = getLevelScales(numLevels, params['scaleIntervals'])
    
    ## Uses D50 as with Matlab.
    ## rgb2lab is a reimplementation of the Matlab function.
    if np.sum(img) > img.size:
        img_into_rgb = img/255
        im2 = rgb2lab(img_into_rgb) * .0025
    else:
        im2 = rgb2lab(img) * .0025
        
    histbin = np.linspace(-100,100, 11)
    histbin[-1]+=1
    
    pyramidLevs = {}
    gradientLevs = {}
    
    for idx, s in enumerate(scales):
        img = img_as_float(img)
        
        ## Resize the image for every defined scale.
        
        if not fast_resize:
            I1 = imresize_like_matlab.imresize(img, canoScale/s)
        
        else:        
            I1cols, I1rows = imresize_like_matlab.speedSizeFromScale(np.array(img.shape)[:2], canoScale/s)
            I1 = cv.resize(img, (I1rows, I1cols))

        nrows, ncols, _ = I1.shape
        rowRem = nrows%sBins
        colRem = ncols%sBins
        
        ## Clip the rows and columns so the scaled image is a multiple of sBins.
        
        if rowRem != 0 or colRem != 0:
            I1 = I1[:(int(nrows-rowRem)), :int((ncols-colRem)), :]  
        
        if 'patchOnly' in params:
            raise ValueError("Not Implemented Yet")
        
        ## Get HOG descriptor
        ## Get rid of in accuracies in matlab/python by rounding to 8 decimal points
        I1 = np.round(I1, 8)
        
        
        feat = hog.features(I1, sBins)
        rows,cols,_ = feat.shape
        
        
        ## Get the a and b channels from the LAB space image.
        #e1 = imresize(im2[:,:,1], output_shape=(rows, cols), kernel='linear')
        #e2 = imresize(im2[:,:,2], output_shape=(rows, cols), kernel='linear')         
        if not fast_resize:       
            e1 = imresize_like_matlab.imresize(im2[:,:,1], output_shape=(rows, cols), method='bilinear')        
            e2 = imresize_like_matlab.imresize(im2[:,:,2], output_shape=(rows, cols), method='bilinear')  
        
        else:
            e1 = cv.resize(im2[:,:,1], (cols, rows))
            e2 = cv.resize(im2[:,:,2], (cols, rows))
        
        ## Add the 31-channel HOG descriptor together with the 2 channels from the LAB image.
        feat = np.concatenate((feat,np.expand_dims(e1,2),np.expand_dims(e2,2)),2)
        
        #I1 = loadmat(matPATH + 'I1.mat')['I1']
        
        #There is already a slight error with I1 at this point(e-3)
        dG = np.gradient(I1)
        GX = dG[1]
        GY = dG[0]
        GI = np.mean((GX*255)**2,2) + np.mean((GY*255)**2,2)
        
        #Gradients are skewed due to differing imresize function.
        #Perhaps possible to get the error margin down by replacing by another imresize function. 
        #Leaving it as is for now, we'll have to take the error in stride.
        #GI = imresize(GI, output_shape=(rows,cols), kernel='linear')
        GI = cv.resize(GI, (cols, rows))
        
        pyramidLevs[idx] = feat
        gradientLevs[idx] = GI
        
    canoSize = {'nrows' : rows, 'ncols' : cols}
    pyramid = {'features' : pyramidLevs, 'scales' : scales, 'canonicalScale' : canoScale, 
               'sbins' : sBins, 'canonicalSize' : canoSize, 'gradimg' : gradientLevs}
    return pyramid


### Scale intervals(8) and Patch Canonical Size(TUPLE(80,80))
@jit(nopython=True)
def getNumPyramidLevels(r, c, Intervals, BasePatchSize):
    lev1 = np.floor(Intervals * np.log2(r/BasePatchSize[0]))
    lev2 = np.floor(Intervals * np.log2(c/BasePatchSize[1]))
    
    numlev = np.min(np.array([lev1, lev2])) + 1

    return numlev    

@jit(nopython=True)
def getLevelScales(numLevels, Intervals):
    sc = 2**(1/Intervals)
    scales = sc**np.arange(numLevels)
    return scales


#@jit(nopython=True)
def getGradientImage(I):
    
    GX, GY, _ = np.gradient(I)
    I1 = np.sum(abs(GX), axis=2) + np.sum(abs(GY), axis=2);
    I1 = I1**2
    
    return I1


    
@jit(nopython=True)
def rgb2lab(img):
    
    ## Reimplementation of Matlab RGB2LAB function
    
    w, h, _ = img.shape
    
    T = 0.008856
    
    if np.sum(img) > img.size:
        img = img/255

    RGB = np.stack((img[:,:,0].flatten(), img[:,:,1].flatten(), img[:,:,2].flatten()))
    
    MAT = np.array([[0.412453, 0.357580, 0.180423],
           [0.212671, 0.715160, 0.072169],
           [0.019334, 0.119193, 0.950227]])
    
    XYZ = np.dot(MAT, RGB)

    X = XYZ[0,:] / 0.950456
    Y = XYZ[1,:]
    Z = XYZ[2,:] / 1.088754
    
    XT = X > T
    YT = Y > T
    ZT = Z > T
    
    
    fX = XT * (np.sign(X) * (np.abs(X))**(1/3)) + (~XT) * (7.787 * X + 16/116)
    
    # Compute L
    Y3 = np.sign(Y) * (np.abs(Y))**(1/3)
    fY = YT * Y3 + (~YT) * (7.787 * Y + 16/116)
    L  = YT * (116 * Y3 - 16.0) + (~YT) * (903.3 * Y)

    fZ = ZT * (np.sign(Z) * (np.abs(Z))**(1/3)) + (~ZT) * (7.787 * Z + 16/116)
    
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)
    
    L = np.reshape(L, (w, h))
    a = np.reshape(a, (w, h))
    b = np.reshape(b, (w, h))
    
    return np.stack((L,a,b),2)

@jit
def speedimg2canonicalsize(img, csize):
    
    #assert csize == 400
    
    r, c, _ = img.shape
    scale = csize/r
    new_r, new_c = imresize_like_matlab.speedSizeFromScale(np.array([r,c]), scale)
    return new_r, new_c, scale
    
def img2canonicalsize(img, csize):
    r, c, _ = img.shape
    
    if r<c:
        scale = csize/r
        return imresize_like_matlab.imresize(img, scale), scale
    else:
        scale = csize/c
        return imresize_like_matlab.imresize(img, scale), scale



def showim(im):
    plt.imshow(im, interpolation='nearest')
    plt.show()
    
def getCanonicalPatchHOGSize(params):
    
    prSize = int(np.round(params['patchCanonicalSize'][0] / params['sBins']) - 2)
    pcSize = int(np.round(params['patchCanonicalSize'][1] / params['sBins']) - 2)
    
    pExtra = 0
    
    try:
        if params['useColor'] == 1:
            pzSize = 33
    except:
        try:
            if params['patchOnly'] == 1:
                pzSize = 1
            else:
                pzSize = 31
        except:
            try:
                if params['useColorHists'] == 1:
                    pExtra=20
            except:
                pass
    return prSize, pcSize, pzSize, pExtra
#@jit(nopython=True)
#params,prSize, pcSize, pzSize, nExtra, gradimg
@jit(nopython=True)
def speedFeaturesForLevel(level, prSize, pcSize, pzSize, nExtra, gradimg):
    rows, cols, dims = level.shape

    rLim = rows - prSize + 1;
    cLim = cols - pcSize + 1;
    
    featDim = prSize * pcSize * pzSize + nExtra
    features = np.zeros((rLim * cLim, featDim))
    
    gradsums = np.zeros(rLim * cLim)
    indexes = np.zeros((rLim * cLim, 2))
    featInd = 0
    
    for j in range(cLim):
        for i in range(rLim):
            #this returns an array of prSize x pcSize x 33 which is 8 x 8 x 33 = 2112
            feat = level[i:i+prSize, j:j+pcSize,:]

            ## Apparently Matlab reshape is Fortran-style column major. 
            features[featInd, :] = feat.T.flatten()

            gradsums[featInd]=np.mean(gradimg[i:i+prSize, j:j+pcSize])
            indexes[featInd, :] = [i, j]
            featInd = featInd + 1
    
    return features, indexes, gradsums
    
def getFeaturesForLevel(level, params, prSize, pcSize, pzSize, nExtra, gradimg):
    
    #Level n-th image from the feature pyramid
    
    rows, cols, dims = level.shape

    rLim = rows - prSize + 1;
    cLim = cols - pcSize + 1;
    
    featDim = prSize * pcSize * pzSize + nExtra
    features = np.zeros((rLim * cLim, featDim))
    
    ## This isnt python
    if 'gradimg' in locals():
        gradsums = np.zeros(rLim * cLim)
        create_grad = True
        print('gradimg')
    else:
        create_grad = False

    indexes = np.zeros((rLim * cLim, 2))
    featInd = 0
    for j in range(cLim):
        for i in range(rLim):
            #this returns an array of prSize x pcSize x 33 which is 8 x 8 x 33 = 2112
            feat = level[i:i+prSize, j:j+pcSize,:]
            feat = np.round(feat,8)
            ## Apparently Matlab reshape is Fortran-style column major. 
            features[featInd, :] = feat.flatten(order='F')
            if(create_grad):
                gradsums[featInd]=np.mean(gradimg[i:i+prSize, j:j+pcSize])
            indexes[featInd, :] = [i, j]
            featInd = featInd + 1
            
    return features, indexes, gradsums

#Cannot determine Numba type of <class 'function'>    
#@jit(nopython=True)
def unentanglePyramid(pmid, params):
    
    prSize, pcSize, pzSize, nExtra = getCanonicalPatchHOGSize(params)
    
    selFeatures = []
    selFeaturesInds = []
    selGradsums = []
    selLevel = []
    totalProcessed = 0

    for idx, (pmid_feature, pmid_gradimg) in enumerate(zip(pmid['features'].values(), pmid['gradimg'].values())):
        
        feats, indexes, gsum = speedFeaturesForLevel(pmid_feature, prSize,
                                                               pcSize, pzSize, nExtra, pmid_gradimg)
        
        #return feats, indexes, gsum
        selGradsums.append(gsum)
        selFeatures.append(feats)
        selFeaturesInds.append(indexes)
                                               
        numFeats = feats.shape[0]
        selLevel.append(np.ones(numFeats) * idx)
        totalProcessed = totalProcessed + numFeats
    
    features = np.concatenate(selFeatures) 
    gradsums = np.concatenate(selGradsums)
    levels = np.concatenate(selLevel)
    indexes = np.concatenate(selFeaturesInds)
    
    return features, levels, indexes, gradsums

    
#@jit(nopython=True)
def getMetadataForPositives(selected, level, indexes, prSize, 
                            pcSize, pyramid, im, ds, suppress_warning=False):
    
    metadata = {}
    pos = im
    
    canoSc = pyramid['canonicalScale']
    
    for idx, selInd in enumerate(selected):
        levelPatch = getLevelPatch(prSize, pcSize, level[selInd], pyramid)
        levSc = pyramid['scales'][level[selInd]]
        x1 = indexes[selInd, 1]
        y1 = indexes[selInd, 0]
        
        ## Are these offsets correct? do we need do subtract 1 from them 
        ## for Matlab porting?
        xoffset = math.floor((x1) * pyramid['sbins'] * levSc / canoSc)
        yoffset = math.floor((y1) * pyramid['sbins'] * levSc / canoSc)
        thisPatch = levelPatch + np.array((xoffset, xoffset, yoffset, yoffset))
        
        
        
        metadata[idx] = {'im': [], 'x1' : [], 'x2' : [], 'y1' : [], 'y2' : [], 
                'flip' : [], 'trunc' : [], 'size' : []}
        metadata[idx]['x1'] = thisPatch[0];
        metadata[idx]['x2'] = thisPatch[1];
        metadata[idx]['y1'] = thisPatch[2];
        metadata[idx]['y2'] = thisPatch[3];
        
        
        if type(im) == int  or im.size < 3:
            metadata[idx]['im'] = ds['conf']['gbz'][ds['conf']['currimset']]['cutoutdir'] + ds['imgs'][ds['conf']['currimset']].iloc[pos]['fullname']
            sz=ds['imgs'][ds['conf']['currimset']].iloc[pos]['imsize']
            metadata[idx]['size'] = {'nrows':sz[0] , 'ncols':sz[1]}
            metadata[idx]['imidx'] = pos
            
            try:
                numel = im.size
                if numel > 1:
                    metadata[idx]['setidx'] = im[1]
            except AttributeError:
                metadata[idx]['setidx'] = ds['conf']['currimset']
                
        else:
            metadata[idx]['size'] = {'nrows':im.shape[0], 'ncols' : im.shape[1]}
        metadata[idx]['flip'] = False
        metadata[idx]['trunc'] = False
        
        metadata[idx]['pyramid'] = np.append(np.array(level[selInd]), indexes[selInd])

        metadata[idx] = clipPatchToBoundary(metadata[idx], suppress_warning)
        
        #### Unsure if it's necessary to clip the patch
            #metadata[idx] = clipPatchToBoundary(metadata[idx])
        
            
        #else:
        #    print(im)
        #    print(type(im))
        #    raise ValueError("Im is supposed to be an integer for the metadata to fill.")
            
                
                
                
    
    return metadata

def clipPatchToBoundary(patch, suppress_warning=False):
    
    rows = patch['size']['nrows']
    cols = patch['size']['ncols']
    doWarn = False
    
    clip = {}
    
    if patch['x1'] < 0:
        clip['x1'] = patch['x1']
        patch['x1'] = 0
        doWarn = True
    
    if patch['y1'] < 0:
        clip['y1'] = patch['y1']
        patch['y1'] = 0
        doWarn = True
    
    if patch['x2'] >= cols:
        clip['x2'] = patch['x2']
        patch['x2'] = cols -1
        doWarn = True
        
    if patch['y2'] >= rows:
        clip['y2'] = patch['y2']
        patch['y2'] = rows -1
        doWarn = True
    
    if doWarn and not suppress_warning:
        print('A patch was clipped! Details:', patch['x1'], patch['y1'], ':', 
              patch['x2'], patch['y2'], rows, cols, clip)
    
    return patch


def getLevelPatch(prSize, pcSize, level, pyramid):
    levSc = pyramid['scales'][level]
    canoSc = pyramid['canonicalScale']

    levelPatch = np.array((0, round((pcSize + 2) * pyramid['sbins'] *
                                    levSc / canoSc) - 1, 0, 
                           round((pcSize + 2) * pyramid['sbins'] *
                                    levSc / canoSc) - 1))    
    return levelPatch





def setup_dir(out_dir):
    try:
        os.mkdir(out_dir)
    except:
        pass
    
    try:
        os.mkdir(os.path.join(out_dir , 'ds'))
    except:
        pass
    
    try:
        os.mkdir(os.path.join(out_dir ,  'ds/sys'))
    except:
        pass
    
    try:
        os.mkdir(os.path.join(out_dir , 'ds/sys/distproc'))
    except:
        pass
    
    
def getcenters(f):
    
    return (f - f.mean(axis=1, keepdims=True)) / ((np.sqrt(np.var(f,1, keepdims=True)) * f.shape[1]) + 1e-15)

def getimg(ds, idx, integer_pixels=False):
    
    cutout_path = ds['conf']['gbz'][ds['conf']['currimset']]['cutoutdir']
    img_path = ds['imgs'][ds['conf']['currimset']].iloc[idx]['fullname']
    file = cutout_path + img_path
    BGR_cutout_img = cv.imread(file)
    cutout_img = BGR_cutout_img[:, :, [2, 1, 0]]
    if not integer_pixels:
        cutout_img = cutout_img/256
    
    return cutout_img

def dsfield(ds, a, b):
    
    try:
        if b in ds[a]:
            return True
        else:
            return False
    
    except KeyError:
        return False


def prepbatchwisebestbin(ds, detsimple, batchidx, npatchesper=5, ranks=False):
    
    
    if not dsfield(ds, 'bestbin', 'alldisclabelcat'):
        try:
            ds['bestbin']['alldisclabelcat'] = []
            ds['bestbin']['alldiscpatchimg'] = {}
            ds['bestbin']['decision'] = []
        except:
            ds['bestbin'] = {}
            ds['bestbin']['alldisclabelcat'] = []
            ds['bestbin']['alldiscpatchimg'] = {}
            ds['bestbin']['decision'] = []
        if ranks:
            ds['bestbin']['ranks'] = []
        ds['bestbin']['group'] = []
            
    detsimple = detsimple.loc[detsimple['detector'] != 0]
    detectors = np.unique(detsimple['detector'])
    alldetectors = detsimple['detector'].to_numpy()
    alldecisions = detsimple['decision'].to_numpy()
    tokeep = np.zeros(alldetectors.shape)
    saveranks = np.zeros(detsimple.shape)
    
    for d in detectors:
        inds = detsimple.loc[detsimple['detector'] == d].index.to_numpy()
        inds2 = np.argpartition(alldecisions[inds], -npatchesper)[-npatchesper:]
        tokeep[inds[inds2]] = 1
    
        if ranks:
            print(inds)
            print(inds2)
            print(saveranks)
            saveranks[inds[inds2]] = ranks[:len(inds2)]
            #raise ValueError('Read the matlab function to see what ranks is supposed to do.')
            
    detsimple = detsimple[tokeep==1]
    
    if ranks:
        saveranks = saveranks[tokeep==1]
        saveranks = saveranks.flatten()
        #raise ValueError('Read the matlab function to see what ranks is supposed to do.')
    
    ds['bestbin']['alldisclabelcat'] = detsimple[['imidx', 'detector']].to_numpy()
    r = extractpatches(ds, detsimple, ds['conf']['currimset'], conf='noresize')
    
    pre_len = len(ds['bestbin']['alldiscpatchimg'])
    for key, val in r.items():
        ds['bestbin']['alldiscpatchimg'][key+pre_len] = val
    ds['bestbin']['decision'].append(detsimple['decision'].to_list())
    if ranks:
        ds['bestbin']['ranks'].append(saveranks)
        #raise ValueError('Read the matlab function to see what ranks is supposed to do.')        
    ds['bestbin']['group'].append([batchidx]*len(detsimple))
    ds['bestbin']['imgs'] = ds['imgs'][ds['conf']['currimset']]
    
    if 'iscorrect' in ds['bestbin']:
        raise ValueError('Not implemented yet, check MatLAB')
    
    ds['bestbin']['tosave'] = np.unique(ds['bestbin']['alldisclabelcat'][:,1])
    ds['bestbin']['tosave'].sort()
    ds['bestbin']['isgeneral'] = np.ones((len(ds['bestbin']['tosave'])))
        
    
    return ds


def extractpatches(ds, detsimple, imgs_all, conf=False, test=False):
    
    res = {}
    
    #imgs = np.empty(len(imgs_all))
    #imgs[:] = np.NaN
    
    imgs = {}
    
    loaded =[]
    order = detsimple['imidx'].sort_values().index

    try:
        if not test:
            pass
    except:
        order = test

    for dsidx in order:
        pos = detsimple.iloc[dsidx]['pos']
        i = detsimple.iloc[dsidx]['imidx']
        if not i in imgs:
            imgs[i] = getimg(ds, i, integer_pixels=True)
            try:
                loaded[loaded==i] = []
                loaded.append(i)
            except TypeError:
                loaded.append(i)
            if(len(loaded) > 1):
                _ = imgs.pop(loaded[0])
                _ = loaded.pop(0)
        
        
        if conf == 'noresize':
            res[dsidx] = imgs[i][pos['x1']:pos['x2'], pos['y1']:pos['y2'], : ]
        else:
            maxsz = max(pos['y2'] - pos['y1'], pos['x2'] - pos['x1'])
            reszx = math.ceil(80*(pos['x2'] - pos['x1'])/maxsz)
            reszy = math.ceil(80*(pos['y2'] - pos['y1'])/maxsz)
            im = imgs[i][pos['y1']:pos['y2']+1, 
                                        pos['x1']:pos['x2']+1]
            
            res[dsidx] = imresize_like_matlab.imresize(im,
                                        output_shape=(reszy, reszx, im.shape[2]))

    return res

def numel(x):
    
    if isinstance(x, (int, np.integer)):
        return 1
    else:
        return len(x)
        

## Trying to implement a functional copy of dssave used in the matlab implementation

## Takes path to file, filename, and data
## Path is taken like so: ds.batch.round for ~/ds/batch/round
def dssave(path, fname,  data):
    
    filename = fname + '.pickle'
    p = path.replace('.','/')
    if p[0] != '/':
        p = '/'+p
    current_path = os.getcwd()
    
    full = os.path.join(current_path + p)
    #print(full)
    try:
        os.makedirs(full)
    except FileExistsError:
        pass
        #print('The directory:', full, 'already exists!')
    
    with open(os.path.join(full,filename), 'wb') as h:
        pickle.dump(data, h)

def loadtopdetsmap(ds, idx):
    
    if not 'savedir' in ds:
        o = pickle.load(open('ds/batch/round/topdetsmap/{}.pickle'.format(idx), 'rb'))
        
    else:
        raise NotImplementedError
def dsload(path, detections=False):

    if detections:
        
        dsidx, k = detections
        
        p = path.replace('.','/')
        if p[0] != '/':
            p = '/'+p
        if p[-1] != '/':
            p += '/'
        current_path = os.getcwd()
        fname = str(k) + '.pickle'
        full = os.path.join(current_path + p + fname)
        
        f = open(full, 'rb')
        obj = pickle.load(f)
        
        try:
            obj = obj[dsidx]
        except KeyError:
            obj = None
    else:
        p = path.replace('.','/')
        if p[0] != '/':
            p = '/'+p

        current_path = os.getcwd()
        fname = '.pickle'
        full = os.path.join(current_path + p + fname)
        f = open(full, 'rb')
        obj = pickle.load(f)
        
    return obj

def makemarks(idx, sz):
    res = np.zeros(sz)
    res[idx] = 1
    return res

def timef(x, inp):
    start = time.time()
    x(inp)
    end = time.time()
    print(end-start)




