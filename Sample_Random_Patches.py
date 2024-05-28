#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 12:08:48 2021

@author: timalph
"""

import numpy as np
import math

from scipy.io import loadmat
import scipy.ndimage
from skimage import io
from time import time
#https://github.com/fatheral/matlab_imresize Credit due here
#import matlab_imresize.imresize
import Utils
import mat_funcs

import cv2 as cv
import os
import pickle
import matplotlib as plt

#sampleBig not implemented. Possible insertion at line 70 under levPatsize


## Sample_Random_Patches is a function that samples a max of samplelimit 
## patches from a singles image.
## The input is:
## ds = dict, datastructure that contains all the paths to the images you need
## pos = int, the index of the image you want to sample from
def main(pos, ds, samplelimit=-1, sd = 20, matlabinds=False, corr='quick'):
    
    params = ds['conf']['params']
    
    
    np.random.seed(sd)    
    
    ## Select the image based on ds and pos
    
    ## Path to the cutout directory that includes the subdirectories of the 
    ## different cities:
    ## You could alter these lines to use the function 'getimg'
    cutout_path = ds['conf']['gbz'][ds['conf']['currimset']]['cutoutdir']
    if cutout_path[-1] != '/':
        cutout_path+= '/'
    city_path = ds['mycity'] + '/'
    img_path = ds['imgs'][ds['conf']['currimset']].iloc[pos]['fullname']
    
    
    
    
    file = os.path.join(cutout_path , img_path)
    
    BGR_cutout_img = cv.imread(file)
    cutout_img = BGR_cutout_img[:, :, [2, 1, 0]]
    cutout_img = cutout_img/255
    
    
    PATH = os.path.join(cutout_path, city_path)
    
    levelFactor = 2
    
    ## IS is the image scaled so the smallest dimension is 400.
    IS, scale = Utils.img2canonicalsize(cutout_img, params['imageCanonicalSize'])
    
    rows, cols, _ = IS.shape
    
    ## IG is the result of taking the gradient in x and y direction and summing
    ## over all channels. It is then squared. 
    IG = Utils.getGradientImage(IS)
    
    
    
    ## Construct the feature pyramid which contains
    
    ## Explain num pyramid levels
    ## The size of the pyramid is decided by floor(intervals * log2(rows/basepatchsize))
    ## Explain level scales
    ##     sc = 2**(1/Intervals)
    ##     scales = sc**np.arange(numLevels)
    ## The featurepyramid is then a collection of hog + lab features on the 
    ## original image and increasingly downsized versions of it. 
    
    pmid = Utils.constructFeaturePyramidForImg(cutout_img, params)
    features, levels, indexes, gradsums = Utils.unentanglePyramid(pmid, params)
    
    levels = levels.astype('int')
    
    ## Select indices in steps of scaleIntervals/2. 
    selLevels = np.arange(0,len(pmid['scales'])-1,int(params['scaleIntervals']/2))
    ## Filter out the scales at which we want to sample, I assume.
    levelScales = pmid['scales'][selLevels]
    prSize, pcSize, _ , _= Utils.getCanonicalPatchHOGSize(params)
    
    
    patches = {}
    patFeats = np.zeros(0)
    probabilities = np.zeros(0)
    
    max_dict_idx = 0
    
    ## Select the last of the features of the selected indices.
    basenperlev = pmid['features'][selLevels[-1]]
    ## I think we're sampling a value from this feature array.
    basenperlev = (basenperlev[0,0,0] - prSize + 1) * (basenperlev[1,0,0] - pcSize + 1)
    
    for idx, scale in enumerate(levelScales):

        levPatSize = np.floor(params['patchCanonicalSize'] * scale)
        
        if 'sampleBig' in params:
            raise ValueError('Not Implemented!')
        else:
            r_new = (rows / (levPatSize[0] / levelFactor)) 
            c_new = (cols / (levPatSize[1] / levelFactor))
            numLevPat = np.floor(r_new * c_new * 2)
            
            levelPatInds = np.where(levels == selLevels[idx])[0]
    
            if numLevPat <= 0:
                raise ValueError("continue is supposed to be here!")
            
            IGS = IG
            
            #### ALERT DELETE #####
            #workspace = loadmat(matPATH + 'getRandForPdf.mat')
            #IGS = workspace['IGS']
            
            ## Old code before I figure out the exact distribution copy
            
            if corr == 'quick':
                pDist = getQuickDistribution(IGS, levPatSize, style='notmat')
            
            elif not matlabinds:
                IGS_obj = Image_object(IGS)
                pDist = IGS_obj.getProbDistribution(levPatSize)
            ##
            elif matlabinds:
                ### This method uses scipy correlate which takessoooo long
                totaltimestart = time()
                pDist = getProbDistribution(IGS, levPatSize)
                print('time for', idx, 'getDist:', time()-totaltimestart)
            pDist1d = pDist.flatten(order='F')
            
            
            ### Check if this function is correct, you deleted the +1
            ### Check what the lowest number is that this can give
            randNums = getRandForPdf(pDist1d, numLevPat)
            
            if matlabinds==True:
                ## Pull randNums from Matlab
                matPATH = '/Users/timalph/Documents/Paris/release/'
                workspace = loadmat(matPATH + 'randNums_for_Python')
                randNums = workspace['mat_randnums' + str(pos+1) + '_' + str(idx+1)].squeeze()-1
            
            elif matlabinds=='neg':
                matPATH = '/Users/timalph/Documents/Paris/release/'
                workspace = loadmat(matPATH + 'negnums_for_python')
                randNums = workspace['negnums_metadata' + '_' + str(pos+1) + '_' + str(idx+1)][0][0].squeeze()-1
            
            #pos = 18
            #pDist1d = workspace['pDist1d']
            #randNums = workspace['randNums']
            
            ### this -1 is here for matlab indexing, unsure if this is right 
            ### because I don't know what the lowest number is that randNums can get
            probs = pDist1d[randNums]
            
            IY, IX = mat_funcs.ind2sub(IGS.shape, randNums)
            #### CURRENTLY HERE
            IY = np.ceil((IY+1)/ (levelScales[idx] * params['sBins']))-1
            IX = np.ceil((IX+1)/ (levelScales[idx] * params['sBins']))-1
            
            nrows, ncols, _ = pmid['features'][selLevels[idx]].shape
            
            IY = IY - math.floor(prSize / 2)
            IX = IX - math.floor(pcSize / 2)
            
            xyToSel = ((IY>=0) * (IY<=nrows-prSize) * (IX>=0) * (IX<=ncols-pcSize))
        
            
            IY = IY[xyToSel]
            IX = IX[xyToSel]
            probs = probs[xyToSel]
            
            inds = sub2ind((nrows - prSize+1, ncols-pcSize+1), IY, IX)
            
            inds_u, m = np.unique(inds, return_index=True)

            inds_u = inds_u.astype('int')
            
            if len(probs) != 1:
                probs = probs.squeeze()[m]

            selectedPatInds = levelPatInds[inds_u]

            metadata = Utils.getMetadataForPositives(selectedPatInds, levels.astype(int), 
                                               indexes, prSize, pcSize, pmid, pos,
                                               ds, suppress_warning=True)
            
            ## slight error margin on the features here (5th decimal)
            feats = features[selectedPatInds,:]
            
            if len(metadata) > 0:
                patInds = cleanUpOverlappingPatches(metadata, params['patchOverlapThreshold'], probs)
                metadata_subset = {dict_idx + max_dict_idx: metadata[key] for dict_idx, key in enumerate(patInds)}
                patches.update(metadata_subset)
                max_dict_idx += len(metadata_subset)
                
                try:
                    patFeats = np.vstack((patFeats, feats[patInds]))
                except:
                    patFeats = feats[patInds]
                
                probabilities = np.append(probabilities, probs[patInds])

    if samplelimit <0:
        #### Samplelimit not implemented, change 25 to samplelimit. 
        inds = np.arange(len(patches))
        np.random.shuffle(inds)
        
        if matlabinds:
            f = open('matlab_tinydataset_randperm.pickle', 'rb')
            randperm_dict = pickle.load(f)
            inds = np.array(randperm_dict[pos+1])-1
            inds = np.delete(inds, np.where(inds>=len(patches)))
                
        
        patches = {idx: patches[key] for idx, key in enumerate(inds[:25])}
        patFeats = patFeats[inds[:25]]
        probabilities = probabilities[inds[:25]]
    elif samplelimit:
        inds = np.arange(len(patches))
        np.random.shuffle(inds)
        
        patches = {idx: patches[key] for idx, key in enumerate(inds[:samplelimit])}
        patFeats = patFeats[inds[:samplelimit]]
        probabilities = probabilities[inds[:samplelimit]]        
        
    return patches, patFeats, probabilities

def cleanUpOverlappingPatches(patches, thresh, probs):
    
    ## get probInds, not the actual sorted array probs[::-1].sort()
    probInds = probs.argsort()[::-1]
    patInds = np.zeros(len(patches))
    indCount = 0
    
    mask = np.zeros((patches[0]['size']['nrows'], patches[0]['size']['ncols']))
    nr = patches[0]['y2'] - patches[0]['y1'] + 1
    nc = patches[0]['x2'] - patches[0]['x1'] + 1
    patchArea= nr * nc

    ## Start with the patch with the highest probability
    ## This only adds patches to the list if the overlap of new patch with 
    ## the union of all
    ## previous patches is more than 0.6. 
    for probindice in probInds:
        p = patches[probindice]
        subMaskArea = np.sum(mask[ p['y1'] : p['y2']+1, p['x1'] : p['x2']+1])
        if subMaskArea / patchArea > thresh:
            continue
        mask[ p['y1'] : p['y2']+1, p['x1'] : p['x2']+1] = 1
        patInds[indCount] = probindice
        indCount = indCount + 1
    patInds = patInds[:indCount]
    patInds.sort()
        
    return patInds.astype('int')


def Gaussian_filter(size, std):
    
    size = (size-1)/2
    x, y = size
    x_lin = np.linspace(-x, x, int(2*x+1))
    y_lin = np.linspace(-y,y, int(2*y+1))
    xm, ym = np.meshgrid(x_lin, y_lin)
    
    arg = -(xm * xm + ym * ym) / (2 * std * std)
    h = np.exp(arg)
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = np.sum(h)
    if sumh != 0:
        h = h/sumh
    
    return h

def getRandForPdf(dist, n):
    # Generates random numbers following a pdf
    n = int(n)
    cumul = np.cumsum(dist)
    steps = np.arange(0,1,1/(len(dist)-1))
    cumulInv = np.zeros(len(cumul))
    cumulInd = 0
    
    for idx, step in enumerate(steps):
        if step < cumul[cumulInd]:
            cumulInv[idx] = cumulInd
        else:
            while (cumulInd < len(steps)) and (step > cumul[cumulInd] + np.finfo(dist.dtype).eps):
                cumulInd += 1
            cumulInv[idx] = cumulInd
    
    numbers = np.round(np.random.uniform(size=n) * (len(dist) - 1))#I think this +1 was for Matlab indexing
    numbers = numbers.astype('int')
    
    return cumulInv[numbers].astype('int')

def sub2ind(siz, v1, v2):
   
   return v1 + v2*siz[0]  


## possibly switch more to the Image object class.

class Image_object:
    
    def __init__(self, I):
        self.img = I
    
    ## This Distribution varies from the original matlab implementation. 
    ## It however doesn't vary that much from the imgaussfilt function
    ## which is now recommended instead of the rotationally symmetric filter
    ## used in the original paper (aldus Mathworks).
    def getProbDistribution(self, pSize):        
        h = Gaussian_filter(pSize, pSize[0]/3)
        IF = cv.filter2D(self.img, -1, h)
        dist = IF / sum(sum(IF))
        return dist

def getQuickDistribution(I, pSize, style='mat'):
    
    #print('I shape:', I.shape, pSize)
    start = time()
    if style=='mat':
        h = matlab_style_gauss2D(shape = pSize, sigma = min(pSize)/3)
    else:
        h = Gaussian_filter(pSize, pSize[0]/3)
        
    #print('time to generate kernel:', time()-start)
    start = time()
    
    #Origins need to be shifted according to whether pSize is even or odd
    if pSize[0]%2 == 0:
        origin = -1
    elif pSize[0]%2 != 0:
        origin = 0
    
    IF = cv.filter2D(I, origin, h, borderType=cv.BORDER_CONSTANT)
    
    #print('time to correlate ndimage:', time()-start)
    #I = np.round(I, 8)
    start=time()
    eps=10^-90
    dist = I/(sum(sum(IF))+eps)
    #print('time to normalize:', time()-start)
    return dist

def getProbDistribution(I, pSize):
    print('I shape:', I.shape, pSize)
    start = time()
    h = matlab_style_gauss2D(shape = pSize, sigma = min(pSize)/3)
    print('time to generate kernel:', time()-start)
    start = time()
    
    #Origins need to be shifted according to whether pSize is even or odd
    if pSize[0]%2 == 0:
        origin = -1
    elif pSize[0]%2 != 0:
        origin = 0
        
    I = scipy.ndimage.correlate(I, h, mode='constant', origin = origin)
    print('time to correlate ndimage:', time()-start)
    #I = np.round(I, 8)
    start=time()
    dist = I/sum(sum(I))
    print('time to normalize:', time()-start)
    return dist
## Taken from https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
## Thanks ali_m!
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h



    
     

    
### This Image_object class can hold all the subfunctions we are using such as
### Patch canonical size and what not. 

#   pSize = levPatSize
#   h = fspecial('gaussian', pSize, min(pSize)/3);
#   I = imfilter(IGS, h);
#   dist = I ./ sum(sum(I));
#   pDist = dist
    
    
    
def matopen(f):
    # quickly open the file and accompanying struct by only passing the relative filename (no '.mat')
    return loadmat('/Users/timalph/Documents/Paris/release/' + f + '.mat')[f]
    

def retrieve_randperms(data):
    data2 = data.split('========')
    data2 = data2[:10]
    
    randperm_d = {}
    
    for a in data2:
        key = a.split('inds')[0].split('  ')[-1]
        key = int(key)
        numberlist = []
        
        numbersplit = a.split('    ')[2:]
        for number in numbersplit:
            try:
                numberlist.append(int(number))
                
            except ValueError:
                try:
                    numberlist += list(map(int, number.split('  ')))
                except:
                    numberlist.append(int(number[:2]))
        randperm_d[key] = numberlist
    return randperm_d

def sumpatches(x):
    x1 = sum(np.array([entry['x1'] for entry in x.values()])+1)
    x2 = sum(np.array([entry['x2'] for entry in x.values()])+1)
    y1 = sum(np.array([entry['y1'] for entry in x.values()])+1)
    y2 = sum(np.array([entry['y2'] for entry in x.values()])+1)
    
    print(x1 + x2 + y1 + y2)
#fp, fm = tfd.test(mat_img, params, matPATH)