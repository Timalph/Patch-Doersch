import os
import pickle
import numpy as np
import pandas as pd
from scipy.io import loadmat
import Create_Dataset
import Utils
import time
import Sample_Random_Patches as srp
import assignNN
import assignNN_old
import argparse
import math

import mat_funcs
import cv2 as cv
import Visualizing
import train


def main(args):
    ds = pickle.load(open(os.path.join(args.outdir, 'finpickle.pickle'), 'rb'))
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
        print("detsimple: {}/{}".format(i+1, len(ds['myiminds'])))
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
                counts[i, 0] = sum(posCounts[0][i, :])
                counts[i, 1] = sum(negCounts[i, :])
                
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
        with open(ds['raw_outdir'] + '/{}_output.html'.format(disptype), 'w') as f:
            f.write(ds['bestbin']['bbhtml'])
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--outdir', type=str, default = '')
    
    args = parser.parse_args()

    main(args)
