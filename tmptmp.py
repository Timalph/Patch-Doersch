# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:33:46 2020

@author: TimAl
"""

import os
import sys

import multiprocessing
import time
import argparse

import numpy as np
import csv

import pickle
import time

import Sample_Random_Patches as srp

import pandas as pd
from scipy.io import loadmat
import Create_Dataset
import Utils
import time
import Sample_Random_Patches as srp
import assignNN



##### YOU CAN TECHNICALLY JUST IMPLEMENT THIS, ONLY THING YOU NEED TO LOOK OUT IS THE HARDCODED DIRECTORIES
##### OUTPUT AS WELL AS INPUT FOR THE WORKERS. ARE YOU GOING TO SPLIT UP YOUR 10k DIRECTORIES EVEN FURTHER?
##### WOULD BE GOOD TO RUN IT FROM BASE /IMAGES_NEW/DIR_001/ and output to /CUT_IMAGES/DIR_001/ to make sure
##### YOU DON'T CLUTTER EVERYTHING

#partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
                

class Consumer(multiprocessing.Process):

    def __init__(self, task_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break

            next_task()
            self.task_queue.task_done()
        return


def sample(args, ind, ds, split):
    print('sample', ind)
    x = srp.main(ind, ds, 25, matlabinds=False, corr='quick')
    with open(os.path.join(args.outdir, 'patches', split, str(ind) +'.pickle'), 'wb') as h:
        pickle.dump(x, h)



        
class Task(object):
    def __init__(self, args, inds, ds, split):
        #print('Task init')
        self.inds = inds
        self.ds = ds
        self.args = args
        self.split = split
        
    def __call__(self):
        #print('Task call')
        sample(self.args, self.inds, self.ds, self.split)
        #seq_24(self.args, self.info)
        #print('Inside call', self.info)
        
    def __str__(self):
        return self.info['filename']


def main(args):
    
    matlab = False
    matlabinds = False


    out_dir = args.outdir

    ## SVM or scikit_SGD (linear svm)
    prediction_model = 'SVC'


    no_images = 10000
    cut = 1200

    ## if you have more images than you want to use for the algo
    possplit = 2000
    negsplit = 8000

    matlabinds=False
    corr = "quick"

    set_of_pos_nbs = ['Volewijck']
    set_of_neg_nbs = ['Apollobuurt', 'BijlmerCentrum', 'GrachtenGordel-West']
    topdir = 'Dataset10k'

    ## ========================================================================
    ## ========================================================================
    
    root_dir = os.path.join(args.path_to_dataset, topdir, 'cutouts/')

    dataframe = Create_Dataset.build_dataframe(set_of_pos_nbs, 
                                               set_of_neg_nbs, 
                                               topdir, 
                                               sampling = {'pos':possplit,
                                                           'neg':negsplit},
                                                           base_dir = args.path_to_dataset)


    Utils.setup_dir(out_dir)
    try:
        os.makedirs(os.path.join(out_dir, 'patches', 'pos'))
    except FileExistsError:
        pass

    try:
        os.makedirs(os.path.join(out_dir, 'patches', 'neg'))
    except FileExistsError:
        pass


    path_out = os.path.abspath(os.getcwd()) + '/' + out_dir
    ds = Create_Dataset.script_setup(out_dir, path_out, dataframe, root_dir)

    ## This seems like an elaborate way to just specify imgs is the dataframe
    #imgs = ds['imgs'][ds['conf']['currimset']]
    imgs = dataframe



    # Does parimgs mean: Paris images?
    parimgs = dataframe.loc[dataframe['city'] == ds['mycity']]
    otherimgs = dataframe.loc[dataframe['city'] != ds['mycity']]

    pos_ids = pd.Index.to_numpy(parimgs.index)
    neg_ids = pd.Index.to_numpy(otherimgs.index)

    ds['ispos'] = np.concatenate((np.ones(len(pos_ids)), np.zeros(len(neg_ids)))).astype('int')

    if not matlabinds:
        np.random.shuffle(pos_ids)
        np.random.shuffle(neg_ids)
        #raise ValueError("check possplit negsplit here, we should constrain the dataset in create dataset")
        ds['myiminds'] = np.concatenate((pos_ids, neg_ids))
    elif matlabinds:
        matPATH = '/Users/timalph/Documents/Paris/release/'
        workspace = loadmat(matPATH + 'matlab_random_shuffle')
        ds['myiminds'] = workspace['matinds'].squeeze() -1

    ds['parimgs'] = parimgs

    ###

    # sample_and_cluster(ds)

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
    amount = len(ds['sample']['initInds'])


    #=================================================
        # Establish communication queues
    tasks = multiprocessing.JoinableQueue()

    # Start consumers
    num_consumers = multiprocessing.cpu_count() * 2
    #num_consumers = 1
    print('Creating %d consumers' % num_consumers)
    consumers = [Consumer(tasks) for i in range(num_consumers)]
    for w in consumers:
        w.start()

    start = time.time()

    for ind in ds['sample']['initInds']:
        ## srp.main()
        tasks.put(Task(args, ind, ds, 'pos'))

    # Add a poison pill for each consumer
    for i in range(num_consumers):
        tasks.put(None)  



    # Wait for all of the tasks to finish
    tasks.join()
    
    end = time.time()

    print(end-start)
    print('saved {} patches'.format(len(ds['sample']['initInds'])))
    print("avg time per patch: {}".format((end-start)/len(ds['sample']['initInds'])))

    print("collecting patches...")


    for ind in ds['sample']['initInds']:
        tmp = pickle.load(open(os.path.join(out_dir, 'patches', str(ind) + '.pickle'), 'rb'))
        ds['sample']['patches'][ind] = tmp[0]
        ds['sample']['features'][ind] = tmp[1]

    print("avg time per patch after opening: {}".format((end-start)/len(ds['sample']['initInds'])))


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
    #workspace = loadmat(matPATH + 'negnums_for_python')
    initInds = np.where(np.logical_and(1-ds['ispos'], ds['isinit']))[0]
    myinds = np.random.permutation(initInds)[:30]
    #order = workspace['ord'].squeeze()-1
    #myinds = order[:min(len(order),30)]

    #if not matlabinds:
    #    raise ValueError("Mapreduce is continuing its count of dsidx I presume. I don't know what the implications of that are for this implementation.")
    print('Sampling negative patches')
    ds['sample']['initInds'] = myinds
    ## no_of_pos_images makes sure this script continues with dsidx where the 
    ## matlabscript starts again after sampling positive patches

    for ind in ds['sample']['initInds']:
        print(ind)
        if matlabinds:
            matlab_neg_inds = 'neg'
        else:
            matlab_neg_inds = False
        ds['sample']['patches'][ind], ds['sample']['features'][ind], _ = srp.main(ind, ds, matlabinds = matlab_neg_inds)


    with open(os.path.join(path_out, 'patches_tmpjuly.pickle'), 'wb') as handle:
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
    #directory = '/Users/timalph/Documents/Paris/release/pyth/out_10krun_May/ds/'

    #multi(ds['myiminds'], 'assignNN', ds, directory)

    with open(os.path.join(path_out, 'patches_tmpjuly.pickle'), 'wb') as handle:
        pickle.dump(ds, handle)





    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--outdir', type=str, default = 'bigparis_test_10k')
    parser.add_argument('--path_to_dataset', type=str, default = '/Users/timalph/Documents/Panoramas/processed')
    args = parser.parse_args()
    
    main(args)
