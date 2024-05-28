#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:32:21 2021

@author: timalph
"""

import numpy as np



### Tested!
def computePascalOverlap(b1, b2):
    
    lb1 = len(b1)
    lb2 = len(b2)
    
    overlaps = np.zeros((lb1,lb2))
    
    
    x11, y11, x12, y12, _ = np.hsplit(b1, 5)
    areab1 = (x12-x11+1) * (y12-y11+1)
    
    x21, y21, x22, y22, _ = np.hsplit(b2, 5)
    areab2 = (x22-x21+1) * (y22-y21+1)
    
    for i in range(lb1):
        for j in range(lb2):
            xx1 = max(x11[i], x21[j])
            yy1 = max(y11[i], y21[j])
            xx2 = min(x12[i], x22[j])
            yy2 = min(y12[i], y22[j])
            
            w = xx2-xx1+1
            h = yy2-yy1+1
            
            if w > 0 and h > 0:
                overlaps[i,j] = w * h / (areab1[i] + areab2[j] - (w * h))
    
    return overlaps

def computePedroOverlap(b1, b2):
    lb1 = len(b1)
    lb2 = len(b2)
    
    overlaps = np.zeros((lb1,lb2))
    
    
    x1, y1, x2, y2, _ = np.hsplit(b1, 5)
    area = (x2-x1+1) * (y2-y1+1)
    
    for i in range(lb1):
        for j in range(lb2):
            x21 = b2[j,0]
            y21 = b2[j,1]
            x22 = b2[j,2]
            y22 = b2[j,3]
            
            xx1 = max(x1[i], x21)
            yy1 = max(y1[i], y21)
            xx2 = min(x2[i], x22)
            yy2 = min(y2[i], y22)
            
            w = xx2-xx1+1
            h = yy2-yy1+1
            
            if w > 0 and h > 0:
                overlaps[i,j] = w * h /area[i]
    return overlaps

