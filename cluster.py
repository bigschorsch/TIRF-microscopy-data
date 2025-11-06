import sys

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


from sdt import roi, image, nbui
#from micro_helpers import pims
import pims
import math
from helpers import *


def remove_cluster(mask, start):
    '''
    Parameters:
        mask: binary matrix originating from adaptive thresh
        start: coordinates of the starting point
    Returns:
        mask: binary matrix with one cluster removed
        cluster: coordinates of pixels in the removed cluster
        size: size of the removed cluster
    '''
    max_x, max_y = np.shape(mask)
    cluster = [[start]]
    step = []
    size = 1
    a = 0 
    con = True
    
    #maps out the cluster
    while con:
        for b in range(len(cluster[a])):
            i,j = cluster[a][b]
            mask[i][j] = False
            
            for n in range((i-1), (i+2)):
                if (n<0) or(n== max_y): break
                
                for m in range((j-1), (j+2)):
                    if (m<0) or(m == max_y): break
                    
                    if mask[n][m]:
                        step.append([n,m])
                        mask[n][m] = False
                        
        if len(step) == 0: break
        
        size = size + len(step)
        cluster.append(step)
        step = []
        
        a += 1
        if a > 200:
            con = False

    return mask, cluster, size




def add_cluster(mask, cluster):
    '''
    Parameter:
        mask: Binary matrix
        cluster: Big cluster that should be added
    Returns:
        mask: Binary matrix with added cluster
    '''
    
    for a in range(len(cluster)):
        for b in range(len(cluster[a])):
            i,j = cluster[a][b]
            mask[i][j] = True
            
    return mask



def cluster(img, block=15, c=0, smoothf=8, thresh=200):
    '''
    Parameter:
        img: image to be corrected
        block: size of area used for thresholding
        c: constant subtracted from mean
        smoothf: for Gaussian smoothing
        thresh: specifies size of clusters to be removed (number of pixels)
    Returns:
        mask: Binary matrix with only big clusters left
    '''
    #applying adaptive thresholding to find high density areas
    mask = image.adaptive_thresh(img, block, c, smooth=smoothf)
    mask = ndimage.binary_closing(mask, iterations=2)
    max_x, max_y = np.shape(mask)

    
    liste = []
    gesamtliste = []
    size = []
    
    #finds starting point in high density area and uses remove_cluster to map out cluster
    for i in range(max_x):
        for j in range(max_y):
            if mask[i][j]:
                start = [i,j]
                mask, liste, s = remove_cluster(mask, start)
                gesamtliste.append(liste)
                size.append(s)
    
    #big clusters are re-added to mask
    for i in range(len(size)): 
        if size[i] > thresh:
            mask = add_cluster(mask, gesamtliste[i])
            
    return mask






