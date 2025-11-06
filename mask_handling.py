import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims

from sdt import roi#, io,  motion, image, nbui

import cv2

#from tqdm.notebook import tnrange, tqdm
#from tqdm.contrib import tzip


###### Functions to create masks:
def make_masks(pattern_file, SPE_file, cell, k_on=3, k_off=5):
    '''
    Creates ROI objects for masks.
    Parameters:
        pattern_file: filename of tif file
        SPE_files: SPE filename of pattern image
        cell: cell specification for figure title
    Returns:
        p_img: pattern image
        mask: mask without eroding or dilating
        mask_ON: dilated mask
        mask_OFF: eroded mask
    '''
    with pims.open(pattern_file) as img_seq:
        pattern_img = img_seq[0]

    with pims.open(SPE_file) as img_seq:
        real_img = img_seq[0]

    img = cv2.imread(pattern_file,0)
    kernel_ON = np.ones((k_on,k_on),np.uint8)
    kernel_OFF = np.ones((k_off,k_off),np.uint8)
    mask_ON_temp = cv2.erode(img,kernel_ON,iterations = 1)
    mask_OFF_temp = cv2.dilate(img,kernel_OFF,iterations = 1)

    fig, ax = plt.subplots(1,4, figsize=(25,6))
    fig.suptitle(cell, weight='bold', fontsize=20)

    ax[0].imshow(real_img, cmap='gray')
    ax[0].set_title('image', fontsize=16)
    ax[1].imshow(pattern_img, cmap='gray')
    ax[1].set_title('pattern', fontsize=16)
    ax[2].imshow(mask_ON_temp, cmap='gray')
    ax[2].set_title('ON', fontsize=16)
    ax[3].imshow(mask_OFF_temp, cmap='gray')
    ax[3].set_title('OFF', fontsize=16)

    p_img = real_img
    mask = roi.MaskROI(pattern_img)
    mask_ON = roi.MaskROI(mask_ON_temp)
    mask_OFF = roi.MaskROI(mask_OFF_temp)
    
    return p_img, mask, mask_ON, mask_OFF, fig


def make_masks_all(pattern_file, SPE_file, cell, k_on=3, k_off=5, excluded_pillars=None):
    '''
    Creates ROI objects for masks.
    Parameters:
        pattern_file: filename of tif file
        SPE_files: SPE filename of pattern image
        cell: cell specification for figure title
        k_on, k_off: integers to define kernel for erosion and dilation of original mask
        excluded_pillars: dictionary specifying ON areas to be excluded for a specific cell ({'c1': ['p1', 'p4'], 'c3': ['p1']})
    Returns:
        p_img: pattern image
        mask: mask without eroding or dilating
        mask_ON: dilated mask
        mask_OFF: eroded mask
        centers: center coordinates of individual ON areas
        radii: radii of individual ON areas
        pattern_ROI: masks for individual ON areas stored in a dictionary
        pattern_ROI_inv: inverted masks for individual ON areas stored in a dictionary
        fig: figure
    '''
    fig, ax = plt.subplots(1,5, figsize=(30,6))
    fig.suptitle(cell, weight='bold', fontsize=20)

    with pims.open(pattern_file) as img_seq:
        pattern_img = img_seq[0]

    with pims.open(SPE_file) as img_seq:
        real_img = img_seq[0]

    img = cv2.imread(pattern_file,0)
    kernel_ON = np.ones((k_on,k_on),np.uint8)
    kernel_OFF = np.ones((k_off,k_off),np.uint8)
    mask_ON_temp = cv2.erode(img,kernel_ON,iterations = 1)
    mask_OFF_temp = cv2.dilate(img,kernel_OFF,iterations = 1)
    

    p_img = real_img
    mask = roi.MaskROI(pattern_img)
    mask_ON = roi.MaskROI(mask_ON_temp)
    mask_OFF = roi.MaskROI(mask_OFF_temp)
    
    
    
    ################# separated pillars
    imsize=np.shape(pattern_img)[0]
    
    #### Create binary image of mask in ON area for plot
    mask = mask_ON
    masked = mask(mask(pattern_img, fill_value=0), invert=True, fill_value=1)
    
    #### Use ON area for finding circles (due to better circularity when using high k_off for masks) - add k_on+k_off to radii (3+7)
    mask_on = mask_ON
    #masked_on = mask_on(mask_on(pattern, fill_value=0), invert=True, fill_value=1)
    
    centers, radii = get_circles(mask_on, pattern_img)
    
    radii = [r+5 for r in radii]
    
    #centers_cells[cell] = centers
    #radii_cells[cell] = radii
    
    #### create ROI object for each pillar
    pattern_ROI = {}
    pattern_ROI_inv = {}

    for i in range(len(centers)):
        pattern_ROI['p{}'.format(i+1)] = roi.EllipseROI(centers[i], (radii[i], radii[i]))
        
        r = max(radii)
        x = np.linspace(-r,r,200)

        p = circle_f(x,r,*centers[i])
        p_add = [[0,imsize], [imsize,imsize], [imsize,0], [0,0], [0,imsize]]

        temp = np.append(p_add, p)
        temp = np.append(temp, p[0])
        temp = np.split(temp, len(temp)/2)
        pattern_ROI_inv['p{}'.format(i+1)] = roi.PathROI(temp)
        
    #### assign
    #mask_sep[cell] = pattern_ROI
    #mask_sep_inv[cell] = pattern_ROI_inv
    
    ax[4].set_title('separated ON areas', weight='bold')
    ax[4].imshow(masked, cmap='gray')
    for i in range(len(centers)):
        p = centers[i]
        x = p[0]
        y = p[1]

        text = 'p{}'.format(i+1)

        ax[4].annotate(text, (x, y), ha='center', va='center', weight='bold', fontsize=15)
    
    if excluded_pillars != None and any(excluded_pillars) == True:
        for c, pl in excluded_pillars.items():
            if cell == c:
                cur = mask_on(p_img, fill_value=0)
                if type(pl) != list:
                    raise ValueError('ON areas to be excluded have to be given in a list!')
                for p in pl:
                    cur = pattern_ROI_inv[p](cur, fill_value=0)

                mask_on = roi.MaskROI(cur)
                pattern_ROI = {k:v for k,v in pattern_ROI.items() if k not in pl}
                pattern_ROI_inv = {k:v for k,v in pattern_ROI_inv.items() if k not in pl}
        
    
    ax[0].imshow(real_img, cmap='gray')
    ax[0].set_title('image', weight='bold', fontsize=16)
    ax[1].imshow(pattern_img, cmap='gray')
    ax[1].set_title('pattern', weight='bold', fontsize=16)
    ax[2].imshow(mask_on(mask_ON_temp, fill_value=0), cmap='gray')
    ax[2].set_title('ON', weight='bold', fontsize=16)
    ax[3].imshow(mask_OFF_temp, cmap='gray')
    ax[3].set_title('OFF', weight='bold', fontsize=16)
    
    
    return p_img, mask, mask_ON, mask_OFF, centers, radii, pattern_ROI, pattern_ROI_inv, fig

################## separating and grouping pillars
def get_circles(mask, pattern):
    '''
    Returns center coordinates and radii of circles in an image (-> idividual pillars of a pattern)
    Parameters:
        mask: mask applied to pattern
        pattern: image of pattern
    Returns:
        centers: center coordinates 
        radii: radii
    '''
    #### Create binary image of mask in OFF area
    masked = mask(mask(pattern, fill_value=0), invert=True, fill_value=1)
    
    #### find circles
    image = masked.astype(np.uint8)

    contours, hierarchy = cv2.findContours(
        image,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )


    centers = []
    radii = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # there is one contour that contains all others, filter it out
        if area > 500:
            continue

        br = cv2.boundingRect(contour)
        radii.append(br[2])

        m = cv2.moments(contour)
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        centers.append(center)

    #radius = int(np.average(radii)) + 5
    
    return centers, radii

def circle_f(x,r,x0,y0):
    '''
    calculates f(x) of a circle with radius r, center coordinates x0 and y0 for given x values
    '''
    y_t = np.sqrt(r**2-x**2)
    
    y_all = list(y_t) + list(y_t[::-1][1:]*-1)
    x_all = list(x) + list(x[::-1][1:])
    
    points = [(x+x0, y+y0) for x,y in zip(x_all, y_all)]
    
    return points