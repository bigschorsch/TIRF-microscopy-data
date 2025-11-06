import random
import numpy as np
from scipy import stats
from scipy import signal
from scipy.stats import norm
from scipy.optimize import curve_fit

from tqdm.notebook import tqdm
from tqdm.contrib import tzip

def get_pdf(data, lim=None):
    '''
    Calculates the propability density function of given data up to x=lim
    Parameters:
        data
        lim: integer: limit on x-axis for pdf calculation. If not given maximum of data will be taken
    Returns
        x,y: x and y values of calculated pdf
    '''
    
    if lim==None:
        lim = int(max(data))
    
    x = np.linspace(0, lim, lim*100)
    y = stats.gaussian_kde(data).pdf(x)
    return x,y


def convolve_pdf(pdf, lim_con):
    '''
    Calculates the convolutions of the given pdf up to the highest oligomeric state set
    Parameters:
        pdf
        lim_con: highest oligomeric state
    Returns:
        pdf_list: List containing all calculated pdfs
    '''
    pdf_list = [pdf]
    for i in range(lim_con-1):
        y = signal.convolve(pdf_list[i], pdf_list[0], mode='full', method='auto')/sum(pdf_list[i])
        pdf_list.append(y)
    return pdf_list


def unify(pdf_list, size):
    '''
    Ensures, that all elements in pdf_list have the same size, so curve_fit can be applied
    Parameter:
        pdf_list: contains the pdfs
        size: required size
    Returns:
        pdf_list: unified pdf_list
    '''
    for i in range(len(pdf_list)):
        data = list(pdf_list[i])
        diff = size - len(data)
        if diff > 0:
            filler = np.zeros(diff)
            data.extend(filler)
        elif diff <0:
            diff = abs(diff)
            del data[-diff:]
        pdf_list[i] = data
    return pdf_list



def func(pdf_list, alpha1, alpha2, alpha3=0, alpha4=0, alpha5=0):
    '''
    Function used for fitting
    Parameters:
        pdf_list: contains the pdfs
        alpha: parameters which will be adjusted buy curve fit
    Returns:
        ydata: 
    '''
    alpha = [alpha1,alpha2,alpha3,alpha4,alpha5]
    
    l = len(pdf_list)
    
    ydata = pdf_list[0]*alpha1
    sum_alpha = alpha1
    
    for i in range(1,(l-2)):
        ydata += (pdf_list[i] * alpha[i])
        sum_alpha += alpha[i]
        
    ydata += (pdf_list[l-2]*(1 - sum_alpha))
    return ydata



def func_12(pdf_list, alpha1, alpha2):
    '''
    Function used for fitting if the highest oligomeric state is 3
    Parameters:
        pdf_list: contains the pdfs
        alpha: parameters which will be adjusted buy curve fit
    Returns:
        ydata: 
    '''
    ydata= (pdf_list[0] *alpha1)+(pdf_list[1]*alpha2)+(pdf_list[2]*(1-(alpha1+alpha2)))
    return ydata


def bootstrapping(data, pdf_list, lim, size, percent = 50, iterations = 100):
    
    data_len = int(len(data) *(percent/100))
    result = []
    
    for i in tqdm(range(iterations), desc = 'Iterations'):
        boot_data=[]
        
        for j in range(data_len):
            boot_data.append(data[random.randint(0,(len(data)-1))])
    
        xdata, ydata = get_pdf(boot_data, lim=lim)
        ydata = [ydata]
        ydata = unify(ydata, size)
        popt, pcov = curve_fit(func_12, pdf_list, ydata[0])
        result.append([popt[0], popt[1],(1-(sum(popt)))])
    return result