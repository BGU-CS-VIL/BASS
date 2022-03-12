#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:45:03 2018

@author: uzielr
"""


"""
************************************************************************************************************************************************************

                                                                            Imports

************************************************************************************************************************************************************
"""
import Global
from BASS import *
import torch
import os
import sys

import numpy as np


"""
************************************************************************************************************************************************************

                                                                        Help Funtions

************************************************************************************************************************************************************
"""




def blockshaped(arr, nrows, ncols):


    """Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.

    **Parameters**:
     - arr - 2d matrix to split

     - nrows - Size of row after the split

     - ncols - Size of col after the split

    **Returns**:
     - arr - Array after the split

    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w):

    """Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    Use after :meth:`AngleImpl.blockshaped`.
    Create_DataMatrix
    **Parameters**:
     - arr - 2d matrix after split

     - nrows - Size of row before the split

     - ncols - Size of col before the split

    **Returns**:
     - arr - Array before the split

    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))



def softmax(x,axis):

    """Compute softmax values for each sets of scores in x.
    **Parameters**:
      - X - array of point.
      - axis - axis to sum over

    **Returns**:
      - arr -  Exp of the data after softmax

    """
    e_x = np.exp(x - np.max(x,axis=0)[np.newaxis])
    return e_x / e_x.sum(axis)[np.newaxis]


def softmaxTF(x, axis,sum):
    """Compute softmax values for each sets of scores in x.
    **Parameters**:
      - X - array of point.
      - axis - axis to sum over

    **Returns**:
      - arr -  Exp of the data after softmax

    """
    sum.zero_()
    xmax=x.max(dim=axis)[0].unsqueeze(axis)
    x.sub_(xmax)
    x.exp_()
    if(Global.neig_num==5):
        x.div_(sum.add_(x[:,0]).add_(x[:,1]).add_(x[:,2]).add_(x[:,3]).add_(x[:,4]).unsqueeze(axis))
    if(Global.neig_num==4):
        x.div_(sum.add_(x[:, 0]).add_(x[:, 1]).add_(x[:, 2]).add_(x[:, 3]).unsqueeze(axis))


def Create_DataMatrix(figure):
    x,y=np.mgrid[:figure.shape[0],:figure.shape[1]]
    if Global.TIF:
        data = np.array((x.ravel(),y.ravel()))
        for c in range(Global.TIF_C):
            data = np.append(data,np.expand_dims(figure[:,:,c].ravel(),0), axis = 0)
    else:
        L=figure[:,:,0].ravel()
        A=figure[:,:,1].ravel()
        B=figure[:,:,2].ravel()

        data =np.array((x.ravel(),y.ravel(),L,A,B))
    return data.T
