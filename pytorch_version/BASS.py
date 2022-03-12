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
import argparse
import cProfile
import torch
from scipy.spatial import cKDTree
import gc
import os
import Global
import Help_Functions as my_help
import Conn_Functions as my_connectivity
from torch.autograd import Variable
import numpy as np
import time
from PIL import Image
import glob
import sys
import csv
from numpy.linalg import inv
from numpy.linalg import det
import warnings
from copy import deepcopy



"""
************************************************************************************************************************************************************

                                                                            Start

************************************************************************************************************************************************************
"""



def Merge(X,argmax,Nk,it_merge,temp_b,m_v_father_b,m_v_sons_b,b_father_b,b_sons_b):
        """Merge step

        **Parameters**:
         - :math:`X[N,D]` - Data matrix  [Number of pixels, Dimenstion of the data].

         - :math:`argmax[N,2]' Pixel to SP matrix

        **Returns**:
         - Update argmax, and split_lvl
        """
        padded_matrix = Global.Padding0(argmax[:,0].reshape(Global.HEIGHT,-1)).reshape(-1).to(Global.device)
        pair = torch.zeros(Global.K_C + 1).int().to(Global.device)
        left = torch.zeros(Global.K_C + 1,2).int().to(Global.device)
        left[:,0]=torch.arange(0,Global.K_C+1)
        left[:,0]=Global.N_index[0:Global.K_C+1]

        it_merge = 0
        if(it_merge%4==0):
            ind_left = torch.masked_select(Global.inside_padded, (
                        (padded_matrix[Global.inside_padded] != padded_matrix[Global.inside_padded - 1]) & (
                            padded_matrix[Global.inside_padded - 1] > 0)))
            left[padded_matrix[ind_left],1] = padded_matrix[ind_left - 1].int()
        if (it_merge%4 == 1):
            ind_left = torch.masked_select(Global.inside_padded, (
                    (padded_matrix[Global.inside_padded] != padded_matrix[Global.inside_padded + 1]) &(
                    padded_matrix[Global.inside_padded + 1] > 0)))
            left[padded_matrix[ind_left], 1] = padded_matrix[ind_left + 1].int()
        if (it_merge%4 == 2):
            ind_left = torch.masked_select(Global.inside_padded, (
                    (padded_matrix[Global.inside_padded] != padded_matrix[Global.inside_padded - (Global.WIDTH+2)]) & (
                    padded_matrix[Global.inside_padded - (Global.WIDTH+2)] > 0)))
            left[padded_matrix[ind_left], 1] = padded_matrix[ind_left - (Global.WIDTH+2)].int()
        if (it_merge%4 == 3):
            ind_left = torch.masked_select(Global.inside_padded, (
                    (padded_matrix[Global.inside_padded] != padded_matrix[Global.inside_padded + (Global.WIDTH+2)]) & (
                    padded_matrix[Global.inside_padded + (Global.WIDTH+2)] > 0)))
            left[padded_matrix[ind_left], 1] = padded_matrix[ind_left + (Global.WIDTH+2)].int()

        it_merge=it_merge+1


        for i in range(0, Global.K_C + 1):
            val = left[i, 1]
            if ((val > 0 )and (val!=i)):
                if ((pair[i] == 0) and (pair[val] == 0)):
                    if (val < i):
                        pair[i] = val
                        pair[val] = val
                    else:
                        pair[val] = i
                        pair[i] = i

        left[:,1]=pair

        Nk.zero_()
        Nk.index_add_(0, argmax[:, 0], Global.ones)
        Nk = Nk + 0.0000000001

        Nk_merged=torch.add(Nk,Nk[left[:,1].long()])
        alpha=torch.Tensor([float(1000000)]).to(Global.device)
        beta=torch.Tensor([Global.int_scale*alpha+Global.int_scale]).to(Global.device)
        v_father = Nk
        v_merged = Nk_merged


        m_v_father = m_v_father_b[0:Nk.shape[0]].zero_()
        b_father = b_father_b[0:Nk.shape[0]].zero_()

        m_v_father.index_add_(0, argmax[:, 0], X[:, 2:])
        m_v_merged = torch.add(m_v_father, m_v_father[left[:, 1].long()])


        m_merged = torch.div(m_v_merged, v_merged.unsqueeze(1))
        m_father = torch.div(m_v_father, v_father.unsqueeze(1))
        a_father = torch.add(Nk / 2, alpha).unsqueeze(1)
        a_merged = torch.add(Nk_merged / 2, alpha).unsqueeze(1)
        b_father.index_add_(0, argmax[:, 0], torch.pow(X[:, 2:], 2))
        b_merged=torch.add(b_father,b_father[left[:,1].long()])
        b_father=b_father/2
        b_merged=b_merged/2
        b_father.add_(torch.add(beta, -torch.mul(torch.pow(m_father, 2), v_father.unsqueeze(1)) / 2))
        b_merged.add_(torch.add(beta, -torch.mul(torch.pow(m_merged, 2), v_merged.unsqueeze(1)) / 2))


        gamma_2_merged = torch.mvlgamma(a_merged,1)
        gamma_2_father = torch.mvlgamma(a_father, 1)


        ll_2_merged=0.5*torch.log(v_merged).unsqueeze(1)+ \
                    (a_merged*torch.log(b_merged))+\
                    gamma_2_merged- \
                    ((torch.mul(torch.log(Global.PI),Nk_merged/2))+(0.301*Nk_merged)).unsqueeze(1)


        ll_2_father=0.5*torch.log(v_father).unsqueeze(1)+ \
                    (a_father*torch.log(b_father))+\
                    gamma_2_father- \
                    ((torch.mul(torch.log(Global.PI),Nk/2))+(0.301*Nk)).unsqueeze(1)


        ll_2_father = torch.sum(ll_2_father, 1)[0:ll_2_father.shape[0]]
        ll_2_merged = torch.sum(ll_2_merged, 1)[0:ll_2_merged.shape[0]]



        ll_merged_min = torch.min(ll_2_merged[1:ll_2_merged.shape[0]].masked_select(
            ~(torch.isnan(ll_2_merged[1:ll_2_merged.shape[0]]) ^ torch.isinf(ll_2_merged[1:ll_2_merged.shape[0]]))))
        ll_merged_max = torch.max(ll_2_merged[1:ll_2_merged.shape[0]].masked_select(
            ~(torch.isnan(ll_2_merged[1:ll_2_merged.shape[0]]) ^ torch.isinf(ll_2_merged[1:ll_2_merged.shape[0]]))))
        ll_father_min = torch.min(
            ll_2_father.masked_select(~(torch.isnan(ll_2_father) ^ torch.isinf(ll_2_father))))
        ll_father_max = torch.max(
            ll_2_father.masked_select(~(torch.isnan(ll_2_father) ^ torch.isinf(ll_2_father))))

        ll_merged_min=torch.min(ll_merged_min,ll_father_min)
        ll_merged_max=torch.max(ll_merged_max,ll_father_max)


        ll_2_merged = torch.div(torch.add(ll_2_merged, -ll_merged_min), (ll_merged_max - ll_merged_min)) * (-10000) + 0.1
        ll_2_father = torch.div(torch.add(ll_2_father, -ll_merged_min), (ll_merged_max - ll_merged_min))*(-10000) + 0.1

        gamma_alpha_2=torch.mvlgamma(torch.Tensor([Global.ALPHA_MS2/2]).to(Global.device),1)
        gamma_alpha=torch.mvlgamma(torch.Tensor([Global.ALPHA_MS2]).to(Global.device),1)

        gamma_father=torch.mvlgamma(Nk,1)
        gamma_add_father=torch.mvlgamma(Nk_merged,1)
        gamma_alpha_father=torch.mvlgamma(Nk+Global.ALPHA_MS2/2,1)
        gamma_add_alpha_merged = torch.mvlgamma(Nk_merged + Global.ALPHA_MS2, 1)

        prob = -Global.LOG_ALPHA_MS2+gamma_alpha-2*gamma_alpha_2 +\
               gamma_add_father-gamma_add_alpha_merged+ \
               gamma_alpha_father[left[:, 0].long()]-gamma_father[left[:, 0].long()]+ \
               gamma_alpha_father[left[:, 1].long()] - gamma_father[left[:, 1].long()] - 2 + \
               ll_2_merged[left[:, 0].long()] - ll_2_father[left[:, 0].long()] - ll_2_father[[left[:, 1].long()]]

        prob=torch.where(((left[:,0]==left[:,1])+(left[:,1]==0))>0,-torch.Tensor([float("inf")]).to(Global.device),prob)

        idx_rand=torch.where(torch.exp(prob) > 1.0, Global.N_index[0:prob.shape[0]].long(),Global.zeros[0:prob.shape[0]].long()).nonzero()[:, 0]

        pair[left[:,1].long()]=left[left[:,0].long()][:,0]

        left[:,1]=Global.N_index[0:Global.K_C+1]
        left[idx_rand.long(),1]=pair[idx_rand.long()]

        argmax[:,0] = left[argmax[:,0],1]


        if(Global.Print):
            print("Idx Merge Size: ",idx_rand.shape[0])

        Global.split_lvl[idx_rand]= Global.split_lvl[idx_rand]*2
        Global.split_lvl[left[idx_rand,1].long()]=Global.split_lvl[idx_rand]

def Split(X,XXT,argmax,Nk,sons_LL_b,X_sons_b,X_father_b,father_LL_b,C1,c1_temp,clusters_LR,it_split,m_v_sons_b,m_v_father_b,b_sons_b,b_father_b,SigmaXY_b,SigmaXY_i_b,SIGMAxylab_b,Nk_b,X1_b,X2_00_b,X2_01_b,X2_11_b):
    it_split=it_split+1
    K_C_Split=torch.max(argmax[:,1])+1
    if(Nk.shape[0]>K_C_Split):
        K_C_Split=Nk.shape[0]
    Nk_s = torch.zeros(K_C_Split).float().to(Global.device)
    Nk.zero_()
    a_prior_sons = Nk_s
    Global.psi_prior_sons = torch.mul(torch.pow(a_prior_sons, 2).unsqueeze(1), torch.eye(2).reshape(-1, 4).to(Global.device))
    Global.ni_prior_sons = (Global.C_prior * a_prior_sons) - 3
    Nk.index_add_(0, argmax[:, 0], Global.ones)
    Nk = Nk + 0.0000000001
    Nk_s.index_add_(0, argmax[:, 1], Global.ones)
    Nk_s = Nk_s + 0.0000000001



    sons_LL=sons_LL_b[0:Nk_s.shape[0]].zero_()
    X_sons=X_sons_b[0:Nk_s.shape[0]].zero_()
    X_father=X_father_b[0:Global.K_C+1].zero_()
    father_LL=father_LL_b[0:Global.K_C+1].zero_()


    X_sons.index_add_(0,argmax[:,1],X[:,0:2])
    X_father.index_add_(0,argmax[:,0],X[:,0:2])
    sons_LL[:,0]= -torch.pow(X_sons[:,0],2)
    sons_LL[:,1]= -torch.mul(X_sons[:,0],X_sons[:,1])
    sons_LL[:,2]= -sons_LL[:,1]
    sons_LL[:,3]= -torch.pow(X_sons[:,1],2)

    father_LL[:, 0] = -torch.pow(X_father[:,0], 2)
    father_LL[:, 1] = -torch.mul(X_father[:,0], X_father[:,1])
    father_LL[:, 2] = -father_LL[:, 1]
    father_LL[:, 3] = -torch.pow(X_father[:,1], 2)


    sons_LL.index_add_(0, argmax[:,1], XXT)
    father_LL.index_add_(0,argmax[:,0],XXT)

    ni_sons=torch.add(Global.ni_prior_sons,Nk_s)[0:sons_LL.shape[0]]
    ni_father=torch.add(Global.ni_prior,Nk)[0:father_LL.shape[0]]
    psi_sons=torch.add(sons_LL.reshape(-1,4),Global.psi_prior_sons)[0:ni_sons.shape[0]]
    psi_father=torch.add(father_LL.reshape(-1,4),Global.psi_prior)[0:ni_father.shape[0]]
    ni_sons[(ni_sons <= 1).nonzero()] = 2.00000001
    ni_father[(ni_father <= 1).nonzero()] = 2.00000001

    gamma_sons=torch.mvlgamma((ni_sons/2),2)
    gamma_father=torch.mvlgamma((ni_father/2),2)
    det_psi_sons=0.00000001+torch.add(torch.mul(psi_sons[:, 0], psi_sons[:, 3]),-torch.mul(psi_sons[:, 1], psi_sons[:, 2]))
    det_psi_father=0.00000001+torch.add(torch.mul(psi_father[:, 0], psi_father[:, 3]),-torch.mul(psi_father[:, 1], psi_father[:, 2]))
    det_psi_sons[(det_psi_sons <= 0).nonzero()] = 0.00000001
    det_psi_father[(det_psi_father <= 0).nonzero()] = 0.00000001

    det_psi_prior_sons=0.00000001+torch.add(torch.mul(Global.psi_prior_sons[:, 0], Global.psi_prior_sons[:, 3]),-torch.mul(Global.psi_prior_sons[:, 1], Global.psi_prior_sons[:, 2]))
    det_psi_prior_father=0.00000001+torch.add(torch.mul(Global.psi_prior[:, 0], Global.psi_prior[:, 3]),-torch.mul(Global.psi_prior[:, 1], Global.psi_prior[:, 2]))
    det_psi_prior_sons[(det_psi_prior_sons <= 0).nonzero()] = 0.00000001
    det_psi_prior_father[(det_psi_prior_father <= 0).nonzero()] = 0.00000001

    Global.ni_prior_sons[(Global.ni_prior_sons <= 1).nonzero()] = 2.00000001
    Global.ni_prior[(Global.ni_prior <= 1).nonzero()] = 2.00000001
    gamma_prior_sons=torch.mvlgamma((Global.ni_prior_sons / 2),2)
    gamma_prior_father=torch.mvlgamma((Global.ni_prior / 2),2)

    ll_sons= -(torch.mul(torch.log((Global.PI)),(Nk_s)))+ \
             torch.add(gamma_sons,-gamma_prior_sons) + \
             torch.mul(torch.log(det_psi_prior_sons), (Global.ni_prior_sons * 0.5)) - \
             torch.mul(torch.log(det_psi_sons),(ni_sons * 0.5))+\
             torch.log(Nk_s[0:sons_LL.shape[0]])

    ll_father= -(torch.mul(torch.log((Global.PI)),(Nk)))+ \
               torch.add(gamma_father,-gamma_prior_father) + \
               torch.mul(torch.log((det_psi_father)), Global.ni_prior * 0.5) - \
               torch.mul(torch.log(det_psi_father),ni_father * 0.5) +\
               torch.log(Nk[0:father_LL.shape[0]])

    ll_sons_min=torch.min(ll_sons[1:ll_sons.shape[0]].masked_select(~(torch.isinf(ll_sons[1:ll_sons.shape[0]])^torch.isnan(ll_sons[1:ll_sons.shape[0]]))))
    ll_sons_max=torch.max(ll_sons[1:ll_sons.shape[0]].masked_select(~(torch.isinf(ll_sons[1:ll_sons.shape[0]])^torch.isnan(ll_sons[1:ll_sons.shape[0]]))))
    ll_father_min=torch.min(ll_father[1:ll_father.shape[0]].masked_select(~(torch.isinf(ll_father[1:ll_father.shape[0]])^torch.isnan(ll_father[1:ll_father.shape[0]]))))
    ll_father_max=torch.max(ll_father[1:ll_father.shape[0]].masked_select(~(torch.isinf(ll_father[1:ll_father.shape[0]])^torch.isnan(ll_father[1:ll_father.shape[0]]))))

    ll_sons_min=torch.min(ll_sons_min,ll_father_min)
    ll_sons_max=torch.max(ll_sons_max,ll_father_max)


    ll_sons=torch.div(torch.add(ll_sons,-ll_sons_min),(ll_sons_max-ll_sons_min))*(-1000)+0.1
    ll_father=torch.div(torch.add(ll_father,-ll_sons_min),(ll_sons_max-ll_sons_min))*(-1000)+0.1




    alpha=torch.Tensor([float(1000000)]).to(Global.device)
    beta=torch.Tensor([Global.int_scale*alpha+Global.int_scale]).to(Global.device)
    Nk.zero_()
    Nk_s.zero_()
    Nk.index_add_(0, argmax[:, 0], Global.ones)
    Nk = Nk + 0.0000000001
    Nk_s.index_add_(0, argmax[:, 1], Global.ones)
    Nk_s = Nk_s + 0.0000000001
    v_father=Nk
    v_sons=Nk_s




    m_v_sons=m_v_sons_b[0:Nk_s.shape[0]].zero_()
    m_v_father=m_v_father_b[0:Nk.shape[0]].zero_()
    b_sons = b_sons_b[0:Nk_s.shape[0]].zero_()
    b_father = b_father_b[0:Nk.shape[0]].zero_()
    m_v_sons.index_add_(0, argmax[:, 1], X[:,2:])
    m_v_father.index_add_(0, argmax[:, 0], X[:,2:])
    m_sons=torch.div(m_v_sons,v_sons.unsqueeze(1))
    m_father=torch.div(m_v_father,v_father.unsqueeze(1))
    a_sons=torch.add(Nk_s/2,alpha).unsqueeze(1)
    a_father=torch.add(Nk/2,alpha).unsqueeze(1)
    b_sons.index_add_(0, argmax[:, 1], torch.pow(X[:, 2:],2))
    b_father.index_add_(0, argmax[:, 0],torch.pow(X[:, 2:],2))
    b_sons=b_sons/2
    b_father=b_father/2
    b_sons.add_(torch.add(beta,-torch.mul(torch.pow(m_sons,2),v_sons.unsqueeze(1))/2))
    b_father.add_(torch.add(beta,-torch.mul(torch.pow(m_father,2),v_father.unsqueeze(1))/2))

    gamma_2_sons=torch.mvlgamma(a_sons,1)
    gamma_2_father=torch.mvlgamma(a_father,1)

    ll_2_sons=(0.5*torch.log(v_sons).unsqueeze(1))+\
              (torch.log(beta)*alpha)-\
              (a_sons*torch.log(b_sons))+\
              gamma_2_sons-\
              ((torch.mul(torch.log(Global.PI),Nk_s/2))+(0.301*Nk_s)).unsqueeze(1)
    ll_2_father=0.5*torch.log(v_father).unsqueeze(1)+ \
                (a_father*torch.log(b_father))+\
                gamma_2_father- \
                ((torch.mul(torch.log(Global.PI),Nk/2))+(0.301*Nk)).unsqueeze(1)

    ll_2_sons=torch.sum(ll_2_sons,1)[0:ll_sons.shape[0]]
    ll_2_father = torch.sum(ll_2_father, 1)[0:ll_father.shape[0]]

    ll_sons_min = torch.min(ll_2_sons[1:ll_2_sons.shape[0]].masked_select(~(torch.isnan(ll_2_sons[1:ll_2_sons.shape[0]])^torch.isinf(ll_2_sons[1:ll_2_sons.shape[0]]))))
    ll_sons_max = torch.max(ll_2_sons[1:ll_2_sons.shape[0]].masked_select(~(torch.isnan(ll_2_sons[1:ll_2_sons.shape[0]])^torch.isinf(ll_2_sons[1:ll_2_sons.shape[0]]))))
    ll_father_min = torch.min(ll_2_father.masked_select(~(torch.isnan(ll_2_father)^torch.isinf(ll_2_father))))
    ll_father_max = torch.max(ll_2_father.masked_select(~(torch.isnan(ll_2_father)^torch.isinf(ll_2_father))))



    ll_2_sons = torch.div(torch.add(ll_2_sons, -ll_sons_min), (ll_sons_max - ll_sons_min))*(-1000) + 0.1
    ll_2_father = torch.div(torch.add(ll_2_father, -ll_father_min), (ll_father_max - ll_father_min))*(-1000) + 0.1

    ll_sons.add_(ll_2_sons)
    ll_father.add_(ll_2_father)




    gamma_1_sons=torch.mvlgamma(Nk_s,1)
    gamma_1_father=torch.mvlgamma(Nk,1)
    ll_sons=torch.where(Nk_s[0:ll_sons.shape[0]]<35,Global.zeros[0:ll_sons.shape[0]]- torch.Tensor([float("inf")]).to(Global.device),ll_sons)
    ind_sons=clusters_LR[0:gamma_1_sons.shape[0]].long()
    ind_sons[ind_sons>ll_sons.shape[0]-1]=0 #TODO:Check if relevant
    prob=(Global.ALPHA_MS)+\
         ((ll_sons[ind_sons[:,0]]+\
           gamma_1_sons[ind_sons[:,0]]+\
           ll_sons[ind_sons[:,1]]+\
           gamma_1_sons[ind_sons[:,1]])[0:gamma_1_father.shape[0]-1]-
          ((gamma_1_father)+ll_father)[0:gamma_1_father.shape[0]-1])


    idx_rand=torch.where(torch.exp(prob) > 1.0, Global.N_index[0:prob.shape[0]].long(),Global.zeros[0:prob.shape[0]].long()).nonzero()[:, 0]
    if(Global.Print):
        print("Idx Split Size: ",idx_rand.shape[0])
    left = torch.zeros(Global.K_C + 1, 2).int().to(Global.device)
    left[:, 0] = Global.N_index[0:Global.K_C + 1]
    left[idx_rand,1]=1
    pixels_to_change = left[argmax[:,0],1]

    original=torch.where(pixels_to_change==1,argmax[:, 1],argmax[:,0])


    return argmax,Nk,original




def Bass(X, loc):



    if(Global.Print):
        print("CPU->GPU")
    # global SIGMA1
    global SIGMA2
    global SIGMAxylab
    _=1


    X = torch.from_numpy(X).to(Global.device).float()
    loc = torch.from_numpy(loc).to(Global.device).float()

    index2_buffer = torch.zeros(Global.N).to(Global.device)
    r_ikNew_buffer = torch.zeros((Global.N, Global.TIF_C+2)).to(Global.device).reshape(-1)

    range1 = torch.arange(0, Global.N * Global.D_*Global.neig_num).to(Global.device)
    range3 = torch.arange(0, Global.N * Global.neig_num).to(Global.device)
    range4 = torch.arange(0, Global.N * Global.neig_num * Global.D_Inv).to(Global.device)

    range_conn = (torch.arange(Global.N) * Global.neig_num).to(Global.device)

    c1_temp = torch.zeros((Global.N * Global.D_* Global.neig_num)).to(Global.device)

    pi_temp = torch.zeros((Global.N * Global.neig_num)).to(Global.device)

    SigmaXY_temp = torch.zeros((Global.N * Global.neig_num * Global.D_Inv)).to(Global.device).float()
    logdet_temp = torch.zeros((Global.N * Global.neig_num)).to(Global.device)


    SigmaXY_s = torch.zeros(((Global.K_C + 1)*2, Global.D_Inv)).to(Global.device).float()
    SigmaXY_i_s = torch.zeros((Global.K_C + 1)*2, Global.D_Inv).to(Global.device).float()


    SIGMAxylab = torch.zeros((Global.K_C + 1, Global.D_,Global.D_)).to(Global.device).float()
    SIGMAxylab[:, 2, 2] = Global.int_scale
    SIGMAxylab[:, 3, 3] = Global.int_scale
    SIGMAxylab[:, 4, 4] = Global.int_scale




    Nk_s= torch.zeros((Global.K_C + 1)*2).float().to(Global.device)
    X1_s=torch.zeros((Global.K_C+1)*2,Global.D_).float().to(Global.device)


    X2_00_s = torch.zeros((Global.K_C + 1)*2).float().to(Global.device)
    X2_01_s = torch.zeros((Global.K_C + 1)*2).float().to(Global.device)
    X2_11_s = torch.zeros((Global.K_C + 1)*2).float().to(Global.device)

    X_C_SIGMA = torch.zeros(Global.N, Global.neig_num, Global.D_).float().to(Global.device)
    sum_buffer = torch.zeros(Global.N).float().to(Global.device)
    clusters_LR=torch.zeros((Global.HEIGHT)*(Global.WIDTH),2).to(Global.device).int()
    clusters_LR[:,0]=torch.arange(0,(Global.HEIGHT)*(Global.WIDTH)).int()
    XXT=torch.bmm(X[:, 0:2].unsqueeze(2),X[:, 0:2].unsqueeze(1)).reshape(-1,4)
    distances_buffer=torch.zeros(Global.N*Global.neig_num*Global.D_).float().to(Global.device)
    r_ik_5=torch.zeros(Global.N,Global.neig_num).float().to(Global.device)
    neig_buffer=torch.zeros(Global.N,Global.neig_num,Global.potts_area).float().to(Global.device)
    sumP_buffer=torch.zeros(Global.N,Global.neig_num).float().to(Global.device)
    X_C_buffer=torch.zeros(Global.N,Global.neig_num, Global.D_).float().to(Global.device)
    X_C_SIGMA_buf=torch.zeros(Global.N,Global.neig_num,2).float().to(Global.device)

    init=True
    it_merge=0
    it_split=0

    "Start Creating Buffers"

    SigmaXY_b = torch.zeros((Global.N + 1, Global.D_Inv)).to(Global.device).float()
    SigmaXY_i_b = torch.zeros((Global.N + 1, Global.D_Inv)).to(Global.device).float()

    SIGMAxylab_b = torch.zeros((Global.N + 1, Global.D_, Global.D_)).to(Global.device).float()
    SIGMAxylab_b[:, 2, 2] = Global.int_scale
    SIGMAxylab_b[:, 3, 3] = Global.int_scale
    SIGMAxylab_b[:, 4, 4] = Global.int_scale

    Nk_b = torch.zeros(Global.N + 1).float().to(Global.device)
    X1_b = torch.zeros(Global.N + 1, Global.D_).float().to(Global.device)

    X2_00_b = torch.zeros(Global.N + 1).float().to(Global.device)
    X2_01_b = torch.zeros(Global.N + 1).float().to(Global.device)
    X2_11_b = torch.zeros(Global.N + 1).float().to(Global.device)

    sons_LL_b = torch.zeros(Global.N + 1, 4).float().to(Global.device)
    X_sons_b = torch.zeros(Global.N + 1, 2).float().to(Global.device)
    X_father_b = torch.zeros(Global.N + 1, 2).float().to(Global.device)
    father_LL_b = torch.zeros(Global.N + 1, 4).float().to(Global.device)

    m_v_sons_b = torch.zeros(Global.N + 1, Global.TIF_C).float().to(Global.device)
    m_v_father_b = torch.zeros(Global.N + 1, Global.TIF_C).float().to(Global.device)
    b_sons_b = torch.zeros(Global.N + 1, Global.TIF_C).float().to(Global.device)
    b_father_b = torch.zeros(Global.N + 1, Global.TIF_C).float().to(Global.device)
    temp_b = torch.zeros((Global.K_C), 1).long().to(Global.device)

    "End Creating Buffers"


    r_ik, pi = InitBass()
    #while(1):

    Global.K_C=Global.K_C_ORIGINAL
    SigmaXY = SigmaXY_b[0:Global.K_C + 1]
    SigmaXY_i = SigmaXY_i_b[0:Global.K_C + 1]

    SIGMAxylab = SIGMAxylab_b[0:Global.K_C + 1]
    Nk = Nk_b[0:Global.K_C + 1]
    X1 = X1_b[0:Global.K_C + 1]

    X2_00 = X2_00_b[0:Global.K_C + 1]
    X2_01 = X2_01_b[0:Global.K_C + 1]
    X2_11 = X2_11_b[0:Global.K_C + 1]


    it_limit=150

    maxIt = 190

    fixIt_L = 255
    fixIt_H = 275
    it = 0

    argmax = torch.from_numpy(r_ik).to(Global.device).unsqueeze(1).repeat(1,2)
    argmax_start=argmax.clone()

    argmax=argmax_start.clone()

    print("Start")
    if (Global.device == torch.device('cpu')):
        import time
        start = time.time()
    else:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    while (it < maxIt and it<1200):
        it += 1
        if (it % 20 == 0):
            if(Global.Print):
                print("It :", it)
        C1, logdet = M_Step(X, loc, argmax, SigmaXY, SigmaXY_i, Nk, X1, X2_00, X2_01, X2_11, init, Nk_s, X1_s, X2_00_s, X2_01_s, X2_11_s, SigmaXY_s, SigmaXY_i_s, it, maxIt)#Nk_r,X1_r, X2_00_r, X2_01_r,X2_11_r,SigmaXY_r,SigmaXY_l,SigmaXY_i_r,SigmaXY_i_l)  # M-Step

        N_ = X.shape[0]
        pi_t = torch.div(torch.mul(Nk, (1 - Global.PI_0)), (N_ - Nk[0]))
        pi_t[0] = Global.PI_0_T

        if(it>fixIt_L and it%25==1 and it<fixIt_H):
            real_K_C=torch.sum((Nk>2).int())
        if (( (it) % 40 == 1 and (it<it_limit) and (it>60)) or ( (it) % 25 == 1 and it>fixIt_L and it<fixIt_H and real_K_C>Global.K_C_HIGH)) and (Global.split_merges==True):

            if(it>it_limit):
                maxIt+=25
                fixIt_H+=25
                print("Fixing K_C ",real_K_C)

            Merge(X,argmax,Nk,it_merge,temp_b,m_v_father_b,m_v_sons_b,b_father_b,b_sons_b)


        prev_r_ik_max = argmax[:,0].clone()
        c_idx = prev_r_ik_max.view(-1).index_select(0, Global.c_idx)
        c_idx_9 = prev_r_ik_max.view(-1).index_select(0, Global.c_idx_9)
        c_idx_25 = prev_r_ik_max.view(-1).index_select(0, Global.c_idx_25)


        prev_r_ik_max = prev_r_ik_max.view((Global.HEIGHT, Global.WIDTH))
        c1_vals = C1.index_select(0, c_idx).view(-1)
        pi_vals = pi_t.index_select(0, c_idx).view(-1)
        SigmaXY_vals = SigmaXY_i.index_select(0, c_idx).view(-1)
        logdet_vals = logdet.index_select(0, c_idx).view(-1)
        c1_temp.scatter_(0, range1, c1_vals)
        pi_temp.scatter_(0, range3, pi_vals)
        logdet_temp.scatter_(0, range3, logdet_vals)
        SigmaXY_temp.scatter_(0, range4, SigmaXY_vals)

        if(init):
            init=False
            idx_rand=torch.arange(1,Global.K_C+1).to(Global.device)
            argmax[:,1],split,clusters_LR=my_connectivity.Split(prev_r_ik_max,prev_r_ik_max, C1, c1_temp,idx_rand,clusters_LR,it_split)
            it_split=1



        E_Step(X, logdet_temp, c1_temp, pi_temp, SigmaXY_temp.reshape(-1, Global.neig_num, Global.D_Inv), X_C_SIGMA, sum_buffer, c_idx, c_idx_9, c_idx_25, distances_buffer, r_ik_5, neig_buffer, sumP_buffer, X_C_buffer, X_C_SIGMA_buf) #TODO: Check  r_ik_5 pointer
        r_ik_5 = r_ik_5.view(-1, Global.neig_num)



        if(Global.HARD_EM==True):
            argmax[:,0] = my_connectivity.Change_pixel(prev_r_ik_max, r_ik_5, index2_buffer, r_ikNew_buffer, it % 4, c_idx, _,range_conn)#,split_prev_r_ik_max,c_idx_split,r_ik_5_s)
        if (( (it+20)%40   == 1  and it>60 and it<it_limit ) or (it%25==1 and it>fixIt_L and it<fixIt_H and real_K_C<Global.K_C_LOW) ) and (Global.split_merges==True) :

            if(it>it_limit):
                maxIt+=25
                fixIt_H+=25
                print("Fixing K_C ",real_K_C)

            # continue
            argmax[:, 1],sub_clusters,clusters_LR=my_connectivity.Split(argmax[:,0].reshape(Global.HEIGHT,-1),argmax[:,0].reshape(Global.HEIGHT,-1), C1, c1_temp,Global.N_index[1:Global.K_C+1],clusters_LR,it_split)
            argmax,Nk,original = Split(X,XXT,argmax,Nk,sons_LL_b,X_sons_b,X_father_b,father_LL_b,C1,c1_temp,clusters_LR,it_split,m_v_sons_b,m_v_father_b,b_sons_b,b_father_b,SigmaXY_b,SigmaXY_i_b,SIGMAxylab_b,Nk_b,X1_b,X2_00_b,X2_01_b,X2_11_b)


            left2 = torch.zeros((Global.K_C+1)*2 , 2).int().to(Global.device)
            left2[:, 0] = Global.N_index[0:(Global.K_C+1)*2]
            left2[:, 1] = Global.N_index[0:(Global.K_C+1)*2]
            left2[clusters_LR[idx_rand,1].long(),1]=Global.N_index[0:idx_rand.shape[0]].int()+Global.K_C+1
            original=left2[original,1]

            Global.split_lvl[idx_rand]=Global.split_lvl[idx_rand]*0.5
            Global.split_lvl[left2[clusters_LR[idx_rand, 1].long(), 1].long()]=Global.split_lvl[idx_rand]


            Global.K_C=torch.max(original.reshape(-1)).int()


            SigmaXY = SigmaXY_b[0:Global.K_C + 1]
            SigmaXY_i = SigmaXY_i_b[0:Global.K_C + 1]

            SIGMAxylab = SIGMAxylab_b[0:Global.K_C + 1]
            Nk = Nk_b[0:Global.K_C + 1]
            X1 = X1_b[0:Global.K_C + 1]

            X2_00 = X2_00_b[0:Global.K_C + 1]
            X2_01 = X2_01_b[0:Global.K_C + 1]
            X2_11 = X2_11_b[0:Global.K_C + 1]

            prev_r_ik_max = argmax[:, 0].clone()
            argmax[:,0]=original.reshape(-1)


        else:
            prev_r_ik_max = r_ik_5.argmax(1)
            prev_r_ik_max=torch.take(c_idx,torch.add(prev_r_ik_max,range_conn))

            Nk.zero_()
            Nk.index_add_(0, argmax[:, 0], Global.ones)
            c_idx = prev_r_ik_max.view(-1).index_select(0, Global.c_idx)
            c_idx=c_idx.reshape(-1, Global.neig_num)[:,1]
            argmax[:,0]=torch.where(Nk[argmax[:,0]]==1,c_idx,argmax[:,0])

    if (Global.device == torch.device('cpu')):
        end = time.time()
        print("Time taken: ", end-start)
    else:
        end.record()
        torch.cuda.synchronize()
        print("Time taken: ",start.elapsed_time(end))

    Nk.zero_()
    Nk.index_add_(0, argmax[:, 0], Global.ones)
    c_idx = prev_r_ik_max.view(-1).index_select(0, Global.c_idx)
    c_idx=c_idx.reshape(-1, Global.neig_num)[:,1]
    argmax[:,0]=torch.where(Nk[argmax[:,0]]==1,c_idx,argmax[:,0])
    r_ik_for_print = argmax[:,0].cpu().numpy()
    r_ik2_for_print = np.reshape(r_ik_for_print, (Global.HEIGHT, Global.WIDTH)).astype(int)
    framePointsNew = np.zeros((Global.HEIGHT+2, Global.WIDTH+2, Global.TIF_C))
    mean_value = np.zeros((Global.K_C + 1, Global.TIF_C))
    mean_value2=np.zeros((Global.K_C+1,3))


    for i in range(0,Global.K_C+1):
        mean_value2[i] = np.array([255, 0, 0])


    if 1:
        #framePointsNew2 = np.zeros((Global.HEIGHT+2, Global.WIDTH+2, 3))
        framePointsNew3 = np.zeros((Global.HEIGHT+2, Global.WIDTH+2, 3)).astype(np.int)

        if Global.TIF:
            framePointsNew3[1:Global.HEIGHT+1,1:Global.WIDTH+1]= 0 # Global.frame0.astype(np.int)
        else:
            framePointsNew3[1:Global.HEIGHT+1,1:Global.WIDTH+1]= Global.frame0.astype(np.int)
        for i in range(0, Global.K_C + 1):
            if(len(Global.frame0[r_ik2_for_print == i])):
                mean_value[i] = np.mean(Global.frame0[r_ik2_for_print == i], axis=0)


        padded=np.pad(r_ik2_for_print, 1, pad_with, padder=0)



        for i in range(padded.shape[0]):
            for j in range(padded.shape[1]):
                if(padded[i,j]!=0):
                    framePointsNew[i, j] = mean_value[padded[i, j]]
                    #framePointsNew2[i, j] = mean_value[padded[i, j]]
                    if(((padded[i+1,j]>0)and(padded[i,j]!=padded[i+1,j])) or ((padded[i,j-1]>0)and(padded[i,j]!=padded[i,j-1])) or ((padded[i,j+1]>0)and(padded[i,j]!=padded[i,j+1])) or ((padded[i-1,j]>0)and(padded[i,j]!=padded[i-1,j]))):
                        #framePointsNew2[i,j]=mean_value2[padded[i,j]]
                        framePointsNew3[i,j]=mean_value2[padded[i,j]]

        painted=np.zeros(Global.K_C+1)
        count=0
        for i in range(0, Global.HEIGHT):
            for j in range(0, Global.WIDTH):
                if(painted[r_ik2_for_print[i,j]]==0):
                    count=count+1
                    painted[r_ik2_for_print[i,j]]=1

        framePointsNew=framePointsNew[1:Global.HEIGHT+1,1:Global.WIDTH+1]
        framePointsNew3=framePointsNew3[1:Global.HEIGHT+1,1:Global.WIDTH+1]

        if(Global.Plot):
            framePointsNew = framePointsNew.astype('uint8')
            framePointsNew3 = framePointsNew3.astype('uint8')


            os.makedirs(os.path.join(str(Global.save_folder),'contour'),exist_ok=True)
            os.makedirs(os.path.join(str(Global.save_folder),'mean'),exist_ok=True)

            if not Global.TIF:
                pil_im = Image.fromarray(framePointsNew)

            pil_im2 = Image.fromarray(framePointsNew3)

            if Global.TIF:
                from tifffile import imsave
                framePointsNew = np.swapaxes(np.swapaxes(framePointsNew,0,2),1,2)

                imsave(os.path.join(str(Global.save_folder),'mean',(str(Global.csv_file)+'.tif')),framePointsNew,metadata=Global.meta_data)
            else:
                pil_im.save(os.path.join(str(Global.save_folder),'mean',(str(Global.csv_file)+'.png')),"PNG", optimize=True)
            pil_im2.save(os.path.join(str(Global.save_folder),'contour',(str(Global.csv_file)+'.png')),"PNG", optimize=True)

    if(Global.csv):
        r_ik2 = r_ik2_for_print
        for i in range(r_ik2.shape[0]):
            for j in range(r_ik2.shape[1]):
                if(i+j==0):
                    dict={}
                    count_new=0
                    b=np.zeros((r_ik2.shape)).astype(np.uint16)
                try:
                    b[i,j]=dict[r_ik2[i,j]]
                except Exception as e:
                    dict[r_ik2[i,j]]=count_new
                    count_new+=1
                    b[i,j]=dict[r_ik2[i,j]]
        K_C=np.unique(b).shape[0]
        mean_x=np.zeros(K_C+1)
        mean_y=np.zeros(K_C+1)
        ind=np.indices((b.shape))
        for i in range(K_C+1):
           mean_x[i]=np.mean(ind[0][b==i])
           mean_y[i]=np.mean(ind[1][b==i])


        d2=((mean_x[0]-mean_x)*2+(mean_y[0]-mean_y)*2)
        sorted_args=d2.argsort()
        c=np.copy(b)*0
        for i in range(K_C+1):
           c[b==sorted_args[i]]=K_C-i-1
        r_ik2=c
        os.makedirs(os.path.join(str(Global.save_folder),'csv'),exist_ok=True)
        with open(os.path.join(str(Global.save_folder),'csv',(str(Global.csv_file)+'.csv')), "w") as f:
            writer = csv.writer(f)
            writer.writerows(r_ik2)





def E_Step(X, logdet, c1_temp, pi_temp, SigmaXY, X_C_SIGMA, sum, c_idx, c_idx_9, c_idx_25, distances2, r_ik_5, neig, sumP, X_C, X_C_SIGMA_buf):

    """
    Computes the distances of the Data points for each centroid and normalize it,

    """
    torch.add(X.unsqueeze(1), torch.neg(c1_temp.reshape(-1, Global.neig_num, Global.D_)),out=X_C)
    torch.mul(X_C[:, :, 0].unsqueeze(2), SigmaXY[:, :, 0:2],out=X_C_SIGMA_buf)
    torch.addcmul(X_C_SIGMA_buf,1,X_C[:,:,1].unsqueeze(2),SigmaXY[:,:,2:4],out=X_C_SIGMA[:,:,0:2])
    X_C_SIGMA[:, :, 2:] = torch.mul(X_C[:, :, 2:], Global.SIGMA_INT)

    torch.mul(-X_C.view(-1, Global.neig_num,Global.D_),X_C_SIGMA.view(-1,Global.neig_num,Global.D_),out=distances2)
    distances2=distances2.view(-1,Global.neig_num,Global.D_)
    torch.sum(distances2,2,out=r_ik_5)

    r_ik_5.add_(torch.neg(logdet.reshape(-1, Global.neig_num)))
    r_ik_5.add_(torch.log(pi_temp.reshape(-1, Global.neig_num)))
    c_neig = c_idx_25.reshape(-1, Global.potts_area).float()
    torch.add(c_neig.unsqueeze(1), -c_idx.reshape(-1, Global.neig_num).unsqueeze(2).float(),out=neig)
    torch.sum((neig!=0).float(),2,out=sumP)
    r_ik_5.add_(-(Global.Beta_P*sumP))
    (my_help.softmaxTF(r_ik_5, 1,sum))




def InitBass():
    """Initialize the clutsters probability for each point.

    **Parameters**:
     -

    **Returns**:
     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer] .

    .. note:: The probobility randomize unirformly

    .. note:: Changed the initialzation similar for adjacent pixels (3x3)

    """

    x, y = np.mgrid[:Global.HEIGHT, :Global.WIDTH]
    X = np.array((x.ravel(), y.ravel()))
    s=np.int(np.sqrt(Global.N/Global.K_C))
    a=np.int(Global.WIDTH/s)
    b=np.int(Global.HEIGHT/s)
    a_2=a+2


    h, w = np.mgrid[Global.HEIGHT/(2*(b+1)):Global.HEIGHT-1:(Global.HEIGHT/(b+1)),Global.WIDTH/(2*(a+1)):Global.WIDTH-1:(Global.WIDTH/(a+1))]
    h2, w2 = np.mgrid[Global.HEIGHT/(2*(b+1)):Global.HEIGHT-1:(Global.HEIGHT/(b+1)),Global.WIDTH/(2*(a_2+1)):Global.WIDTH-1:(Global.WIDTH/(a_2+1))]

    C = np.array((h.ravel(), w.ravel()),dtype=np.float).T
    C2 = np.array((h2.ravel(), w2.ravel()),dtype=np.float).T





    width=(C[2][1]-C[1][1])*0.5

    if(Global.Print):
        print("Inital number of superpixels : ",C.shape[0])

    C_0 = np.array([[Global.HEIGHT * 5, Global.HEIGHT * 5]])
    C = np.append(C_0, C, axis=0)
    voronoi_kdtree = cKDTree(C)
    extraPoints = X.transpose()
    test_point_dist, test_point_regions = voronoi_kdtree.query(extraPoints)


    Global.K_C = C.shape[0]
    Global.K_C_ORIGINAL = C.shape[0]
    Global.RegSize=Global.N/Global.K_C
    Global.split_lvl= Global.split_lvl+Global.RegSize
    Global.A_prior=Global.N/(Global.K_C)
    r_ik=test_point_regions


    return r_ik, r_ik


"""

Estimation Function

"""





def M_Step(X, loc, argmax, Sigma, SigmaInv, Nk, X1, X2_00, X2_01, X2_11, init, Nk_s, X1_s, X2_00_s, X2_01_s, X2_11_s, SigmaXY_s, SigmaInv_s, it, max_it): #Nk_r,X1_r, X2_00_r, X2_01_r,X2_11_r,SigmaXY_r,SigmaXY_l,SigmaInv_r,SigmaInv_l):

    Nk.zero_()
    Nk_s.zero_()
    X1.zero_()
    X2_00.zero_()
    X2_01.zero_()
    X2_11.zero_()
    argmax=argmax[:,0]
    Nk.index_add_(0, argmax, Global.ones)
    Nk = Nk + 0.0000000001
    X1.index_add_(0,argmax,X)

    C = torch.div(X1, Nk.unsqueeze(1))
    mul=torch.pow(loc[:,0],2)

    X2_00.index_add_(0,argmax,mul)

    mul=torch.mul(loc[:,0],loc[:,1])
    X2_01.index_add_(0,argmax,mul)


    mul=torch.pow(loc[:,1],2)
    X2_11.index_add_(0,argmax,mul)


    Sigma00=torch.add(X2_00,-torch.div(torch.pow(X1[:,0],2),Nk))
    Sigma01=torch.add(X2_01,-torch.div(torch.mul(X1[:,0],X1[:,1]),Nk))
    Sigma11=torch.add(X2_11,-torch.div(torch.pow(X1[:,1],2),Nk))


    a_prior=Global.split_lvl[0:Nk.shape[0]]



    Global.psi_prior=torch.mul(torch.pow(a_prior,2).unsqueeze(1),torch.eye(2).reshape(-1,4).to(Global.device))
    Global.ni_prior=(Global.C_prior*a_prior)-3


    Sigma[:, 0] = torch.div(torch.add(Sigma00, Global.psi_prior[:,0]), torch.add(Nk, Global.ni_prior))
    Sigma[:, 1] = torch.div((Sigma01), torch.add(Nk, Global.ni_prior))
    Sigma[:, 2] = Sigma[:, 1]
    Sigma[:, 3] = torch.div(torch.add(Sigma11, Global.psi_prior[:,3]), torch.add(Nk, Global.ni_prior))

    det=torch.reciprocal(torch.add(torch.mul(Sigma[:,0],Sigma[:,3]),-torch.mul(Sigma[:,1],Sigma[:,2])))
    det[(det <= 0).nonzero()] = 0.00001

    SigmaInv[:, 0] = torch.mul(Sigma[:, 3], det)
    SigmaInv[:, 1] = torch.mul(-Sigma[:, 1], det)
    SigmaInv[:, 2] = torch.mul(-Sigma[:, 2], det)
    SigmaInv[:, 3] = torch.mul(Sigma[:, 0], det)

    SIGMAxylab[:,0:2,0:2]=Sigma[:,0:4].view(-1,2,2)
    logdet=torch.log(torch.mul(torch.reciprocal(det),Global.detInt))
    return C,logdet






"""
************************************************************************************************************************************************************

                                                                        Main Funtions

************************************************************************************************************************************************************
"""




def SuperPixelsSplitMerge():
    X0 = my_help.Create_DataMatrix(Global.frame0)
    Bass(X0, X0[:, 0:2])



def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
        return vector






"""-
************************************************************************************************************************************************************

                                                                            Main

************************************************************************************************************************************************************
"""

if __name__ == "__main__":
    extensions = ("*.png","*.jpg","*.jpeg")
    parser = argparse.ArgumentParser(description='BASS')
    Global.save_folder='out_img'
    parser.add_argument('--img_folder',
                        type=str,
                        default='./img',
                        help='path of the input folder'
                        )
    parser.add_argument('--vis',
                        action='store_true',
                        help='visualize'
                        )
    parser.add_argument('--csv',
                        action='store_true',
                        help='save segmentation as CSV file'
                        )
    parser.add_argument('--sp',
                        type=int,
                        default=400,
                        help='initial number of superpixels'
                        )
    parser.add_argument('--cpu',
                        action='store_true',
                        help='use cpu'
                        )
    parser.add_argument('--add_splits',
                        type=int,
                        default=0,
                        help='large number for more splits (range -10 to 10)'
                        )
    parser.add_argument('--potts',
                    type=float,
                    default=2.7,
                    help='potts term'
                    )
    parser.add_argument('--tif',
                        action='store_true',
                        help='tif file'
                        )
    parser.add_argument('--v',
                        action='store_true',
                        help='verbose'
                        )

    args = parser.parse_args()
    Global.Plot = args.vis
    Global.Print = args.v
    Global.device = torch.device('cpu' if args.cpu else 'cuda')
    Global.add_splits = args.add_splits
    Global.TIF = args.tif
    if args.tif:
        extensions = ("*.tif","*.tiff")
    directory = args.img_folder


    if (not os.path.exists(directory)):
        raise ValueError("Input folder doesn't exist")
    if (args.sp<0):
        raise ValueError("Initial number of superpixels must be positive")
    Global.csv = args.csv
    np.random.seed(34)
    torch.manual_seed(23)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    image_files = []
    for extenstion in extensions:
        image_files.extend(glob.glob(os.path.join(directory, extenstion)))
    image_files = sorted(image_files)

    Global.repeat=False
    count = 0
    for Global.IMAGE1 in image_files:
        torch.cuda.empty_cache()
        count = count + 1
        Global.initVariables()
        Global.Beta_P = (args.potts-2.7) + Global.Beta_P
        Global.K_C = args.sp
        Global.csv_file=Global.IMAGE1[Global.IMAGE1.rfind("/")+1:][:-4]
        Global.repeat=False
        print(Global.csv_file)
        SuperPixelsSplitMerge()
