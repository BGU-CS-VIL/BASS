from BASS import *


import numpy as np
import cv2

def Create_LookupTable():
    return np.load('./lookup.npy')

def binary(num, length=8):
    return int(format(num, '#0{}b'.format(length + 2)))

def Create_Matrix(padded_matrix,mod):
    if(mod==1):
        pixels = padded_matrix.index_select(0, Global.idx_pixels_1).view(9, -1)
    elif (mod == 2):
        pixels = padded_matrix.index_select(0, Global.idx_pixels_2).view(9, -1)
    elif (mod == 3):
        pixels = padded_matrix.index_select(0, Global.idx_pixels_3).view(9, -1)
    elif (mod == 0):
        pixels = padded_matrix.index_select(0, Global.idx_pixels_4).view(9, -1)
    matrix=torch.add(-pixels.unsqueeze(0),pixels.unsqueeze(1))
    matrix=torch.add((matrix==0),(pixels==(-1)))
    return matrix>0

def Create_Matrix_Sub(padded_matrix_split,mod):
    pixels_sub=padded_matrix_split.index_select(0,Global.idx_pixels[mod]).view(9,-1)
    matrix_sub=torch.add(-pixels_sub.unsqueeze(0),pixels_sub.unsqueeze(1))
    matrix=((matrix_sub==0))
    return matrix>0

def Create_number(matrix):
    temp=torch.index_select(matrix,1,Global.shift_index)
    return (torch.sum(temp.byte()<<Global.shift,dim=1))

def Change_pixel(prev_r_ik,r_ik,index2,r_ik_t5,mod,c_vals,r_ik_tk,range_conn):

    padded_matrix = Global.Padding(prev_r_ik)
    padded_matrix = padded_matrix.view(-1)
    number = Create_number(Create_Matrix(padded_matrix, mod))
    number = torch.index_select(number, 0, Global.ne4_idx)
    number2 = torch.index_select(Global.LOOKUP_TABLE, 0, number.view(-1))
    ind = torch.all(number2.view(5, -1), dim=0).float()  # ind that we can change
    index2.zero_()
    if(mod==1):
        index2.scatter_(0, Global.idx_1, ind)  # 0 -same as before , 1 - Can change (All N pixels)
    if (mod == 2):
        index2.scatter_(0, Global.idx_2, ind)  # 0 -same as before , 1 - Can change (All N pixels)
    if (mod == 3):
        index2.scatter_(0, Global.idx_3, ind)  # 0 -same as before , 1 - Can change (All N pixels)
    if (mod == 0):
        index2.scatter_(0, Global.idx_4, ind)  # 0 -same as before , 1 - Can change (All N pixels)
    # index2 = index2.scatter_(0, Global.idx[:, mod], ind)  # 0 -same as before , 1 - Can change (All N pixels)
    max_neigh=torch.argmax(r_ik, dim=1)
    valsNew = torch.add(max_neigh,range_conn)
    valsK = torch.where(index2 == 1, c_vals[valsNew], prev_r_ik.view(-1))


    return valsK

def Split(prev_r_ik,split_prev_r_ik,c1,c1_idx,idx_rand,clusters_LR,it_split):
    idxL=idx_rand.long()
    padded_matrixR = Global.Padding0(split_prev_r_ik).reshape(-1)
    padded_matrixL = Global.Padding0(split_prev_r_ik).reshape(-1)

    distances=Global.Distances_padded
    distances.zero_()
    centers=torch.index_select(c1,0,idxL)[:,0:2]
    new_centerL= torch.add(centers, -2).floor_().int()
    if(it_split%2==0):
        new_centerL[:, 0]=new_centerL[:,0]+3
        new_centerL[:, 1] = new_centerL[:, 1] - 2
    else:
        new_centerL[:, 1]=new_centerL[:,1]+4
        new_centerL[:, 0] = new_centerL[:, 0] - 2
    idxR = idx_rand.long()
    centersR = torch.index_select(c1, 0, idxR)[:, 0:2]
    new_centerR = torch.add(centersR, +2).ceil_().int()

    if (it_split % 2 == 0):
        new_centerR[:, 0] = new_centerR[:, 0] - 3
        new_centerR[:, 1] = new_centerR[:, 1] + 2
    else:
        new_centerR[:, 1] = new_centerR[:, 1] - 4
        new_centerR[:, 0] = new_centerR[:, 0] + 2

    new_center_idxR = new_centerR[:, 0] * (Global.WIDTH + 2) + new_centerR[:, 1] + 1  # idx in padded
    new_center_idxL = new_centerL[:,0]*(Global.WIDTH+2)+new_centerL[:,1]+1 #idx in padded
    condL=((new_centerL[:,0]<Global.HEIGHT)&(new_centerL[:,0]>=0)&(new_centerL[:,1]<Global.WIDTH)&(new_centerL[:,1]>=0))
    condR=((new_centerR[:,0]<Global.HEIGHT)&(new_centerR[:,0]>=0)&(new_centerR[:,1]<Global.WIDTH)&(new_centerR[:,1]>=0))
    cond=(condL&condR)

    new_center_idxL=torch.masked_select(new_center_idxL,cond)
    new_center_idxR=torch.masked_select(new_center_idxR,cond)
    idxL=torch.masked_select(idxL,cond)
    idxR=torch.masked_select(idxR,cond)

    new_center_idxL =torch.masked_select(new_center_idxL,((padded_matrixL[new_center_idxL.long()]==idxL)&(new_center_idxL>0)))
    new_center_idxR =torch.masked_select(new_center_idxR,((padded_matrixR[new_center_idxR.long()]==idxR)&(new_center_idxR>0)))

    prev_centersL=padded_matrixL.take(new_center_idxL.long())
    prev_centersR=padded_matrixR.take(new_center_idxR.long())




    if(new_center_idxL.size()==0):
        return 0

    neigh=Global.neig_padded
    dataL=torch.zeros((Global.HEIGHT+2)*(Global.WIDTH+2),2).to(Global.device).int()
    dataL[:,0]=torch.arange(0,(Global.HEIGHT+2)*(Global.WIDTH+2)).int()
    data2=dataL.clone()
    dataR=dataL.clone()

    dataL[new_center_idxL.long(),1] = prev_centersL.int()
    dataR[:, 1].zero_()
    dataR[new_center_idxR.long(), 1] = prev_centersR.int()
    loop=1
    distance_add=Global.ones.clone()
    valid_vL = new_center_idxL
    valid_vR = new_center_idxR
    while (loop):
        distance_add.add_(1)
        exp_clustersL = torch.take(padded_matrixL, valid_vL.long()).unsqueeze(1).repeat(1, 4).reshape(-1)
        idxL = torch.add(neigh.unsqueeze(0), valid_vL.unsqueeze(1))
        padded_matrixL.index_fill_(0, valid_vL.long(), 0)
        valid_indL = torch.masked_select(idxL.reshape(-1), padded_matrixL[idxL.reshape(-1).long()] == exp_clustersL)
        dataL[valid_indL.long(), 1] = padded_matrixL[valid_indL.long()].int()
        valid_indL = torch.masked_select(dataL[:, 0], dataL[:, 1] > 0)
        exp_clustersR = torch.take(padded_matrixR, valid_vR.long()).unsqueeze(1).repeat(1, 4).reshape(-1)
        idxR = torch.add(neigh.unsqueeze(0), valid_vR.unsqueeze(1))
        padded_matrixR.index_fill_(0, valid_vR.long(), 0)
        valid_indR = torch.masked_select(idxR.reshape(-1), padded_matrixR[idxR.reshape(-1).long()] == exp_clustersR)
        dataR[valid_indR.long(), 1] = padded_matrixR[valid_indR.long()].int()
        valid_indR = torch.masked_select(dataR[:, 0], dataR[:, 1] > 0)
        if ((valid_indL.shape[0]  < 1)and(valid_indL.shape[0]  < 1)):
            loop = 0
        else:
            if(valid_indL.shape[0] >0):
                distances[:, 0].index_add_(0, valid_indL.long(), distance_add[0:valid_indL.shape[0]])
                dataL[:, 1].zero_()
                valid_vL = valid_indL
            if(valid_indR.shape[0]>0):
                distances[:, 1].index_add_(0, valid_indR.long(), distance_add[0:valid_indR.shape[0]])
                dataR[:, 1].zero_()
                valid_vR = valid_indR

    distances[new_center_idxL.long(),0]=0


    idx=idx_rand.long()
    idx_args=torch.arange(0,idx.shape[0]).to(Global.device)
    split_idx=torch.masked_select(dataL[:,0],distances[:,1]>0).long()
    idx_final=torch.zeros(idx.shape[0],2).to(Global.device)
    idx_final[:,0]=idx_args
    idx_final[:,0]=idx
    padded_matrix = Global.Padding0(split_prev_r_ik).reshape(-1)
    distances[new_center_idxR.long(),1]=0
    new_idx=torch.where(distances[split_idx,0]<distances[split_idx,1],Global.zeros[:distances[split_idx,0].shape[0]],Global.ones[:distances[split_idx,0].shape[0]])

    new_cluster_idx=torch.masked_select(split_idx,new_idx.bool())
    dataR[:,1].zero_()
    idx=idx.long()
    data2[idx,1]=Global.N_index[0:idx.shape[0]].int()+2+torch.max(split_prev_r_ik.reshape(-1)).int()

    dataR[new_cluster_idx.long(),1]= ( Global.Padding0(split_prev_r_ik).reshape(-1).to(Global.device)[new_cluster_idx]).int()
    dataR[new_cluster_idx.long(),1]=(data2[( Global.Padding0(split_prev_r_ik).reshape(-1).to(Global.device)[new_cluster_idx]).long(),1])
    padded_matrix[new_cluster_idx.long()]=(dataR[new_cluster_idx.long(),1].long()).clone()
    clusters_LR[:,1].zero_()
    clusters_LR[idx,1]=data2[idx,1]

    prev_r_ik_padded =  Global.Padding0(prev_r_ik).reshape(-1)

    prev_r_ik_padded[split_idx.long()]=( Global.Padding0(split_prev_r_ik.to(Global.device)).reshape(-1)[split_idx.long()]).clone() #NEW

    padded_matrix=padded_matrix.reshape(Global.HEIGHT+2,-1)

    prev_r_ik_padded=prev_r_ik_padded.reshape(Global.HEIGHT+2,-1) # NEW
    return padded_matrix[1:Global.HEIGHT+1,1:Global.WIDTH+1].reshape(-1),prev_r_ik_padded[1:Global.HEIGHT+1,1:Global.WIDTH+1].reshape(-1),clusters_LR


def Merge(prev_r_ik,split_prev_r_ik,clusters_LR):
    padding = torch.nn.ConstantPad2d((1, 1, 1, 1), 0).to(Global.device)
    padded_matrix = padding(prev_r_ik).reshape(-1).to(Global.device)
    padded_matrix_split = padding(split_prev_r_ik).reshape(-1).to(Global.device)

    left=torch.zeros(Global.K_C+1).int().to(Global.device)
    pair=torch.zeros(Global.K_C+1).int().to(Global.device)

    ind_left=torch.masked_select(Global.inside_padded,((padded_matrix[Global.inside_padded]==padded_matrix[Global.inside_padded-1])&(padded_matrix[Global.inside_padded-1]>0)))
    left[padded_matrix[ind_left]]=padded_matrix[ind_left-1].int()

    for i in range(0,Global.K_C+1):
        val=left[i]
        if(val>0):
            if((pair[i]==0)and(pair[val]==0)):
                if(val<i):
                    pair[i]=val
                    pair[val]=val
                else:
                    pair[val]=i
                    pair[i]=i

    pair_temp=torch.arange(0,Global.K_C+1).to(Global.device).int()
    pair_new=torch.where(pair>0,pair,pair_temp)

    padded_matrix=pair_new[padded_matrix]

    padded_matrix=padded_matrix.reshape(Global.HEIGHT+2,-1)
    padded_matrix_split=padded_matrix_split.reshape(Global.HEIGHT+2,-1)
    return padded_matrix[1:Global.HEIGHT + 1, 1:Global.WIDTH + 1].reshape(-1),padded_matrix_split[1:Global.HEIGHT + 1, 1:Global.WIDTH + 1].reshape(-1)
