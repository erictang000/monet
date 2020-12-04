"""
CTRW_single_net_testing.py
This script contains functions for activating and testing of the alpha exponent
of any given CTRW trajectory.
"""

import numpy as np
from keras.models import load_model
from utils import generate_CTRW
import seaborn as sns
import scipy.io


def CTRW_net_on_file(file):
    ### change here to load a different network model
    net_file = './Models/model-alphaCTRW-estimate_300.h5'


    model = load_model(net_file)

    # load mat file and extract trajectory data
    
    f = scipy.io.loadmat(file)
    for k in f.keys():
        if k[0]=='_':
            continue
        varName = k
        data = f[varName]
        NAxes = (np.shape(data)[1]-1)
    
    numTraj = len(np.unique(data[:,NAxes]))
    alpha_pred = np.zeros([numTraj,1])
    flag = np.zeros([numTraj,1])
    pred_mean = np.zeros((1,1))
    for i in np.arange(1,numTraj+1):
        pred = np.zeros((NAxes,1))
        for j in range(NAxes):
            x = data[np.argwhere(data[:,NAxes]==i),j]
            x = x-np.mean(x)
            if len(x)>=300: # classification network is trained on 300 step trajectories
                flag[i-1] = 1 # mark trajectories that are being analyzed
                dx = np.diff(x,axis=0)
                variance= np.std(dx)
                dx =dx/variance
                dx = np.reshape(dx[:299],[1,299,1]) #change this number based on step size
            pred[j,:] = model.predict(dx) # get the results for 1D 
            pred_mean = np.mean(pred,axis=0) # calculate mean prediction of N-dimensional trajectory 
        alpha_pred[i-1,:] = pred_mean

    
    return alpha_pred


    
# def CTRW_on_sim():

# print('hello')
# dx,label1=generate_CTRW(batchsize=32,steps=300,T=15,sigma=0.1)
# ## change here to load a different network model   
# N =np.shape(dx)[0]
# net_file = './Models/model-alphaCTRW-estimate.h5'
# model = load_model(net_file)     
# for j in range(N):
#     dummy = np.zeros((1,299,1))
#     dummy[0,:,:] = dx[j,:,:]
#     alpha = model.predict(dummy) # get the results for 1D 
#     print(alpha)
#     print(label1[j])
#     print('-') 

 
#CTRW_on_sim()
alpha_mean=CTRW_net_on_file(file = './data/20191212-20200116-20200313-AuNRs_300.mat')
pickle.dump(alpha_mean, open( "./results/alpha-20191212-20200116-20200313-300.p", "wb" ) )