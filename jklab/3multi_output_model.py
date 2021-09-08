###############################################################################
# This file contains the code to train the SpliceAI model.
###############################################################################

import numpy as np
import sys
import time
import scipy
import h5py
import keras.backend as kb
import tensorflow as tf
from spliceai import *
from utils import *
from multi_gpu import *
from constants import * 
from keras.models import load_model
import os
import random

assert int(sys.argv[1]) in [80, 200, 400, 800, 2000, 10000]

###############################################################################
# Model
###############################################################################

SL=5000
CL_max=400
data_dir='./'

file_name=str(sys.argv[2]).split('/')[2].split('.')[0]


L = 32
N_GPUS = 2

if int(sys.argv[1]) == 80:
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    BATCH_SIZE = 18*N_GPUS

elif int(sys.argv[1]) == 200:
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1, 1, 1, 2, 2])
    BATCH_SIZE = 18*N_GPUS    
    
elif int(sys.argv[1]) == 400:
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
    BATCH_SIZE = 18*N_GPUS
    
elif int(sys.argv[1]) == 800:
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                     5,5,5,5])
    BATCH_SIZE = 12*N_GPUS
    
elif int(sys.argv[1]) == 2000:
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    21, 21, 21, 21])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                     10, 10, 10, 10])
    BATCH_SIZE = 12*N_GPUS
elif int(sys.argv[1]) == 10000:
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    21, 21, 21, 21, 41, 41, 41, 41])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                     10, 10, 10, 10, 25, 25, 25, 25])
    BATCH_SIZE = 6*N_GPUS
# Hyper-parameters:
# L: Number of convolution kernels
# W: Convolution window size in each residual unit
# AR: Atrous rate in each residual unit

CL = 2 * np.sum(AR*(W-1))
assert CL <= CL_max and CL == int(sys.argv[1])
print("Context nucleotides: %d" % (CL))
print("Sequence length (output): %d" % (SL))



a3_flag = False
a5_flag = False
io_flag = True


model = load_model('400_2o_default_model.h5')
loss_list=[categorical_crossentropy_2o]


model.summary()
model_m = make_parallel(model, N_GPUS)
model_m.compile(loss=loss_list, optimizer='adam')

###############################################################################
# Training and validation
###############################################################################

h5f = h5py.File('dataset_train_all.h5', 'r')


a3ss = h5py.File('a3ss_inside_out.h5', 'r')
a5ss = h5py.File('a5ss_inside_out.h5', 'r')



### training : 0~1315154
### val : 1315155~1348876
### test : 1348877~1686095


num_idx = len(h5f.keys())//2
idx_all = np.random.permutation(num_idx)
idx_train = idx_all[:int(0.9*num_idx)]
idx_valid = idx_all[int(0.9*num_idx):]

num_idx_a5ss = len(a5ss.keys())//2
idx_all_a5ss = np.array([i for i in range(num_idx_a5ss)])
idx_train_a5ss = idx_all_a5ss[:178]
idx_valid_a5ss = idx_all_a5ss[178:-1]


num_idx_a3ss = len(a3ss.keys())//2
idx_all_a3ss = np.array([i for i in range(num_idx_a3ss)])
idx_train_a3ss = idx_all_a3ss[:226]
idx_valid_a3ss = idx_all_a3ss[226:-1]


EPOCH_NUM = 15

start_time = time.time()


acceptor_best=0.0
donor_best=0.0




for epoch_num in range(EPOCH_NUM):
    iter_list=[]
    div=1
    tim=9
    
    for _ in range(int((178/div))*tim):
        idx = np.random.choice(len(idx_train_a5ss))
        iter_list.append( ( idx , 1 ) )

    for _ in range(int((226/div))):
        idx = np.random.choice(len(idx_train_a3ss))
        iter_list.append( ( idx , 2 ) )

    random.shuffle(iter_list)
        
    for it in iter_list:
        idx=it[0]
       
        if it[1]==0:
            X = h5f['X' + str(idx)][:]
            Y = h5f['Y' + str(idx)][:]
            

        elif it[1]==1:
            X = a5ss['X' + str(idx)][:]
            Y = a5ss['Y' + str(idx)][:]
            

        else:
            X = a3ss['X' + str(idx)][:]
            Y = a3ss['Y' + str(idx)][:]
            
            
        
        Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)
        
        real_Yc=[]
        real_Yc.append(Yc[0])
        
        model_m.fit(Xc, real_Yc, batch_size=BATCH_SIZE, verbose=0)


    # Printing metrics (see utils.py for details)

        ###########################

        
    Y_true_sd1 = np.array([])
    Y_pred_sd1 = np.array([])
    Y_true_sd2 = np.array([])
    Y_pred_sd2 = np.array([])

    for idx in idx_valid_a5ss:

        X = a5ss['X' + str(idx)][:]
        Y = a5ss['Y' + str(idx)][:]

        Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)

        Yp = model_m.predict(Xc, batch_size=BATCH_SIZE)[0]

        Y_true_sd1=np.concatenate((Y_true_sd1,Y[0][:,518+0+1,1]))
        Y_pred_sd1=np.concatenate((Y_pred_sd1,Yp[:,518+0+1,1]))

        Y_true_sd2=np.concatenate((Y_true_sd2,Y[0][:,518+44+1,1]))
        Y_pred_sd2=np.concatenate((Y_pred_sd2,Yp[:,518+44+1,1]))

    print("\nA5SS IO SD1: " + str(scipy.stats.pearsonr(Y_true_sd1,Y_pred_sd1)[0]**2) )
    print("\nA5SS IO SD2: " + str(scipy.stats.pearsonr(Y_true_sd2,Y_pred_sd2)[0]**2) )


    
    ###########################
    ###########################
    ###########################
    ###########################
    ###########################
    ###########################
    ###########################
    ###########################
    ###########################
    
   
    Y_true_sa1 = np.array([])
    Y_pred_sa1 = np.array([])
    Y_true_sa2 = np.array([])
    Y_pred_sa2 = np.array([])

    for idx in idx_valid_a3ss:

        X = a3ss['X' + str(idx)][:]
        Y = a3ss['Y' + str(idx)][:]

        Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)

        Yp = model_m.predict(Xc, batch_size=BATCH_SIZE)

        Y_true_sa1=np.concatenate((Y_true_sa1,Y[0][:,519 + 235 +1,1]))
        Y_pred_sa1=np.concatenate((Y_pred_sa1,Yp[:,519 + 235 +1,1]))

        Y_true_sa2=np.concatenate((Y_true_sa2,Y[0][:,519 + 388 +1,1]))
        Y_pred_sa2=np.concatenate((Y_pred_sa2,Yp[:,519 + 388 +1,1]))

    print("\nA35SS IO SA1: " + str(scipy.stats.pearsonr(Y_true_sa1,Y_pred_sa1)[0]**2) )
    print("\nA35SS IO SA2: " + str(scipy.stats.pearsonr(Y_true_sa2,Y_pred_sa2)[0]**2) )

    

    
    ###########################
    ###########################
    ###########################
    ###########################
    ###########################
    ###########################
    ###########################
    ###########################
    ###########################
    ###########################
    
        


        
    ###########################  
    
    print()
    print("Learning rate: %.6f" % (kb.get_value(model_m.optimizer.lr)))
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    
    
    model.save('NewOutputs/inside_out/'+str(file_name)+'_'+str(epoch_num)+'.h5')
    
    
    if (epoch_num+1) >= 10:
            kb.set_value(model_m.optimizer.lr,
                         0.6*kb.get_value(model_m.optimizer.lr))
        # Learning rate decay
  


a5ss.close()
a3ss.close()

h5f.close()
###############################################################################
