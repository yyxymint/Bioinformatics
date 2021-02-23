###############################################################################
# This file has the functions necessary to create the SpliceAI model.
###############################################################################

from keras.models import Model
from tensorflow import keras
from keras.layers import Input,Cropping1D,Conv1D,BatchNormalization,Activation,add
import keras.backend as kb
import numpy as np
import tensorflow as tf

def ResidualUnit(l, w, ar):
    # Residual unit proposed in "Identity mappings in Deep Residual Networks"
    # by He et al.

    def f(input_node):
        bn1 = BatchNormalization()(input_node)
        act1 = Activation('relu')(bn1)
        conv1 = Conv1D(l, w, dilation_rate=ar, padding='same')(act1)
        bn2 = BatchNormalization()(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv1D(l, w, dilation_rate=ar, padding='same')(act2)
        output_node = add([conv2, input_node])

        return output_node

    return f


def SpliceAI(L, W, AR):
    # L: Number of convolution kernels
    # W: Convolution window size in each residual unit
    # AR: Atrous rate in each residual unit

    assert len(W) == len(AR)
    
    
    
    CL = 2 * np.sum(AR*(W-1))

    input0 = keras.Input(shape=(None, 4))
    conv = Conv1D(L, 1)(input0)
    skip = Conv1D(L, 1)(conv)

    for i in range(len(W)):
    
    
        conv = ResidualUnit(L, int(W[i]), int(AR[i]))(conv)
#         conv = ResidualUnit(L, 1, 1)(conv)
       
        if (((i+1) % 4 == 0) or ((i+1) == len(W))):
            # Skip connections to the output after every 4 residual units
            dense = Conv1D(L, 1)(conv)
            skip = add([skip, dense])

    
    
    skip = Cropping1D(int(CL//2))(skip)
    

    output0 = [[] for t in range(1)]

    for t in range(1):
        output0[t] = Conv1D(3, 1, activation='softmax')(skip)
    
    model = Model(inputs=input0, outputs=output0)

    return model


def categorical_crossentropy_2d(y_true, y_pred):
    # Standard categorical cross entropy for sequence outputs
    
    yt1=y_true[:, :, 0]
    yt2=y_true[:, :, 1]
    yt3=y_true[:, :, 2]
    
    yt1=tf.cast(yt1, tf.float32)
    yt2=tf.cast(yt2, tf.float32)
    yt3=tf.cast(yt3, tf.float32)
    
    return - kb.mean(yt1*kb.log(y_pred[:, :, 0]+1e-10)
                   + yt2*kb.log(y_pred[:, :, 1]+1e-10)
                   + yt3*kb.log(y_pred[:, :, 2]+1e-10),axis=None)
