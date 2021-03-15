# -*- coding: utf-8 -*-
"""
@author: Purnendu Mishra
"""
# Tensorflow library import

from   tensorflow.keras.models import Model
from   tensorflow.keras.layers import Conv2D, Input, BatchNormalization, UpSampling2D, Concatenate
from   tensorflow.keras.layers import Activation, Add, Conv2DTranspose, Reshape, MaxPooling2D

from   tensorflow.keras.initializers import RandomNormal
from   tensorflow.keras.regularizers import l2

from   CustomLayer import SpatialSoftmax, SoftArgMaxConv

#%%

def IqbalNet(input_shape = (None, None, 3), n_keys = 21, mode = 'latent'):
    
    filters = 256
    
    inputs = Input(shape = input_shape)
    
    # Front
    f = front(filters = filters)(inputs)    # 128 x 128
    
    # down sample
    d1 = down_sample(filters = filters)(f)  # 64 x 64
    d2 = down_sample(filters = filters)(d1) # 32 x 32
    d3 = down_sample(filters = filters)(d2) # 16 x 16
    d4 = down_sample(filters = filters)(d3) # 8 x 8
    d5 = down_sample(filters = filters)(d4) # 4 x 4 
    d6 = down_sample(filters = filters)(d5) # 2 x 2
    
    # up sample
    u1 = conv_bn_relu(filters = filters)(d6)
    u1 = UpSampling2D(size=(2,2), interpolation='nearest')(u1) # 4 x 4 
    u1 = Concatenate()([u1, d5]) # 4 x 4
    
    u2 = up_sample(filters = filters)(u1)
    u2 = Concatenate()([u2,d4]) # 8 x 8 
    
    u3 = up_sample(filters = filters)(u2)  # 16 x 16
    u3 = Concatenate()([u3, d3])

    u4 = up_sample(filters = filters)(u3)  # 32 X 32
    u4 = Concatenate()([u4,d2])    
    
    u5 = up_sample(filters = filters)(u4)  # 64 X 64
    u5 = Concatenate()([u5,d1])   

    u6 = up_sample(filters = filters)(u5) 

    b  = back(filters = filters)(u6) 
    
    out = Conv2D(filters         =  n_keys, 
                     kernel_size = (7,7), 
                     padding     = 'same',
                     activation  = 'linear',
                     name        = 'heatmaps')(b)   
    
    if mode == 'heatmap':
        return Model(inputs = inputs, outputs = out, name ='iqbal_net')
    
    elif mode == 'latent':
        softmax = SpatialSoftmax(name = 'spatial_softmax')(out)
        argmax  = SoftArgMaxConv(name = 'soft_arg_max')(softmax)
        keys    = Reshape((n_keys * 2,), name = 'output')(argmax)
        
        return Model(inputs = inputs, outputs = keys, name ='iqbal_net')
        
        
    

def back(**params):
    filters       = params['filters']
    def f(x):
        x = conv_bn_relu(filters     = filters,
                         kernel_size = (7,7))(x)
        x = conv_bn_relu(filters     = filters,
                         kernel_size = (7,7))(x)
        
        return x
    return f


def up_sample(**params):
    filters       = params['filters']
    def f(x):
        x = conv_bn_relu(filters      = filters,
                          kernel_size = (1,1),
                          padding     = 'valid')(x)
        
        x  = conv_bn_relu(filters = filters)(x)
        
        x  = UpSampling2D(size=(2,2), interpolation='nearest')(x)
        return x
    return f
                          

def down_sample(**params):
    filters       = params['filters']
    
    def f(x):
        x = conv_bn_relu(filters = filters)(x)
        x = MaxPooling2D(pool_size = (2,2), 
                         strides   = (2,2), 
                         padding   ='same')(x)
        x = conv_bn_relu(filters = filters)(x)
        return x
    return f


def front(**params):
    filters       = params['filters']
    def f(x):
        x = conv_bn_relu(filters = filters)(x)
        x = MaxPooling2D(pool_size = (2,2), strides = (1,1), name = 'pool_1', padding='same')(x)
        
        x = conv_bn_relu(filters = filters)(x)
        x = MaxPooling2D(pool_size = (2,2), strides = (1,1), name = 'pool_2', padding='same')(x)
        return x
    return f



def conv_bn_relu(**params):
    filters       = params['filters']
    kernel_size   = params.setdefault('kernel_size', (3,3))
    dilation_rate = params.setdefault('dilation_rate', (1,1))
    strides       = params.setdefault('strides', (1,1))
    padding       = params.setdefault('padding','same')
    kernel_init   = params.setdefault('kernel_initializer', RandomNormal(stddev = 0.001))
    kernel_reg    = params.setdefault('kernel_regularizer', l2(0.001))
    

    def f(x):
        x       =  Conv2D(filters, 
                          kernel_size,
                          strides = strides, 
                          padding = padding,
                          kernel_initializer  = kernel_init,
                          kernel_regularizer  = kernel_reg,
                          dilation_rate       = dilation_rate
                          )(x)
        
        x       = BatchNormalization(axis = -1, fused=True)(x)
        
        return Activation('relu')(x)
    
    return f

if __name__=='__main__':
    model = IqbalNet(input_shape = (128,128, 3), n_keys = 21, mode = 'heatmap')
    model.summary()
