#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#############################################################################################
#Experiment code from paper pulished at ICANN19
#Title: Improving Deep Image Clustering With Spatial Transformer Layers
#Authors: Thiago V. M. de Souza[1] and  Cleber Zanchettin[2]
#Institute: [1,2] Universidade Federal de Pernambuco - Centro de Informática, Recife, Brazil
#Email: [1]tvms@cin.ufpe.br; [2]cz@cin.ufpe.br
#
#Experiment: ST-DAC + 1ST-Layers
#Test Dataset: MNIST
#
#Code Published at: 
#https://github.com/tvmsouza/ST-DAC
#
#Third Party and reference github repositories:
#https://github.com/EderSantana/seya
#https://github.com/vector-1127/DAC
#############################################################################################

from __future__ import print_function
import os,sys

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='device=cuda0,gpuarray.preallocate=0.2,mode=FAST_RUN,floatX=float32,optimizer=fast_compile,dnn.enabled=True,exception_verbosity=high'


import scipy
import numpy as np
import h5py
from keras.datasets import mnist,cifar10, cifar100
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Dropout, merge, Lambda, Reshape, Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization, AveragePooling2D, Highway
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.engine.topology import Layer
import matplotlib.pyplot as plt
import scipy.io as sio
import json
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
from sys import path
from seya.layers.attention import SpatialTransformer
path.append('../../DAC')
from myMetrics import *
import cv2

import pandas as pd
#np.random.seed(1337)  # for reproducibility


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

global upper, lower
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    channel_shift_range=0.05,
    rescale=0.975,
    zoom_range=[0.95,1.05]
)

class Adaptive(Layer):
    def __init__(self, **kwargs):
        super(Adaptive, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.nb_sample = input_shape[0]
        self.nb_dim = input_shape[1]
        
    def call(self, x, mask = None):
        y = self.transfer(x)
        return y
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],input_shape[1])
        
    def transfer(self, x):
        y = K.pow(K.sum(x**2, axis = 1), 0.5)
        y = K.expand_dims(y, dim = 1)
        y = K.repeat_elements(y, self.nb_dim, axis = 1)
        return x/y

class DotDist(Layer):
    def __init__(self, **kwargs):
        super(DotDist, self).__init__(**kwargs)
        
    def call(self, x, mask = None):
        d = K.dot(x, K.transpose(x))
        return d
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],input_shape[0])

def myLoss(y_true, y_pred):

    mp_pos = np.array(np.logical_or(np.equal(y_pred,1),np.equal(y_pred,0)))
    loss_pos = ((-y_true[mp_pos]*np.log(y_pred[mp_pos])-(1-y_true[mp_pos])*np.log(1-y_pred[mp_pos])))
    loss = K.mean(loss_pos)
    return loss

#data
nb_classes = 10
img_channels, img_rows, img_cols = 3, 32, 32

(X_train, y_true), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.
X_train = np.vstack((X_train,X_test))
y_true = np.hstack((np.squeeze(y_true),np.squeeze(y_test)))

y_true = y_true.astype('int64')

X_train_small = []
y_true_small = []

size_dataset = 1.0

for i in range(0,10):
	(y_true_idx) = np.nonzero(y_true == i)
	y_true_idx = y_true_idx[0]
	y_true_count = np.count_nonzero(y_true == i)
	y_true_count = round(size_dataset*y_true_count)

	y_true_idx = y_true_idx[0:y_true_count]
	if i == 0:
		X_train_small = X_train[y_true_idx]
		y_true_small = y_true[y_true_idx]
	else:
		X_train_small = np.vstack((X_train_small,X_train[y_true_idx]))
		y_true_small = np.hstack((y_true_small,y_true[y_true_idx]))

X_train = X_train_small
y_true = y_true_small

mean_image0 = np.mean(X_train)
X_train -= mean_image0

print(X_train.shape)

index = np.arange(X_train.shape[0])
np.random.shuffle(index)
X_train = X_train[index]
y_true = y_true[index]

tempmap = np.copy(index)
#parameters
batch_size = 32
epoch = 10
nb_epoch = 1
upper = 0.99
mid = 0.8
lower = 0.8
th = upper
eta = (upper-lower)/epoch
etau = (upper-mid)/epoch
etal = (mid-lower)/epoch
nb = 1000
lr = 0.0001

inp_loc = Input(shape=(img_channels,img_rows, img_cols))
    

def locNet3():

    b3 = np.zeros((2, 3), dtype='float32')
    b3[0, 0] = 1.0
    b3[1, 1] = 1.0
    b3 = b3.flatten()

    W3 = np.zeros((50, 6), dtype='float32')
    weights3 = [W3, b3]

    
    inp3 = Input(shape=(1, 28, 28))
    x3 =(MaxPooling2D(pool_size=(2, 2))(inp3))
    x3 =(Convolution2D(20, 5, 5, border_mode='same')(x3))
    x3 =(Activation('tanh')(x3))
    x3 =(MaxPooling2D(pool_size=(2, 2))(x3))
    y3 =(Convolution2D(20, 5, 5, border_mode='same')(x3))
    y3 =(Activation('tanh')(y3))
    out5 =(Flatten()(y3))

    out3 = (Dense(50)(out5))
    out3 =(Activation('tanh')(out3))
    out3 = (Dense(6, weights=weights3)(out3))
    out3 =(Activation('tanh')(out3))

    return Model(inp3,out3)

def locNet2():

    b3 = np.zeros((2, 3), dtype='float32')
    b3[0, 0] = 1.0
    b3[1, 1] = 1.0
    b3 = b3.flatten()

    W3 = np.zeros((50, 6), dtype='float32')
    weights3 = [W3, b3]

    
    inp3 = Input(shape=(128, 14, 14))
    x3 =(MaxPooling2D(pool_size=(2, 2))(inp3))
    x3 =(Convolution2D(20, 5, 5, border_mode='same')(x3))
    x3 =(Activation('tanh')(x3))
    x3 =(MaxPooling2D(pool_size=(2, 2))(x3))
    y3 =(Convolution2D(20, 5, 5, border_mode='same')(x3))
    y3 =(Activation('tanh')(y3))
    out5 =(Flatten()(y3))

    out3 = (Dense(50)(out5))
    out3 =(Activation('tanh')(out3))
    out3 = (Dense(6, weights=weights3)(out3))
    out3 =(Activation('tanh')(out3))

    return Model(inp3,out3)


modelLoc = locNet3()
modelLoc2 = locNet2()


#model
inp_ = Input(shape=(1, 28, 28))
spt = SpatialTransformer(localization_net=modelLoc, downsample_factor=1, name='spatial')(inp_)
x = Convolution2D(64,3,3,init = 'he_normal',border_mode='same', input_shape=(1,28,28))(spt)
x = BatchNormalization(mode=2,axis = 1)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size = (2,2))(x)
x = BatchNormalization(mode=2,axis = 1)(x)
x = Convolution2D(128,3,3,init = 'he_normal',border_mode='same')(x)
x = BatchNormalization(mode=2,axis = 1)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size = (2,2))(x)
x = BatchNormalization(mode=2,axis = 1)(x)
x = Convolution2D(256, 3, 3,init = 'he_normal',border_mode='same')(x)
x = BatchNormalization(mode = 2,axis = 1)(x)
x = Activation('relu')(x)
x0 = Flatten()(x)
x1 = Dense(3096)(x0)
x1 = BatchNormalization(mode = 2)(x1)
x1 = Activation('relu')(x1)
x1 = Dropout(0.5)(x1)
x1 = Dense(nb_classes)(x1)
x1 = BatchNormalization(mode = 2)(x1)
x1 = Activation('relu')(x1)
y = Activation('softmax',name = 'aux_output')(x1)
z = Adaptive()(y) 
dist = DotDist(name = 'main_output')(z)

norm_l2 = Model(input=[inp_], output=[x0])
cluster_l1 = Model(input=[inp_], output=[y])
cluster_l2 = Model(input=[inp_], output=[z])
model = Model(input=[inp_], output=[y, dist])


cluster_l1_pred = K.function([inp_,K.learning_phase()], y)
cluster_l2_pred = K.function([inp_,K.learning_phase()], z)
model_pred = K.function([inp_,K.learning_phase()],K.learning_phase())
model.summary()

cluster_l1.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(0.0001), 
              loss={'main_output':'binary_crossentropy','aux_output':'binary_crossentropy'}, 
              loss_weights={'main_output':1,'aux_output':10})




location = str(sys.argv[0])
idx = str(sys.argv[1])
location = location.replace('.py','_'+idx+'.h5')
weight_path = location.replace('.h5','')
print(location)
print(weight_path)
if not os.path.exists(weight_path):
    os.makedirs(weight_path)


acc = []
ari = []
nmi = []
ind = []
output = []



Output = cluster_l1.predict(X_train)
y_pred = np.argmax(Output,axis = 1)
nmit = NMI(y_true,y_pred)
arit = ARI(y_true,y_pred)
acct, indt = ACC(y_true,y_pred)
model_weight_location = location.replace('.h5','/model_weight_epoch_{}.h5'.format(0))
cluster_l1.save_weights(model_weight_location)
print(X_train.shape)
print(nmit, arit, acct)
acc.append(acct)
ari.append(arit)
nmi.append(nmit)
ind.append(indt)
output.append(Output)

index = np.arange(X_train.shape[0])
index_loc = np.arange(nb)



for e in range(epoch):
	np.random.shuffle(index)
	if X_train.shape[0]>nb:
		for i in range(X_train.shape[0]//nb):
			Xbatch = X_train[index[np.arange(i*nb,(i+1)*nb)]]
			Y = cluster_l2.predict(Xbatch)
			Ybatch = (np.sign(np.dot(Y,Y.T)-th)+1)/2
			a = Ybatch > upper
			print(a.shape)
			count_one  = float(np.count_nonzero(Ybatch > upper))/2.0
			count_zero = float(np.count_nonzero(Ybatch < lower))/2.0
			base = count_one+count_zero
			print(count_zero)
			print(count_one)
			count_one = float(round(count_zero/count_one))
			print(count_one)
			for k in range(nb_epoch):
				np.random.shuffle(index_loc)
				for j in range(Xbatch.shape[0]//batch_size):
					address = index_loc[np.arange(j*batch_size,(j+1)*batch_size)]
					X_batch = Xbatch[address]
					Y_batch = Ybatch[address,:][:,address]
					Y_ = Y[address]
					sign = 0
					for X_batch_i in datagen.flow(X_batch, batch_size=batch_size,shuffle=False):
						loss = model.train_on_batch([X_batch_i],[Y_, Y_batch])
						sign += 1
						if sign>0:
							break
			print('Epoch: %d, batch: %d/%d, loss: %f, loss1: %f, loss2: %f, nb1: %f'%
				(e+1,i+1,X_train.shape[0]//nb,loss[0],loss[1],loss[2],np.mean(Ybatch)))
	else:
		print('error')
		
	
	subModel = Model(input=model.layers[1].input, output=model.layers[1].output)

	if ((e%2)==0) :
		upper = upper - etau
	else:
		lower = lower + etal
	th = th - eta
	
	
	predictions = subModel.predict(X_train)
	
	for i in range(0,10):
		x_train_zeros = predictions[y_true == i]
		print(x_train_zeros.shape)
		mean_image0 = np.mean(x_train_zeros,axis=0)
		var_image0 = np.var(x_train_zeros,axis=0)
		std_image0 = np.std(x_train_zeros,axis=0)
		print(mean_image0.shape)
		outputl = 'mean_'+str(e)+str(i)+'.png'
		scipy.misc.imsave(outputl, mean_image0.reshape(28,28))
		outputl = 'std_'+str(e)+str(i)+'.png'
		scipy.misc.imsave(outputl, std_image0.reshape(28,28))
		outputl = 'var_'+str(e)+str(i)+'.png'
		scipy.misc.imsave(outputl, var_image0.reshape(28,28))
	
	Output = cluster_l1.predict(X_train)
	y_pred = np.argmax(Output,axis = 1)
	nmit = NMI(y_true,y_pred)
	arit = ARI(y_true,y_pred)
	acct, indt = ACC(y_true,y_pred)
	model_weight_location = location.replace('.h5','/model_weight_epoch_{}.h5'.format(e+1))
	cluster_l1.save_weights(model_weight_location)
	print(nmit, arit, acct)
	acc.append(acct)
	ari.append(arit)
	nmi.append(nmit)
	ind.append(indt)
	output.append(Output)
	
	if os.path.exists(location):
		os.remove(location)
	file = h5py.File(location,'w')
	file.create_dataset('acc',data = acc)
	file.create_dataset('nmi',data = nmi)
	file.create_dataset('ari',data = ari)
	file.create_dataset('ind',data = ind)
	file.create_dataset('output',data = output)
	file.create_dataset('tempmap',data = tempmap)
	file.close()
	









