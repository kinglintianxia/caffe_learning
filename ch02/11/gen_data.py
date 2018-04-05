#!/usr/bin/env python
# coding: utf-8
##################
# king@ 2018.04.05
# HDF5 data layer

import h5py
import numpy as np


N = 10000
t = np.linspace(0, 2*np.pi, N)
x1 = np.array([t, 30+np.cos(t)+0.3*np.random.rand(N)])
x2 = np.array([t, 29+np.cos(t)+0.3*np.random.rand(N)])
y1 = np.zeros((N,1))
y2 = np.ones((N,1))
X = np.concatenate((x1.T, x2.T)).astype('float32')
y = np.concatenate((y1, y2)).astype('float32')
#X = np.concatenate((x1.T, x2.T))
#y = np.concatenate((y1, y2))

for i in range(2):
    X[:,i] = (X[:,i]-np.min(X[:,i]))/(np.max(X[:,i])-np.min(X[:,i]))

idx = np.arange(len(y))
np.random.shuffle(idx)

filenum = 10
filelen = len(y)/10

for i in range(filenum):
    with h5py.File('train'+str(i)+'.h5', 'w') as f:
        f.create_dataset('data',  data=X[idx[i*filelen:(i+1)*filelen],:])
        f.create_dataset('label', data=y[idx[i*filelen:(i+1)*filelen],:], dtype="i")

filelist = range(filenum)
# train list 4 of 5 parts
with open('train.h5list','w') as f:
    for i in filelist[0:filenum*4/5]:
        f.write('train'+str(i)+'.h5\n')
# test list 1 of 5 parts
with open('test.h5list','w') as f:
    for i in filelist[filenum*4/5:]:
        f.write('train'+str(i)+'.h5\n')
