#!/usr/bin/env python
# coding: utf-8
##################
# king@ 2018.04.05
# HDF5 data layer

import caffe
from caffe import layers as L 
import numpy as np
import matplotlib.pyplot as plt 

def net(hdf5_list, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5_list, ntop=2)   # ntop: two param returns
    n.ip1 = L.InnerProduct(n.data, num_output=50, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=2, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    n.accu = L.Accuracy(n.ip2, n.label)
    return n.to_proto()


# prototxt
with open('hdf5_train.prototxt','w') as f:
    f.write(str(net('train.h5list', 100)))
with open('hdf5_test.prototxt','w') as f:
    f.write(str(net('test.h5list', 50)))

# solver
solver = caffe.SGDSolver('hdf5_solver.prototxt')

# plot
niter = 1000
train_loss = np.zeros(niter)
train_acc = np.zeros(niter)
test_acc = np.zeros(niter)

# step solve
for i in range(niter):
    solver.step(1)
    train_loss[i] = solver.net.blobs['loss'].data 
    train_acc[i] = solver.net.blobs['accu'].data
    test_acc[i] = solver.test_nets[0].blobs['accu'].data 

ax1 = plt.subplot(1,1,1)
ax2 = ax1.twinx()
ax1.plot(np.arange(niter), train_loss)
ax2.plot(np.arange(niter), test_acc, 'r', np.arange(niter), train_acc, 'm')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
plt.savefig('loss&accu.png')

