#!/usr/bin/env python
# coding: utf-8
##################
# king@ 2018.04.05
# HDF5 data layer

import caffe

from pylab import *
from caffe import layers as L

def net(hdf5, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    n.ip1 = L.InnerProduct(n.data, num_output=50, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=2, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    n.accu = L.Accuracy(n.ip2, n.label)
    return n.to_proto()

with open('auto_train01.prototxt', 'w') as f:
    f.write(str(net('train.h5list', 100)))
with open('auto_test01.prototxt', 'w') as f:
    f.write(str(net('test.h5list', 50)))


solver = caffe.SGDSolver('auto_solver01.prototxt')

niter = 1001
test_interval = 10
train_loss = zeros(niter)
test_acc = zeros(niter)
train_acc = zeros(niter)

# iteration and save the loss, accuracy
for it in range(niter):
    solver.step(1)  
    train_loss[it] = solver.net.blobs['loss'].data
    train_acc[it] = solver.net.blobs['accu'].data
    test_acc[it] = solver.test_nets[0].blobs['accu'].data

#output graph
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(arange(niter), test_acc, 'r', arange(niter), train_acc, 'm')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
_.savefig('converge01.png')


