#!/usr/bin/env python
# coding: utf-8
##################
# king@2018.04.07
# lmdb data multilabel

import caffe

from pylab import *
from caffe import layers as L
from caffe import params as P
imgdata_mean = 20

def net(datafile, labelfile, batch_size, mean_value=0):
    n = caffe.NetSpec()
    n.data=L.Data(source=datafile, backend = P.Data.LMDB, batch_size=batch_size, ntop=1, transform_param=dict(scale=1.0/30.0, mean_value=mean_value))
    n.label=L.Data(source=labelfile, backend = P.Data.LMDB, batch_size=batch_size, ntop=1)
    n.ip1 = L.InnerProduct(n.data, num_output=50, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SigmoidCrossEntropyLoss(n.ip2, n.label)
    return n.to_proto()

with open( 'auto_train00.prototxt', 'w') as f:
    f.write(str(net('lmdb_train_data',  'lmdb_train_label', 200, imgdata_mean)))
with open( 'auto_test00.prototxt', 'w') as f:
    f.write(str(net( 'lmdb_test_data',  'lmdb_test_label', 50, imgdata_mean)))


solver = caffe.SGDSolver( 'auto_solver00.prototxt')



niter = 1001
test_interval = 100
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter * 1.0 / test_interval)))
print len(test_acc)


# The main solver loop
for it in range(niter):     # 1000 step
    solver.step(1)  # SGD by Caffe
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='data')
    # calu test accu
    if it % test_interval == 0:     
        correct = 0
        # 平均20次
        for test_it in range(20):
            solver.test_nets[0].forward()
            data = solver.test_nets[0].blobs['ip2'].data
            label = solver.test_nets[0].blobs['label'].data
            label.shape = data.shape
            # Positive values map to label 1, while negative values map to label 0
            # data.shape:(50, 10)
            for i in range(len(data)): # 50 batch_size
                for j in range(len(data[i])): # 10  label_size
                    # sigmoid, x>0, label=1
                    if data[i][j] > 0 and label[i][j] == 1: 
                        correct += 1.0
                    # sigmoid, x<0, label=0
                    elif data[i][j] <= 0 and label[i][j] == 0:
                        correct += 1.0
        # test_acc.shape = 11 = [(0~99),(100~199),...]
        test_acc[int(it / test_interval)] = correct * 1.0 / (size(data)  * 20)  # size(data)=500
        print 'Iteration', it, 'testing accuracy is ', str(correct * 1.0 / (size(data)  * 20))


# output graph
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
_.savefig('converge00.png')


