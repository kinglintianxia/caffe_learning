#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001

#import os, sys
#CAFFE_HOME = '.../caffe/'
#sys.path.insert(0, CAFFE_HOME + 'caffe/python')

import caffe

from pylab import *
from caffe import layers as L
from caffe import params as P

def net(dbfile, batch_size, mean_value=0):
    n = caffe.NetSpec()
    n.data, n.label=L.Data(source=dbfile, backend = P.Data.LMDB, batch_size=batch_size, ntop=2, transform_param=dict(scale=0.00390625))
    n.ip1 = L.InnerProduct(n.data, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    n.accu = L.Accuracy(n.ip2, n.label, include={'phase':caffe.TEST})
    return n.to_proto()

with open( 'auto_train00.prototxt', 'w') as f:
    f.write(str(net( '/home/king/Documents/caffe/examples/mnist/mnist_train_lmdb', 64)))
with open('auto_test00.prototxt', 'w') as f:
    f.write(str(net('/home/king/Documents/caffe/examples/mnist/mnist_test_lmdb', 100)))


solver = caffe.SGDSolver('auto_mnist_solver.prototxt')
solver.net.forward()
solver.test_nets[0].forward()

#solver.step(1)
solver.solve()


