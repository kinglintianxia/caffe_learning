#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001

import caffe

from pylab import *
from caffe import layers as L
from caffe import params as P

data_path = "/home/hbk/caffe/examples/cifar10/"
train_net_file = 'auto_train00.prototxt'
test_net_file = 'auto_test00.prototxt'
solver_file = "auto_solver.prototxt"



def net(datafile, mean_file, batch_size):
    n = caffe.NetSpec()
    n.data,n.label=L.Data(source=datafile, backend = P.Data.LMDB, batch_size=batch_size, ntop=2, transform_param=dict(scale=1.0/255.0, mean_file=mean_file))
    n.ip1 = L.InnerProduct(n.data, num_output=200, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    n.accu = L.Accuracy(n.ip2, n.label, include={'phase':caffe.TEST})
    return n.to_proto()

### net file generate #####

with open( train_net_file, 'w') as f:
    f.write(str(net(data_path+'cifar10_train_lmdb',  data_path+'mean.binaryproto', 200)))
with open( test_net_file, 'w') as f:
    f.write(str(net(data_path+'cifar10_test_lmdb',  data_path+'mean.binaryproto', 100)))


### solver file generate ######
from caffe.proto import caffe_pb2
s = caffe_pb2.SolverParameter()

s.train_net = train_net_file
s.test_net.append(test_net_file)
s.test_interval = 500  
s.test_iter.append(100) 
s.display = 500
s.max_iter = 10000     
s.weight_decay = 0.005
s.base_lr = 0.1
s.lr_policy = "step"
s.gamma = 0.1
s.stepsize = 5000
s.solver_mode = caffe_pb2.SolverParameter.GPU

with open(solver_file, 'w') as f:
    f.write(str(s))


### iter to calculate the models weight #####
solver = caffe.get_solver(solver_file)


niter = 2001
train_loss = zeros(niter)
test_acc = zeros(niter)


# The main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    train_loss[it] = solver.net.blobs['loss'].data
    test_acc[it] = solver.test_nets[0].blobs['accu'].data


#output graph
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(arange(niter), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
_.savefig('converge01.png')
