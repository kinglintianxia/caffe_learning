#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001

import caffe, h5py

from pylab import *
from caffe import layers as L

def net(hdf5, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    n.ip1 = L.InnerProduct(n.data, num_output=50, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=4, weight_filler=dict(type='xavier'))
    n.loss = L.SigmoidCrossEntropyLoss(n.ip2, n.label)  # SigmoidCrossEntropyLoss
    return n.to_proto()

with open('auto_train00.prototxt', 'w') as f:
    f.write(str(net('train.h5list', 100)))
with open('auto_test00.prototxt', 'w') as f:
    f.write(str(net('test.h5list', 50)))


solver = caffe.SGDSolver('auto_solver00.prototxt')



niter = 200
test_interval = 10
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter * 1.0 / test_interval))) # ceil(): Return the ceiling of the input, element-wise.
print len(test_acc)


# The main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='data')
    # calu test accu
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        data = solver.test_nets[0].blobs['ip2'].data
        label = solver.test_nets[0].blobs['label'].data
        # 平均100次
        for test_it in range(100):
            solver.test_nets[0].forward()
            # Positive -> label 1, negative -> label 0
            for i in range(len(data)):  # bath_size
                for j in range(len(data[i])):   # label_size
                    # softmax classify
                    if data[i][j] > 0 and label[i][j] == 1:
                        correct += 1
                    elif data[i][j] <= 0 and label[i][j] == 0:
                        correct += 1
        test_acc[int(it / test_interval)] = correct * 1.0 / (len(data) * len(data[0]) * 100)

# output graph
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
_.savefig('converge00.png')


