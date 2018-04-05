#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001


import caffe

from pylab import *
from caffe import layers as L

imgdata_mean = 108

def net(img_list, batch_size, mean_value=0):
    n = caffe.NetSpec()
    n.data, n.label=L.ImageData(source=img_list,batch_size=batch_size,new_width=28,new_height=28,ntop=2,transform_param=dict(scale = 1/255.0, mean_value=mean_value))
    n.ip1 = L.InnerProduct(n.data, num_output=50, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()

with open('auto_train00.prototxt', 'w') as f:
    f.write(str(net('train00.imglist', 200, imgdata_mean)))
with open('auto_test00.prototxt', 'w') as f:
    f.write(str(net('test00.imglist', 50, imgdata_mean)))


solver = caffe.SGDSolver('auto_solver00_step.prototxt')
niter = 3000
test_interval = 100
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter * 1.0 / test_interval)))
print len(test_acc)


# The main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='data')


    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        data = solver.test_nets[0].blobs['ip2'].data
        label = solver.test_nets[0].blobs['label'].data
        for test_it in range(60):
            solver.test_nets[0].forward()
            
            for i in range(len(data)):
                    if np.argmax(data[i]) == label[i]:
                        correct += 1

        test_acc[int(it / test_interval)] = correct * 1.0 / (len(data)  * 100)

# output graph
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
_.savefig('converge00.png')

