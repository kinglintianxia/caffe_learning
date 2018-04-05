#!/usr/bin/env python3
# coding: utf-8

######################
# king@2018.04.04 
# Dummy data for convnets

import caffe
from caffe import layers as L 
# Define net
def net():
	n = caffe.NetSpec()
	n.data = L.DummyData(dummy_data_param=dict(num=10, channels=1, height=28, width=28, data_filler=dict(type='gaussian')))
	n.label = L.DummyData(dummy_data_param=dict(num=10,channels=1, height=1, width=1, data_filler=dict(type='gaussian')))
	n.ip1 = L.InnerProduct(n.data,num_output=50,weight_filler=dict(type=('xavier')))
	n.relu1 = L.ReLU(n.ip1,in_place=True)
	n.ip2 = L.InnerProduct(n.relu1, num_output=4,weight_filler=dict(type=('xavier')))
	n.loss = L.SoftmaxWithLoss(n.ip2,n.label)

	return n.to_proto()

# Write to proto file 
with open('dummy_data.prototxt', 'w') as f:
	f.write(str(net()))

# Load solver
solver = caffe.SGDSolver('dummy_data_slover.prototxt')

# 
# solver.net.forward()
# solver.step(1)
# solver.solve()

print'data.shape: ', solver.net.blobs['data'].data.shape
print'label.shape: ', solver.net.blobs['label'].data.shape
