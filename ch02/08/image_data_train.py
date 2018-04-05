#!/usr/bin/env python3
# coding: utf-8
##################
# king@ 2018.04.04
# Brif: iamge data train

import caffe
from pylab import *
from caffe import layers as L 
# Define net
def net(image_list, batch_size, mean_value=0):
	n = caffe.NetSpec()																							# normalize
	n.data, n.label = L.ImageData(source=image_list, batch_size=batch_size, new_height=28, new_width=28, ntop=2, transform_param=dict(scale=1.0/255.0))
	n.ip1 = L.InnerProduct(n.data, num_output=50, weight_filler=dict(type='xavier'))
	n.relu1 = L.ReLU(n.ip1, in_place=True)
	n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
	n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
	return n.to_proto()

# Write proto file
with open('image_data_train.prototxt','w') as f:
	f.write(str(net('train00.imglist',200)))
with open('image_data_test.prototxt','w') as f:
	f.write(str(net('test00.imglist',50)))

# solver
# solver = caffe.SGDSolver('image_data_solver.prototxt')
solver = caffe.get_solver('image_data_solver.prototxt')

# solve
solver.solve()
# for i in range(100):
	# solver.step(1)