#!/usr/bin/env python
# coding: utf-8
#####################
# king@ 2018.04.11
# conv layer

import caffe
from caffe import layers as L 
from caffe import params as P 
import caffe.draw
from caffe.proto import caffe_pb2
from google.protobuf import text_format

# paths
path = ''
solver_file = path + 'solver.prototxt'
net_file = path + 'conv_pool_py.prototxt'

# conv net
def conv_pool_net():
	n = caffe.NetSpec()
	n.data = L.DummyData(dummy_data_param=dict(num=20,channels=1,height=64,width=64,data_filler=dict(type="gaussian")))
	n.label = L.DummyData(dummy_data_param=dict(num=20,channels=10,height=1,width=1,data_filler=dict(type='gaussian')))
	n.conv1 = L.Convolution(n.data,num_output=20,kernel_size=4,stride=3,pad=0)	# (64-4)/3+1=21
	n.relu1 = L.ReLU(n.conv1,in_place=True)	# 21x21
	n.pool1 = L.Pooling(n.relu1,pool=P.Pooling.MAX,kernel_size=2,stride=2)	# ceil((21-2)/2+1)=11
	# iter conv net
	for i in range(2):
		# conv_name = 'conv' + str(i+2)
		# relu_name = 'relu' + str(i+2)
		# pool_name = 'pool' + str(i+2)
		n.conv1 = L.Convolution(n.pool1,num_output=10,kernel_size=4,stride=2,pad=3)	# int((11+3*2-4)/2+1)=7
		n.relu1 = L.ReLU(n.conv1,in_place=True)
		n.pool1 = L.Pooling(n.relu1,pool=P.Pooling.MAX,kernel_size=2,stride=2)

	n.ip2 = L.InnerProduct(n.pool1,num_output=10,weight_filler=dict(type='xavier'))
	n.loss = L.SigmoidCrossEntropyLoss(n.ip2,n.label) # SigmoidCrossEntropyLoss
	return n.to_proto()

# solver
def gen_solver(solver_file,net_file,test_net_file=None):
	s = caffe_pb2.SolverParameter()
	s.train_net = net_file
	if not test_net_file:
		s.test_net.append(net_file)
	else:
		s.test_net.append(test_net_file)
	# parameters
	s.test_interval = 500	# test net for every 500 train
	s.test_iter.append(100)	# 测试迭代次数，假设测试数据有8000个，那batch size=80
	s.max_iter = 20000	# max iteration time
	
	s.base_lr = 0.001	# learning rate
	s.momentum = 0.9
	s.weight_decay = 5e-4 # 正则化权值衰减因子，防止过拟合

	s.lr_policy = 'step' # 学习率衰减方法
	s.stepsize = 1000	# 只对step方法有效， base_lr*gamma^floor(iter/stepsize)
	s.gamma = 0.1

	s.display = 500         # 输出日志间隔迭代次数
	s.snapshot = 10000      # 在指定迭代次数时保存模型
	s.snapshot_prefix = 'shapshot'
	s.type = 'SGD'  # 迭代算法类型, ADADELTA, ADAM, ADAGRAD, RMSPROP, NESTEROV
	s.solver_mode = caffe_pb2.SolverParameter.GPU

	# Save solver file
	with open(solver_file, 'w') as f:
		f.write(str(s))

	solver = caffe.SGDSolver(solver_file)
	return solver 

def print_net_shape(net):
    print "------data and diff output shape------"
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + ' out \t' + str(blob.data.shape)
        print layer_name + ' diff\t' + str(blob.diff.shape)

    print "------weight and bias shape------"
    for layer_name, param in net.params.iteritems():
        print layer_name + ' weight\t' + str(param[0].data.shape), str(param[1].data.shape)
        print layer_name + ' diff  \t' + str(param[0].diff.shape), str(param[1].diff.shape)

def draw_net(net_file, jpg_file):
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(net_file).read(), net)
    caffe.draw.draw_net_to_file(net, jpg_file, 'LR')



with open( net_file, 'w') as f:
    f.write(str(conv_pool_net()))

solver = gen_solver(solver_file, net_file)
print_net_shape(solver.net)
draw_net(net_file, "a.png")
