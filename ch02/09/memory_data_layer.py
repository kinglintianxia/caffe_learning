#!/usr/bin/env python
# coding: utf-8
#########################
# king@ 2018.04.05
# MemoryDataLayer

import caffe
import numpy as np
from pylab import *

solver = caffe.SGDSolver('hbk_mnist_solver.prototxt')
# solver.net.forward()

N = 1000
t = np.linspace(0, 2*np.pi, N)
x1 = np.array([t, 30+np.cos(t)])
x2 = np.array([t, 29+np.cos(t)])
y1 = np.zeros((N,1))
y2 = np.ones((N,1))
X = np.concatenate((x1.T, x2.T)).astype('float32')
y = np.concatenate((y1, y2)).astype('float32')


idx = np.arange(len(y))

# float32, C_CONTIGUOUS

# input graph
_, ax1 = subplots()
ax1.scatter(X[0:N,0],X[0:N,1],c='r')
ax1.scatter( X[N:,0],X[N:,1], c='b')
ax1.set_xlabel('t')
ax1.set_ylabel('x')
_.savefig('input.png')

X_train = X[idx,:].reshape(X.shape[0],2,1,1)
y_train = y[idx].reshape(y.shape[0],1,1,1)
# C_CONTIGUOUS
X_train = np.require(X_train,requirements='C')
# 
solver.net.set_input_arrays(X_train, y_train)
solver.test_nets[0].set_input_arrays(X_train, y_train)

for i in range(3001):
    solver.step(1)


""" 
# homework
x_min, x_max = np.min(X[:,0])-0.5, np.max(X[:,0])+0.5
y_min, y_max = np.min(X[:,1])-0.5, np.max(X[:,1])+0.5


xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
	                      np.arange(y_min, y_max, 0.02))
Z = np.zeros(xx.shape)
"""
# xx，yy当作模型的输入，通过forward过程得到模型的输出
# 输出赋值给Z。
# 最后用ax1.conourf(xx,yy,Z)
