#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001


CLASS_NUM = 5
caffe_root = "/home/hbk/caffe/"
import sys
sys.path.insert(0, caffe_root + 'python')


import tempfile
import numpy as np
from pylab import *
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

CLASS_NUM = 5
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2

# =========================== Generate Model =========================

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1, param=learned_param):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group, param=param)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=None):
    fc = L.InnerProduct(bottom, num_output=nout, param=param)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(batch_size=256, include_acc=False, train = True, learn_all=True):

    subset = 'train' if train else 'test'
    source = caffe_root + 'data/flickr_style/%s.txt' % subset
    transform_param = dict(mirror=train, crop_size=227,
        mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    
    n=caffe.NetSpec()
    n.style_data, n.style_label = L.ImageData(
        transform_param=transform_param, source=source,
        batch_size=50, new_height=256, new_width=256, ntop=2)

    param = learned_param if learn_all else frozen_param

    # the net itself
    n.conv1, n.relu1 = conv_relu(n.style_data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6, 4096, param=param)
    n.drop7 = L.Dropout(n.relu7, in_place=True)
    n.fc8_style = L.InnerProduct(n.drop7, num_output=CLASS_NUM, param=learned_param)
    n.loss = L.SoftmaxWithLoss(n.fc8_style, n.style_label)

    n.acc = L.Accuracy(n.fc8_style, n.style_label)
    return n.to_proto()

def make_net(learn_all=True):
    train_file = None
    test_file = None
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(caffenet(batch_size=100,learn_all=learn_all)))
	train_file = f.name

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(caffenet(batch_size=50, train=False, learn_all=learn_all)))
        test_file = f.name
    return train_file, test_file


# =========================== Generate Solver =========================

def get_solver(train_net_path, test_net_path, base_lr=0.001):
    s = caffe_pb2.SolverParameter()

    s.train_net = train_net_path

    s.test_net.append(test_net_path)
    s.test_interval = 500  # Test after every 1000 training iterations.
    s.test_iter.append(10) # Test on 10 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1
    
    s.max_iter = 20000  
    
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 10000

    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 100

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 5000
    s.snapshot_prefix = caffe_root + 'models/finetune_flickr_style/finetune_flickr_style'
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    
    # Write the solver to a temporary file and return its filename.
    solver_file = None
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        solver_file = f.name

    solver = caffe.get_solver(solver_file)
    return solver


# =========================== Run Solver To Train =========================

def run_solvers(niter, solvers, disp_interval=10):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)     
    # Save the learned weights from both nets.
    weights = {}
    for name, s in solvers:
        weights[name] = 'weights.%s.caffemodel' % name
        s.net.save(weights[name])
    return loss, acc, weights

# =========================== Predict =========================

def eval_style_net(net_file, weights, test_iters=10):
    test_net = caffe.Net(net_file, weights, caffe.TEST)
    accuracy = 0
    for it in xrange(test_iters):
        accuracy += test_net.forward()['acc']
    accuracy /= test_iters
    return test_net, accuracy


if __name__=="__main__":

    # 生成两种网络配置文件，一种只有最后一层是可学习的，一种所有层都可学习。
    train_file_learn, test_file_learn = make_net()
    # 只有最后一层是可学习的
    train_file_frozen, test_file_frozen = make_net(learn_all=False)


    """
    生成Solver
    用上面两种网络配置文件生成三种网络。
    用caffenet的参数预训练作为参数初始值，然后冻结其他层，只留下最后一层全连接层可训练。
    caffenet的参数作为初始值，不冻结层，所有层均可训练
    没有初始值，只用了caffenet的结构，用自己的数据重新训练
    """
    pretrained_frozen_solver = get_solver(train_file_frozen, test_file_frozen)
    untrained_solver = get_solver(train_file_learn, test_file_learn)
    pretrained_learn_all_solver = get_solver(train_file_learn, test_file_learn)
    
    import os
    weights = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    assert os.path.exists(weights)
    pretrained_frozen_solver.net.copy_from(weights)
    pretrained_learn_all_solver.net.copy_from(weights)


    # Run three solvers
    niter = 200
    print 'Running solvers for %d iterations...' % niter
    solvers = [('pretrained_f', pretrained_frozen_solver),
               ('untrained', untrained_solver),
               ('pretrained_l', pretrained_learn_all_solver)]
    loss, acc, weights = run_solvers(niter, solvers)
    print 'Done.'

    # 可以根据loss和acc进行结果可视化
    # ......

    
　　　　"""
    _, acc_f = eval_style_net(train_file_learn,"weights.pretrained_f.caffemodel")
    _, acc_l = eval_style_net(train_file_learn,"weights.pretrained_l.caffemodel")
    _, acc_u = eval_style_net(train_file_untrained,"weights.untrained.caffemodel")

    print "frozen net acc is ",acc_f
    print "learnall net acc is ", acc_l
    print "untrained net acc is ", acc_u
    """
