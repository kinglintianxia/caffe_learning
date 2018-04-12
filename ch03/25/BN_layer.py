#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001

import caffe
from caffe import layers as L
from caffe import params as P
import caffe.draw
from caffe.proto import caffe_pb2
from google.protobuf import text_format


# 配置文件路径和文件名
path=''             
train_net_file = path+'train_batch_norm_py.prototxt'


def batch_norm_net():
    n = caffe.NetSpec()
    n.data = L.DummyData(dummy_data_param=dict(num=64,channels=1,height=28,width=28,data_filler=dict(type="gaussian")))
    n.label = L.DummyData(dummy_data_param=dict(num=64,channels=1,height=1,width=1,data_filler=dict(type="gaussian")))
    n.conv1 = L.Convolution(n.data,kernel_size=7,stride=2,num_output=32,pad=3)
    n.pool1 = L.Pooling(n.conv1,pool=P.Pooling.MAX,kernel_size=2,stride=2)
    n.relu1 = L.ReLU(n.pool1,in_place=True)

    n.norm1 = L.BatchNorm(n.relu1, moving_average_fraction=0.9, in_place=True)
    n.scale1 = L.Scale(n.norm1,bias_term=True,in_place=True)

    n.ip2 = L.InnerProduct(n.scale1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()


with open( train_net_file, 'w') as f:
    f.write(str(batch_norm_net( )))


def draw_net(net_file, jpg_file):
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(net_file).read(), net)
    caffe.draw.draw_net_to_file(net, jpg_file, 'BT')

draw_net(train_net_file, "a.jpg")

