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
net_file = path+'concat_slice_py.prototxt'


def concat_slice_net():
    n = caffe.NetSpec()
    n.data = L.DummyData(dummy_data_param=dict(num=20,channels=50,height=64,width=64,data_filler=dict(type="gaussian")))
    n.a, n.b,n.c = L.Slice(n.data, ntop=3, slice_point=[20,30],axis=0)
    n.d = L.Concat(n.a,n.b,axis=0)
    n.e = L.Eltwise(n.a,n.c)

    return n.to_proto()


def draw_net(net_file, jpg_file):
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(net_file).read(), net)
    caffe.draw.draw_net_to_file(net, jpg_file, 'BT')


with open( net_file, 'w') as f:
    f.write(str(concat_slice_net()))


draw_net(net_file, "a.jpg")




