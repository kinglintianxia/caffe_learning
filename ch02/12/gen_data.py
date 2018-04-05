#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001

import h5py
import numpy as np

f = h5py.File('train00.h5', 'w')
# 1200 data, each is a 128-dim vector
f.create_dataset('data', (1200, 128), dtype='f8')
# Data's labels, each is a 4-dim vector
f.create_dataset('label', (1200, 4), dtype='i')

# Fill in something with fixed pattern
# Regularize values to between 0 and 1, or SigmoidCrossEntropyLoss will not work
for i in range(1200):
    a = np.empty(128)
    if i % 4 == 0:
        for j in range(128):
            a[j] = j / 128.0;
        l = [1,0,0,0]
    elif i % 4 == 1:
        for j in range(128):
            a[j] = (128 - j) / 128.0;
        l = [1,0,1,0]
    elif i % 4 == 2:
        for j in range(128):
            a[j] = (j % 6) / 128.0;
        l = [0,1,1,0]
    elif i % 4 == 3:
        for j in range(128):
            a[j] = (j % 4) * 4 / 128.0;
        l = [1,0,1,1]
    f['data'][i] = a
    f['label'][i] = l

f.close()

with open('train.h5list','w') as f:
    f.write('train00.h5')

with open('test.h5list','w') as f:
    f.write('train00.h5')
