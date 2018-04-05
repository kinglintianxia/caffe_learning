#!/usr/bin/env python
# coding: utf-8
######################
# king@2018.04.04 
# Image data generate

import os
import cv2
import numpy as np
import pdb

def write_img_list(data, filename):
    with open(filename, 'w') as f:
    	# xrange() 函数用法与 range 完全相同，所不同的是生成的不是一个数组，而是一个生成器。
        for i in xrange(len(data)):
            f.write(data[i][0]+' '+str(data[i][1])+'\n')


image_size = 28
s='ABCDEFGHIJ'

filedir='/home/king/Documents/king/caffe_learning/ch02/05/notMNIST_small/'


# 1. read file
# os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
filedir2 = os.listdir(filedir)
print 'filedir2: ', filedir2

datasets=[]
data=[]
for subdir in filedir2:
    if os.path.isdir(filedir+subdir):
        files=os.listdir(filedir+subdir)	# *.png
        dataset = np.ndarray(shape=(len(files), image_size, image_size),
                         dtype=np.float32)
        
        num_image = 0
        for file in files:	# image name
            if file[-3:]=='png':
                tmp=cv2.imread(filedir+subdir+'/'+file,cv2.IMREAD_GRAYSCALE)
                #判断图像大小是否符合要求，不符合则跳过
                try:
                    if tmp.shape==(image_size,image_size):
    				# Python rfind()返回字符串最后一次出现的位置，如果没有匹配项则返回-1。
                        datasets.append((filedir+subdir+'/'+file, s.rfind(subdir)))
                        data.append(tmp)
                        num_image+=1
                    else:
                        print subdir,file,tmp.shape
                except:
                    print subdir,file,tmp
            else:
                print file

#随机化数据序列，计算均值
np.random.shuffle(datasets)
print np.mean(np.array(data))

TRAIN_NUM = 4*len(datasets)/5

write_img_list(datasets[0:TRAIN_NUM], 'train00.imglist')
write_img_list(datasets[TRAIN_NUM:], 'test00.imglist')


