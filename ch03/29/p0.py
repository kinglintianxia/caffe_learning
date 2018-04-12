#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001

import os

datapath = "/home/hbk/caffe/data/DogsVSCats/"

train_files = os.listdir(datapath+"train")
test_files = os.listdir(datapath+"test")

def write_img_list(phase, filename):
    files_list = os.listdir(datapath+phase)
    with open(filename,'w') as f:
        for file in files_list:
            if file[0]=='c':
                label='0'
            elif file[0]=='d':
                label='1'
            else:
                label='2'
                assert "Error name"+file
            #f.write(datapath+phase+file+' '+label+'\n')
            f.write(phase+file+' '+label+'\n')
write_img_list('train/', 'train2.txt')
write_img_list('test/', 'test2.txt')     
