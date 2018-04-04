# Install caffe
---
##1 install cuda9.1
* follow CUDA_Installation_Guide_Linux.pdf
* add follow two lines to ~/.bashr
```shell 
export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 
```
##2 install cudnn
* [download link](https://developer.nvidia.com/rdp/cudnn-download)
* version: cuDNN v7.1.2 Library for Linux
```shell
$ sudo cp cudnn.h /usr/local/cuda/include/ #复制头文件
$ sudo cp lib* /usr/local/cuda/lib64/ #复制动态链接库
$ cd /usr/local/cuda/lib64/
$ sudo rm -rf libcudnn.so libcudnn.so.7  #删除原有动态文件
$ sudo ln -s libcudnn.so.7.1.2 libcudnn.so.7  #生成软衔接
$ sudo ln -s libcudnn.so.7 libcudnn.so  #生成软链接
$ sudo vim /etc/ld.so.conf.d/cuda.conf
>/usr/local/cuda/lib64
```
##3 opencv 3.4.0
##4 caffe
* [BVLC/caffe](https://github.com/BVLC/caffe)
* Dependencies
```shell
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libatlas-base-dev
```
* Modify Makefile.config
```shell
sudo cp Makefile.config.example Makefile.config
sudo gedit Makefile.config
> USE_CUDNN := 1
> OPENCV_VERSION := 3
> WITH_PYTHON_LAYER := 1
> INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
> LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial
> CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
            -gencode arch=compute_35,code=sm_35 \  
            -gencode arch=compute_50,code=sm_50 \  
            -gencode arch=compute_52,code=sm_52 \  
            -gencode arch=compute_60,code=sm_60 \  
            -gencode arch=compute_61,code=sm_61 \  
            -gencode arch=compute_61,code=compute_61 
```
* Modify Makefile
```shell
> NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
> LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_serial_hl hdf5_serial
* Python requirements
```shell
cd caffe/python
for req in $(cat requirements.txt); do pip install $req; done
```
* Compilation
```shell
make all
make test
make runtest
make pycaffe
```
##5 Test caffe
```shell
./build/tools/caffe; shows help imformation
import caffe no error ocurs
``` 
