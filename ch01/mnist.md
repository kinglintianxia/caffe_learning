# mnist example
---
##1 download mnist data
```shell 
sudo ./data/mnist/get_mnist.sh
```
##2 create lmdb
```shell
sudo ./examples/mnist/create_mnist.sh
```
##3 train lenet
```shell
./build/tools/caffe train --solver=/home/king/Documents/caffe/examples/mnist/lenet_solver.prototxt
```
##4 draw net
```shell
python python/draw_net.py ./examples/mnist/lenet.prototxt lenet.png --rankdir=LR
```
##5 parse log
```shell
./build/tools/caffe train --solver=/home/king/Documents/caffe/examples/mnist/lenet_solver.prototxt 2>&1 | tee lenet.log
./tools/extra/plot_training_log.py.example 6 lenet.png ./lenet.log
``` 
