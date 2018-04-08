#!/usr/bin/env python
# coding: utf-8
##################
# king@2018.04.07
# lmdb data generator
# lighting memory-mapped database
import numpy as np
import lmdb
import caffe


def write_lmdb(filename, X,y):
    """
    filename: lmdb data dir
    x: data
    y: label
    """
    N = len(y)
    map_size = X.nbytes * 10

    env = lmdb.open(filename, map_size=map_size)

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(N):
            """
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(y[i])
            str_id = '{:08}'.format(i)

            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
            """
            datum = caffe.io.array_to_datum(X[i,:,:,:])
            datum.label = int(y[i])
            txn.put('{:0>10d}'.format(i), datum.SerializeToString())

def read_lmdb(filename):
    """
    filename: lmdb data dir
    x: last data
    y: last label
    """
    env = lmdb.open(filename, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        datum = caffe.proto.caffe_pb2.Datum()
        
        i=0
        for key, value in cursor:
            i=i+1

        datum.ParseFromString(value)
        #flat_x = np.fromstring(datum.data, dtype=np.uint8)
        #x = flat_x.reshape(datum.channels, datum.height, datum.width)
        x = caffe.io.datum_to_array(datum)
        y = datum.label
        return x,y



if __name__ == '__main__':
    N = 10000
    # image data, size(batch_size, channels, height, width)=size(1000,3,32,32)
    X1 = np.random.randint(1, 10, (N, 3, 32, 32))
    y1 = np.zeros(N, dtype=np.int64)    # label:0
    X2 = np.random.randint(1, 10, (N, 3, 32, 32))+10
    y2 = np.ones(N, dtype=np.int64) # label:1
    X3 = np.random.randint(1, 10, (N, 3, 32, 32))+20
    y3 = np.ones(N, dtype=np.int64) * 2 # # label:2
    X4 = np.random.randint(1, 10, (N, 3, 32, 32))+30
    y4 = np.ones(N, dtype=np.int64) * 3 # label:3

    X = np.vstack((X1, X2, X3, X4)) # Stack arrays in sequence vertically (row wise).
    y = np.hstack((y1, y2, y3, y4)) # Stack arrays in sequence horizontally (column wise).
    # random data
    idx = np.arange(len(y))
    np.random.shuffle(idx)

    TRAIN_NUM = 4*len(y)/5

    write_lmdb("lmdb_train", X[idx[0:TRAIN_NUM], :, :, :], y[idx[0:TRAIN_NUM]])
    write_lmdb("lmdb_test", X[idx[TRAIN_NUM:], :, :, :], y[idx[TRAIN_NUM:]])

    X1, y1 = read_lmdb("lmdb_train")

    print 'X1.shape: ', X1.shape, '\ny1: ', y1
    print 'np.mean(X): ', np.mean(X)
