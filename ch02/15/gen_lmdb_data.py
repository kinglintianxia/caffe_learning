#!/usr/bin/env python
# coding: utf-8
##################
# king@2018.04.07
# lmdb data multilabel

import numpy as np
import lmdb
import caffe
# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.

def write_lmdb_data(filename, X):
    """
    filename: lmdb data dir
    x: data
    y: label
    """
    N = X.shape[0]
    map_size = X.nbytes * 10
    env = lmdb.open(filename, map_size=map_size)

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(N):
            datum = caffe.io.array_to_datum(X[i,:,:,:])
            txn.put('{:0>10d}'.format(i), datum.SerializeToString())


if __name__ == '__main__':
    N = 1000

    # Let's pretend this is interesting data
    # numpy.random.randint(low, high=None, size=None, dtype='l')
    # Return random integers from low (inclusive) to high (exclusive).
    X1 = np.random.randint(1, 10, (N, 3, 32, 32))
    # 0,0,0,0,....
    y1 = np.zeros((N,10,1,1), dtype=np.int64)
    
    X2 = np.random.randint(1, 10, (N, 3, 32, 32))+10
    # 0,1,0,1,0,....
    y2 = np.zeros((N,10,1, 1), dtype=np.int64)
    # label: [0,1,0,1,0,0,0,0,0,0]
    y2[:,1,:,:] = 1; y2[:,3,:,:] = 1    # y2[1000,10,1,1]

    X3 = np.random.randint(1, 10, (N, 3, 32, 32))+20
    # 1,0,1,0,0,....
    y3 = np.zeros((N,10,1, 1), dtype=np.int64)
    # label: [1,0,1,0,0,0,0,0,0,0]
    y3[:,0,:,:] = 1; y3[:,2,:,:] = 1
    
    X4 = np.random.randint(1, 10, (N, 3, 32, 32))+30
    # label: [1,1,1,1,1,1,1,1,1,1]
    y4 = np.ones((N,10,1,1), dtype=np.int64) 

    X = np.vstack((X1, X2, X3, X4)) # Stack arrays in sequence vertically (row wise).
    y = np.vstack((y1, y2, y3, y4))

    idx = np.arange(len(y))
    np.random.shuffle(idx)

    TRAIN_NUM = 4*len(y)/5

    write_lmdb_data("lmdb_train_data", X[idx[0:TRAIN_NUM], :, :, :])
    write_lmdb_data("lmdb_train_label", y[idx[0:TRAIN_NUM], :])
    write_lmdb_data("lmdb_test_data", X[idx[TRAIN_NUM:], :, :, :])
    write_lmdb_data("lmdb_test_label", y[idx[TRAIN_NUM:], :])
    print np.mean(X)
