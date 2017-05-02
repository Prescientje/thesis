import numpy as np
import pandas as pd
import tensorflow as tf
from math import ceil
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sys import argv
from datetime import date
#script, sample_len, superfactor, test_num, loud = argv
#loud = 1 means print lots of output
#dir_name = './logs/'

def get_recall(con):
    tp = con[0][0]
    tpfn = con[0,:].sum()
    return tp/tpfn

def get_specificity(con):
    tn = con[1][1]
    fptn = con[1,:].sum()
    return tn/fptn

def get_missrate(con):
    return 1-get_recall(con)

def get_fallout(con):
    return 1-get_specificity(con)

def get_precision(con):
    tp = con[0][0]
    tpfp = con[:,0].sum()
    return tp/tpfp

def get_accuracy(con):
    return (con[0][0] + con[1][1])/(con.sum().sum())

def getSection(array,i):
    width = array.shape[1]
    i1 = int((i-1)*width/10)
    i2 = int(i*width/10)
    #print(i1)
    #print(i2)
    test  = array[:,i1:i2]
    train_front  = array[:,:i1]
    train_back   = array[:,i2:]
    train = np.c_[train_front,train_back]
    return train.T,test.T

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def makeLayer(x,insize,outsize,activation,wname,bname,stddeviation=0.1):
    W = tf.Variable(tf.truncated_normal([insize,outsize],stddev=stddeviation),name=wname)
    if activation == "relu":
        b = tf.Variable(tf.truncated_normal([1,outsize],stddev=stddeviation),name=bname)
        return W,b,tf.nn.relu(tf.matmul(x,W) + b)
    else:
        return W,tf.matmul(x,W)

def makeLayerBN(x,insize,outsize,activation,wname,bname,stddeviation=0.1):
    W = tf.Variable(tf.truncated_normal([insize,outsize],stddev=stddeviation),name=wname)
    b = tf.Variable(tf.truncated_normal([1,outsize],stddev=stddeviation),name=bname)
    wxb = tf.matmul(x,W) + b
    mean, variance = tf.nn.moments(wxb, axes=[0])
    scale = tf.Variable(tf.ones([outsize]))
    shift = tf.Variable(tf.zeros([outsize]))
    eps = 0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    def mean_var_with_update():
        ema_op = ema.apply([mean,variance])
        with tf.control_dependencies([ema_op]):
            return tf.identity(mean), tf.identity(variance)
    m2, v2 = mean_var_with_update()
    wxb2 = tf.nn.batch_normalization(wxb, m2, v2, shift, scale, eps)
    return W,b,tf.nn.relu(wxb2)

def makeLayerBNout(x,insize,outsize,activation,wname,bname,stddeviation=0.1):
    W = tf.Variable(tf.truncated_normal([insize,outsize],stddev=stddeviation),name=wname)
    wx = tf.matmul(x,W)
    mean, variance = tf.nn.moments(wx, axes=[0])
    scale = tf.Variable(tf.ones([outsize]))
    shift = tf.Variable(tf.zeros([outsize]))
    eps = 0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    def mean_var_with_update():
        ema_op = ema.apply([mean,variance])
        with tf.control_dependencies([ema_op]):
            return tf.identity(mean), tf.identity(variance)
    m2, v2 = mean_var_with_update()
    wx2 = tf.nn.batch_normalization(wx, m2, v2, shift, scale, eps)
    return W,wx2

def crossmatrix(m1,m2):
    result = np.zeros(m1.shape)
    for j in range(m1.shape[0]-1):
        for i in range(m1.shape[1]-1):
            if m1[j,i] > m2[j,i]:
                result[j,i]=1 
