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

def crossmatrix(m1,m2):
    result = np.zeros(m1.shape)
    for j in range(m1.shape[0]-1):
        for i in range(m1.shape[1]-1):
            if m1[j,i] > m2[j,i]:
                result[j,i]=1 
