import numpy as np
import pandas as pd
import tensorflow as tf
#script, sample_len, superfactor, test_num, loud = argv
from math import ceil
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sys import argv
from datetime import date
from preprocess2 import preprocess
from helper import *
#loud = 1 means print lots of output
#dir_name = './logs/'

def print_write(s):
    print(s)
    logfile.write(s)
    logfile.write("\n")

def print_confusion_vals(con):
    print_write("Specificity = %0.6f" % get_specificity(con))
    print_write("Miss Rate   = %0.6f" % get_missrate(con))
    print_write("Fallout     = %0.6f" % get_fallout(con))
    print_write("Recall      = %0.6f" % get_recall(con))
    print_write("Precision   = %0.6f" % get_precision(con))
    print_write("Accuracy    = %0.6f" % get_accuracy(con))

script, sample_len, superfactor, test_num, loud = argv
today = date.today()
file_s = "logs/" + today.strftime("%y%m%d") + "-xyz-%s.txt" % test_num
left_save = sample_len + "-left/" + sample_len + "-left"
right_save = sample_len + "-right/" + sample_len + "-right"
both_save = sample_len + "-both/" + sample_len + "-both"
lmodelname = "savedModels-xyz/" + left_save 
rmodelname = "savedModels-xyz/" + right_save
bmodelname = "savedModels-xyz/" + both_save

#hand = int(hand)
#modelname = "savedModels/" + learn_rate + "-" + hand
print()
print(file_s)
logfile = open(file_s, 'w')

sample_len = int(sample_len)
learn_rate = 0.01
stddeviation = 0.01
loud = int(loud)
batchsize = 200
epochs = 500
superfactor = int(superfactor)
print_write("xyz-MODELBUILDER TEST")
print_write("Sample length = %d" % (sample_len))
print_write("Learning rate = %.3f" % (learn_rate))
print_write("Stddev = %.3f" % (stddeviation))
print_write("Superfactor = %d" % (superfactor))
print_write("Epochs = %d" % (epochs))
print_write("Batch size = %d" % (batchsize))
print_write("lmodelname = %s" % lmodelname)
print_write(" ")


leftvals, rightvals = preprocess(sample_len)


print(leftvals.shape)
print(rightvals.shape)
#samples = df.values[5:,:] #all rows, from 5th column on
left  = leftvals.T[:,5:] #all rows, from 5th column on
right = rightvals.T[:,5:] #all rows, from 5th column on
oleft = leftvals.T[:,3:5]
oright = rightvals.T[:,3:5]

leftvals = 0
rightvals = 0

print_write(str(left.shape))
print_write(str(oleft.shape))

cols   = 3*left.shape[1]
rows   = left.shape[0] - 3 + 1
print_write("cols = %d" % cols)
print_write("rows = %d" % rows)
leftsamples   = np.zeros((rows,cols),dtype=np.float32)
rightsamples  = np.zeros((rows,cols),dtype=np.float32)
outleft  = np.zeros((rows,6),dtype=np.float32)
outright = np.zeros((rows,6),dtype=np.float32)
sample_len *= 3

for j in range(rows):
    for i in range(0,cols,sample_len):
        leftsamples[j,i:i+sample_len] = left[j+i//sample_len]
        rightsamples[j,i:i+sample_len] = right[j+i//sample_len]
    for k in range(0,6,2):
        outleft[j,k:k+2] = oleft[j+k//2]
        outright[j,k:k+2] = oright[j+k//2]

left=0
right=0
oleft=0
oright=0

leftsamples = leftsamples.T
rightsamples = rightsamples.T
outleft = outleft.T
outright = outright.T
print_write("left samples shape = %s" % str(leftsamples.shape))
print_write("out left shape     = %s" % str(outleft.shape))
#print_write("samplelen     = %s" % str(sample_len))

#print_write("xVals shape = %s" % (xVals.shape,))
#print_write("outputs shape = %s" % (outputs.shape,))
#print_write("out_test shape = %s" % (out_test.shape,))
print_write(" ")

# Model is softmax(x_in*weight_x + y_in*weight_y + z_in*weight_z + b)
#shape means we have the amount of values we take (sample_len)
#                 by the total number of those samples we have.

lxyz_place = tf.placeholder(tf.float32, [None,cols])
lout_place = tf.placeholder(tf.float32, [None,2])
weight_lxyz1 = tf.Variable(tf.truncated_normal([cols,6],stddev=stddeviation))
bl1          = tf.Variable(tf.truncated_normal([1,6],stddev=stddeviation))
weight_lxyz2 = tf.Variable(tf.truncated_normal([6,2],stddev=stddeviation))
bl2          = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation))
layerl1 = tf.nn.relu(tf.matmul(lxyz_place,weight_lxyz1) + bl1)
layl2 = tf.matmul(layerl1,weight_lxyz2) + bl2
layerl2 = tf.nn.softmax(layl2)
CE_left = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(layl2,lout_place))
optimizer_left = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08).minimize(CE_left)
rxyz_place = tf.placeholder(tf.float32, [None,cols])
rout_place = tf.placeholder(tf.float32, [None,2])
weight_rxyz1 = tf.Variable(tf.truncated_normal([cols,6],stddev=stddeviation))
br1          = tf.Variable(tf.truncated_normal([1,6],stddev=stddeviation))
weight_rxyz2 = tf.Variable(tf.truncated_normal([6,2],stddev=stddeviation))
br2          = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation))
layerr1 = tf.nn.relu(tf.matmul(rxyz_place,weight_rxyz1) + br1)
layr2 = tf.matmul(layerr1,weight_rxyz2) + br2
layerr2 = tf.nn.softmax(layr2)
CE_right= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(layr2,rout_place))
optimizer_right= tf.train.AdamOptimizer(learning_rate=learn_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08).minimize(CE_right)
lsaver = tf.train.Saver([weight_lxyz1, bl1, weight_lxyz2, bl2])
rsaver = tf.train.Saver([weight_rxyz1, br1, weight_rxyz2, br2])

weight_lxyzb1 = tf.Variable(tf.truncated_normal([cols,6],stddev=stddeviation))
weight_rxyzb1 = tf.Variable(tf.truncated_normal([cols,6],stddev=stddeviation))
bb1           = tf.Variable(tf.truncated_normal([1,6],stddev=stddeviation))
bbl1           = tf.Variable(tf.truncated_normal([1,6],stddev=stddeviation))
bbr1           = tf.Variable(tf.truncated_normal([1,6],stddev=stddeviation))
weight_xyzb2  = tf.Variable(tf.truncated_normal([6,2],stddev=stddeviation))
weight_xyzbl2  = tf.Variable(tf.truncated_normal([6,2],stddev=stddeviation))
weight_xyzbr2  = tf.Variable(tf.truncated_normal([6,2],stddev=stddeviation))
bb2           = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation))

#layerb1 = tf.nn.relu(tf.matmul(lxyz_place,weight_lxyzb1) + \
#                     tf.matmul(rxyz_place,weight_rxyzb1) + bb1)
layerbl1 = tf.nn.relu(tf.matmul(lxyz_place,weight_lxyzb1) + bbl1)
layerbr1 = tf.nn.relu(tf.matmul(lxyz_place,weight_lxyzb1) + bbr1)

#layb2 = tf.matmul(layerb1,weight_xyzb2) + bb2
layb2 = tf.matmul(layerbl1,weight_xyzbl2)+tf.matmul(layerbr1,weight_xyzbr2)+bb2
layerb2 = tf.nn.softmax(layb2)
CE_both = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(layb2,lout_place))
optimizer_both = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08).minimize(CE_both)
#bsaver = tf.train.Saver([weight_lxyzb1, weight_rxyzb1, bb1,
#                         weight_xyzb2, bb2])
bsaver = tf.train.Saver([weight_lxyzb1, weight_rxyzb1, bbl1, bbr1
                         weight_xyzbl2, weight_xyzbr2, bb2])

init = tf.initialize_all_variables()

#sess = tf.Session()
#sess.run(init)


train_scores = np.zeros([10,3])
test_scores = np.zeros([10,3])
lconfusion_sums = np.zeros([2,2])
lconfusion_sumst = np.zeros([2,2])
rconfusion_sums = np.zeros([2,2])
rconfusion_sumst = np.zeros([2,2])
bconfusion_sums = np.zeros([2,2])
bconfusion_sumst = np.zeros([2,2])
minscoret = np.array([100,100,100])
maxscoret = np.zeros(3)


for k in range(10): 
    sess = tf.Session()
    sess.run(init)
    train_lxyzi, test_lxyzi = getSection(leftsamples,k+1)
    train_louti, test_louti = getSection(outleft,k+1)
    train_rxyzi, test_rxyzi = getSection(rightsamples,k+1)
    train_routi, test_routi = getSection(outright,k+1)
    train_ldata_feeder={lxyz_place: train_lxyzi,
                               lout_place: train_louti[:,2:4]}
    test_ldata_feeder={lxyz_place: test_lxyzi,
                              lout_place: test_louti[:,2:4]}
    train_rdata_feeder={rxyz_place: train_rxyzi,
                               rout_place: train_routi[:,2:4]}
    test_rdata_feeder={rxyz_place: test_rxyzi,
                              rout_place: test_routi[:,2:4]}
    train_bdata_feeder={lxyz_place: train_lxyzi,
                               rxyz_place: train_rxyzi,
                               lout_place: train_louti[:,2:4]}
    test_bdata_feeder={lxyz_place: test_lxyzi,
                              rxyz_place: test_rxyzi,
                              lout_place: test_louti[:,2:4]}
    nb = int(np.ceil(len(train_lxyzi)/batchsize))
    if loud > 0:
        print_write(" ")
        print_write(" ")
        print_write("Trial %d %s" % ((k+1), lmodelname))
        print_write(" ")

    for e in range(epochs):
        #print_write("epoch %d of %d xyz" % (e+1,epochs))
        #print_write("train len = %d" % len(train_lxyzi))
        ix = 0
        perm = np.random.permutation(len(train_lxyzi))
        for i in range(nb):
            batch_range = np.arange(ix,ix+batchsize)
            if ix+batchsize > len(train_lxyzi):
                batch_range = np.arange(ix,len(train_lxyzi))
            #print(perm[batch_range][:5])
            #print(i)
            #print(batch_range[:5])
            #print(train_lxi[perm[batch_range]][0][:5])
            lbatch_feed = {lxyz_place: train_lxyzi[perm[batch_range]],
                           lout_place: train_louti[perm[batch_range]][:,2:4]}
            rbatch_feed = {rxyz_place: train_rxyzi[perm[batch_range]],
                           rout_place: train_routi[perm[batch_range]][:,2:4]}
            bbatch_feed = {lxyz_place: train_lxyzi[perm[batch_range]],
                           rxyz_place: train_rxyzi[perm[batch_range]],
                           lout_place: train_louti[perm[batch_range]][:,2:4]}

            ix += batchsize
            sess.run(optimizer_left, feed_dict=lbatch_feed)
            sess.run(optimizer_right,feed_dict=rbatch_feed)
            sess.run(optimizer_both, feed_dict=bbatch_feed)
        if (e+1) % (epochs/5) == 0:
            print(e+1,"xyz",sample_len)
    loutputs_pred = sess.run(layerl2, feed_dict=train_ldata_feeder)
    loutputs_test = sess.run(layerl2, feed_dict= test_ldata_feeder)
    routputs_pred = sess.run(layerr2, feed_dict=train_rdata_feeder)
    routputs_test = sess.run(layerr2, feed_dict= test_rdata_feeder)
    boutputs_pred = sess.run(layerb2, feed_dict=train_bdata_feeder)
    boutputs_test = sess.run(layerb2, feed_dict= test_bdata_feeder)

    #calculate Xscore (train/test, Xconfusion matrix (train/test), and Xroc_auc (train/test)
    #for left, right, and both
    ls = accuracy_score(train_ldata_feeder[lout_place].argmax(axis=1),loutputs_pred.argmax(axis=1))
    lst = accuracy_score(test_ldata_feeder[lout_place].argmax(axis=1),loutputs_test.argmax(axis=1))
    lc = confusion_matrix(train_ldata_feeder[lout_place].argmax(axis=1),loutputs_pred.argmax(axis=1))
    lct = confusion_matrix(test_ldata_feeder[lout_place].argmax(axis=1),loutputs_test.argmax(axis=1))
    lconfusion_sums += lc
    lconfusion_sumst += lct
    #try:
    #    lra = roc_auc_score(train_ldata_feeder[k][lout_place].argmax(axis=1),loutputs_pred.argmax(axis=1))
    #except ValueError:
    #    lra = -1
    #try:
    #    lrat = roc_auc_score(test_ldata_feeder[k][lout_place].argmax(axis=1),loutputs_test.argmax(axis=1))
    #except ValueError:
    #    lrat = -1

    rs = accuracy_score(train_rdata_feeder[rout_place].argmax(axis=1),routputs_pred.argmax(axis=1))
    rst = accuracy_score(test_rdata_feeder[rout_place].argmax(axis=1),routputs_test.argmax(axis=1))
    rc = confusion_matrix(train_rdata_feeder[rout_place].argmax(axis=1),routputs_pred.argmax(axis=1))
    rct = confusion_matrix(test_rdata_feeder[rout_place].argmax(axis=1),routputs_test.argmax(axis=1))
    rconfusion_sums += rc
    rconfusion_sumst += rct

    bs = accuracy_score(train_bdata_feeder[lout_place].argmax(axis=1),boutputs_pred.argmax(axis=1))
    bst = accuracy_score(test_bdata_feeder[lout_place].argmax(axis=1),boutputs_test.argmax(axis=1))
    bc = confusion_matrix(train_bdata_feeder[lout_place].argmax(axis=1),boutputs_pred.argmax(axis=1))
    bct = confusion_matrix(test_bdata_feeder[lout_place].argmax(axis=1),boutputs_test.argmax(axis=1))
    bconfusion_sums += bc
    bconfusion_sumst += bct

    if loud > 0:
        print_write(" ")
        print_write(" ")
        print_write("ltrain accuracy score = %f" % (ls))
        print_write("ltest  accuracy score = %f" % (lst))
        print_write("ltrain confusion matrix =\n%s" % (str(lc)))
        print_write("ltest  confusion matrix =\n%s" % (str(lct)))
        #print_write("ltrain roc auc score = %.6f" % (lra))
        #print_write("ltest  roc auc score = %.6f" % (lrat))
        print_write(" ")
        print_write("rtrain accuracy score = %f" % (rs))
        print_write("rtest  accuracy score = %f" % (rst))
        print_write("rtrain confusion matrix =\n%s" % (str(rc)))
        print_write("rtest  confusion matrix =\n%s" % (str(rct)))
        print_write(" ")
        print_write("btrain accuracy score = %f" % (bs))
        print_write("btest  accuracy score = %f" % (bst))
        print_write("btrain confusion matrix =\n%s" % (str(bc)))
        print_write("btest  confusion matrix =\n%s" % (str(bct)))
        print_write(" ")
    #the count of true  negatives is C_{0,0},
                 #false negatives is C_{1,0}, 
                 #true  positives is C_{1,1}, 
             #and false positives is C_{0,1}.

    train_scores[k,0] = ls
    test_scores[k,0] = lst
    train_scores[k,1] = rs
    test_scores[k,1] = rst
    train_scores[k,2] = bs
    test_scores[k,2] = bst

    if lst > maxscoret[0]:
        maxscoret[0] = lst
        lsaver.save(sess=sess,save_path=lmodelname)
        print_write("lsaver saved %d" % (k+1))
    if lst < minscoret[0]:
        minscoret[0] = lst
    if rst > maxscoret[1]:
        maxscoret[1] = rst
        rsaver.save(sess=sess,save_path=rmodelname)
        print_write("rsaver saved %d" % (k+1))
    if rst < minscoret[1]:
        minscoret[1] = rst
    if bst > maxscoret[2]:
        maxscoret[2] = bst
        bsaver.save(sess=sess,save_path=bmodelname)
        print_write("bsaver saved %d" % (k+1))
    if bst < minscoret[2]:
        minscoret[2] = bst
        

print_write(" ")
print_write("Left Train Confusion Vals")
print_confusion_vals(lconfusion_sums)
print_write("Left Test Confusion Vals")
print_confusion_vals(lconfusion_sumst)
print_write("Right Train Confusion Vals")
print_confusion_vals(rconfusion_sums)
print_write("Right Test Confusion Vals")
print_confusion_vals(rconfusion_sumst)
print_write("Both Train Confusion Vals")
print_confusion_vals(bconfusion_sums)
print_write("Both Test Confusion Vals")
print_confusion_vals(bconfusion_sumst)
print_write("Both Test Confusion Matrix")
print_write("%s" % (str(bconfusion_sumst)))

print_write(" ")
print_write("average train score   = \n%s" % str(train_scores.mean(axis=0)))
print_write("average test score    = \n%s" % str(test_scores.mean(axis=0)))
print_write("best test score       = \n%s" % str(maxscoret))
print_write("worst test score      = \n%s" % str(minscoret))
