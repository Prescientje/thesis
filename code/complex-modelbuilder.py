import numpy as np
import pandas as pd
import tensorflow as tf
#script, sample_len, superfactor, test_num, loud = argv
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
file_s = "logs/" + today.strftime("%y%m%d") + "-cmb-%s.txt" % test_num
left_save = sample_len + "-left/" + sample_len + "-left"
right_save = sample_len + "-right/" + sample_len + "-right"
both_save = sample_len + "-both/" + sample_len + "-both"
lmodelname = "savedModels-complex/" + left_save 
rmodelname = "savedModels-complex/" + right_save
bmodelname = "savedModels-complex/" + both_save

print()
print(file_s)
logfile = open(file_s, 'w')

sample_len = int(sample_len)
learn_rate = 0.002
stddeviation = 0.01
loud = int(loud)
batchsize = 200
epochs = 1000
superfactor = int(superfactor)
print_write("complex-MODELBUILDER TEST")
print_write("Sample length = %d" % (sample_len))
print_write("Learning rate = %.6f" % (learn_rate))
print_write("Stddev = %.6f" % (stddeviation))
print_write("Superfactor = %d" % (superfactor))
print_write("Epochs = %d" % (epochs))
print_write("Batch size = %d" % (batchsize))


leftvals,rightvals = preprocess(sample_len)
print_write("lmodelname = %s" % lmodelname)
print_write(" ")


#samples = df.values[5:,:] #all rows, from 5th column on
leftsamples  = leftvals[5:,:] #all rows, from 5th column on
rightsamples = rightvals[5:,:] #all rows, from 5th column on

xleft = leftsamples[::3,:]
yleft = leftsamples[1::3,:]
zleft = leftsamples[2::3,:]
xright = rightsamples[::3,:]
yright = rightsamples[1::3,:]
zright = rightsamples[2::3,:]
outleft = leftvals[3:5,:]
outright = rightvals[3:5,:]
leftvals = 0
rightvals = 0

print_write("starting tensorflow section")

# Model is softmax(x_in*weight_x + y_in*weight_y + z_in*weight_z + b)
#shape means we have the amount of values we take (sample_len)
#                 by the total number of those samples we have.

lx_place = tf.placeholder(tf.float32, [None,sample_len])
ly_place = tf.placeholder(tf.float32, [None,sample_len])
lz_place = tf.placeholder(tf.float32, [None,sample_len])
lxx_place = tf.placeholder(tf.float32, [None,sample_len])
lyy_place = tf.placeholder(tf.float32, [None,sample_len])
lzz_place = tf.placeholder(tf.float32, [None,sample_len])
lxy_place = tf.placeholder(tf.float32, [None,sample_len])
lyz_place = tf.placeholder(tf.float32, [None,sample_len])
lzx_place = tf.placeholder(tf.float32, [None,sample_len])
lout_place = tf.placeholder(tf.float32, [None,2])

weight_lx = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ly = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lz = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lxx = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lyy = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lzz = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lxy = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lyz = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lzx = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
bl        = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation))
lsaver = tf.train.Saver([weight_lx, weight_ly, weight_lz,
                         weight_lxx, weight_lyy, weight_lzz,
                         weight_lxy, weight_lyz, weight_lzx, bl])

pred_left = tf.matmul(lx_place,weight_lx) + \
                                tf.matmul(ly_place,weight_ly) + \
                                tf.matmul(lz_place,weight_lz) + \
                                tf.matmul(lxx_place,weight_lxx) + \
                                tf.matmul(lyy_place,weight_lyy) + \
                                tf.matmul(lzz_place,weight_lzz) + \
                                tf.matmul(lxy_place,weight_lxy) + \
                                tf.matmul(lyz_place,weight_lyz) + \
                                tf.matmul(lzx_place,weight_lzx) + bl
prediction_left = tf.nn.softmax(pred_left)
#CE_left = tf.reduce_mean(-tf.reduce_sum(lout_place*tf.log(prediction_left), 
                                        #reduction_indices=[1]))
CE_left = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_left,lout_place))
optimizer_left = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08).minimize(CE_left)

rx_place = tf.placeholder(tf.float32, [None,sample_len])
ry_place = tf.placeholder(tf.float32, [None,sample_len])
rz_place = tf.placeholder(tf.float32, [None,sample_len])
rxx_place = tf.placeholder(tf.float32, [None,sample_len])
ryy_place = tf.placeholder(tf.float32, [None,sample_len])
rzz_place = tf.placeholder(tf.float32, [None,sample_len])
rxy_place = tf.placeholder(tf.float32, [None,sample_len])
ryz_place = tf.placeholder(tf.float32, [None,sample_len])
rzx_place = tf.placeholder(tf.float32, [None,sample_len])
rout_place = tf.placeholder(tf.float32, [None,2])

weight_rx = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ry = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rz = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rxx = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ryy = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rzz = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rxy = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ryz = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rzx = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
br        = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation))
rsaver = tf.train.Saver([weight_rx, weight_ry, weight_rz,
                         weight_rxx, weight_ryy, weight_rzz,
                         weight_rxy, weight_ryz, weight_rzx, br])

pred_right = tf.matmul(rx_place,weight_rx) + \
                                 tf.matmul(ry_place,weight_ry) + \
                                 tf.matmul(rz_place,weight_rz) + \
                                 tf.matmul(rxx_place,weight_rxx) + \
                                 tf.matmul(ryy_place,weight_ryy) + \
                                 tf.matmul(rzz_place,weight_rzz) + \
                                 tf.matmul(rxy_place,weight_rxy) + \
                                 tf.matmul(ryz_place,weight_ryz) + \
                                 tf.matmul(rzx_place,weight_rzx) + br
prediction_right = tf.nn.softmax(pred_right)
CE_right = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_right,rout_place))
optimizer_right = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                         beta1=0.9,
                                         beta2=0.999,
                                         epsilon=1e-08).minimize(CE_right)

weight_lxb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lyb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lzb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lxxb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lyyb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lzzb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lxyb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lyzb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lzxb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rxb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ryb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rzb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rxxb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ryyb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rzzb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rxyb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ryzb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rzxb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
bb        = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation))
bsaver = tf.train.Saver([weight_lxb, weight_lyb, weight_lzb,
                         weight_lxxb, weight_lyyb, weight_lzzb,
                         weight_lxyb, weight_lyzb, weight_lzxb,
                         weight_rxb, weight_ryb, weight_rzb,
                         weight_rxxb, weight_ryyb, weight_rzzb,
                         weight_rxyb, weight_ryzb, weight_rzxb, bb])

pred_both = tf.matmul(lx_place,weight_lxb) + \
                                tf.matmul(ly_place,weight_lyb) + \
                                tf.matmul(lz_place,weight_lzb) + \
                                tf.matmul(lxx_place,weight_lxxb) + \
                                tf.matmul(lyy_place,weight_lyyb) + \
                                tf.matmul(lzz_place,weight_lzzb) + \
                                tf.matmul(lxy_place,weight_lxyb) + \
                                tf.matmul(lyz_place,weight_lyzb) + \
                                tf.matmul(lzx_place,weight_lzxb) + \
                                tf.matmul(rx_place,weight_rxb) + \
                                tf.matmul(ry_place,weight_ryb) + \
                                tf.matmul(rz_place,weight_rzb) + \
                                tf.matmul(rxx_place,weight_rxxb) + \
                                tf.matmul(ryy_place,weight_ryyb) + \
                                tf.matmul(rzz_place,weight_rzzb) + \
                                tf.matmul(rxy_place,weight_rxyb) + \
                                tf.matmul(ryz_place,weight_ryzb) + \
                                tf.matmul(rzx_place,weight_rzxb) + bb 
prediction_both = tf.nn.softmax(pred_both)
CE_both = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_both,lout_place))
optimizer_both = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08).minimize(CE_both)

init = tf.initialize_all_variables()

#sess = tf.Session()
#sess.run(init)

ce_scores = np.zeros([10,3])
train_scores = np.zeros([10,3])
test_scores = np.zeros([10,3])
lconfusion_sums = np.zeros([2,2])
lconfusion_sumst = np.zeros([2,2])
rconfusion_sums = np.zeros([2,2])
rconfusion_sumst = np.zeros([2,2])
bconfusion_sums = np.zeros([2,2])
bconfusion_sumst = np.zeros([2,2])
minscoret = np.zeros(3)
for i in range(3):
    minscoret[i] = 100
maxscoret = np.zeros(3)


for k in range(10): 
    sess = tf.Session()
    sess.run(init)

    train_lxi, test_lxi = getSection(xleft,k+1)
    train_lyi, test_lyi = getSection(yleft,k+1)
    train_lzi, test_lzi = getSection(zleft,k+1)
    train_louti, test_louti = getSection(outleft,k+1)
    train_rxi, test_rxi = getSection(xright,k+1)
    train_ryi, test_ryi = getSection(yright,k+1)
    train_rzi, test_rzi = getSection(zright,k+1)
    train_routi, test_routi = getSection(outright,k+1)
    train_ldata_feeder={lx_place: train_lxi,
                               ly_place: train_lyi,
                               lz_place: train_lzi,
                               lxx_place: train_lxi*train_lxi,
                               lyy_place: train_lyi*train_lyi,
                               lzz_place: train_lzi*train_lzi,
                               lxy_place: train_lxi*train_lyi,
                               lyz_place: train_lyi*train_lzi,
                               lzx_place: train_lzi*train_lxi,
                               lout_place: train_louti}
    test_ldata_feeder={lx_place: test_lxi,
                              ly_place: test_lyi,
                              lz_place: test_lzi,
                              lxx_place: test_lxi*test_lxi,
                              lyy_place: test_lyi*test_lyi,
                              lzz_place: test_lzi*test_lzi,
                              lxy_place: test_lxi*test_lyi,
                              lyz_place: test_lyi*test_lzi,
                              lzx_place: test_lzi*test_lxi,
                              lout_place: test_louti}
    train_rdata_feeder={rx_place: train_rxi,
                               ry_place: train_ryi,
                               rz_place: train_rzi,
                               rxx_place: train_rxi*train_rxi,
                               ryy_place: train_ryi*train_ryi,
                               rzz_place: train_rzi*train_rzi,
                               rxy_place: train_rxi*train_ryi,
                               ryz_place: train_ryi*train_rzi,
                               rzx_place: train_rzi*train_rxi,
                               rout_place: train_routi}
    test_rdata_feeder={rx_place: test_rxi,
                              ry_place: test_ryi,
                              rz_place: test_rzi,
                              rxx_place: test_rxi*test_rxi,
                              ryy_place: test_ryi*test_ryi,
                              rzz_place: test_rzi*test_rzi,
                              rxy_place: test_rxi*test_ryi,
                              ryz_place: test_ryi*test_rzi,
                              rzx_place: test_rzi*test_rxi,
                              rout_place: test_routi}
    train_bdata_feeder={lx_place: train_lxi,
                               ly_place: train_lyi,
                               lz_place: train_lzi,
                               lxx_place: train_lxi*train_lxi,
                               lyy_place: train_lyi*train_lyi,
                               lzz_place: train_lzi*train_lzi,
                               lxy_place: train_lxi*train_lyi,
                               lyz_place: train_lyi*train_lzi,
                               lzx_place: train_lzi*train_lxi,
                               rx_place: train_rxi,
                               ry_place: train_ryi,
                               rz_place: train_rzi,
                               rxx_place: train_rxi*train_rxi,
                               ryy_place: train_ryi*train_ryi,
                               rzz_place: train_rzi*train_rzi,
                               rxy_place: train_rxi*train_ryi,
                               ryz_place: train_ryi*train_rzi,
                               rzx_place: train_rzi*train_rxi,
                               lout_place: train_louti}
    test_bdata_feeder={lx_place: test_lxi,
                              ly_place: test_lyi,
                              lz_place: test_lzi,
                              lxx_place: test_lxi*test_lxi,
                              lyy_place: test_lyi*test_lyi,
                              lzz_place: test_lzi*test_lzi,
                              lxy_place: test_lxi*test_lyi,
                              lyz_place: test_lyi*test_lzi,
                              lzx_place: test_lzi*test_lxi,
                              rx_place: test_rxi,
                              ry_place: test_ryi,
                              rz_place: test_rzi,
                              rxx_place: test_rxi*test_rxi,
                              ryy_place: test_ryi*test_ryi,
                              rzz_place: test_rzi*test_rzi,
                              rxy_place: test_rxi*test_ryi,
                              ryz_place: test_ryi*test_rzi,
                              rzx_place: test_rzi*test_rxi,
                              lout_place: test_louti}

    nb = int(np.ceil(len(train_lxi)/batchsize))

    if loud > 0:
        print_write(" ")
        print_write(" ")
        print_write("Trial %d %s" % ((k+1), lmodelname))
        print_write(" ")

    for e in range(epochs):
        #print("epoch %d of %d complex %d" % (e+1,epochs,sample_len))
        #print_write("train len = %d" % len(train_lxi))
        ix = 0
        perm = np.random.permutation(len(train_lxi))
        for i in range(nb):
            batch_range = np.arange(ix,ix+batchsize)
            if ix+batchsize > len(train_lxi):
                #print("here")
                batch_range = np.arange(ix,len(train_lxi))
            #print(i)
            #print(batch_range[:5])
            #print(train_lxi[perm[batch_range]][0][:5])
            lbatch_feed = {lx_place: train_lxi[perm[batch_range]],
                           ly_place: train_lyi[perm[batch_range]],
                           lz_place: train_lzi[perm[batch_range]],
                           lxx_place: train_lxi[perm[batch_range]]*train_lxi[perm[batch_range]],
                           lyy_place: train_lyi[perm[batch_range]]*train_lyi[perm[batch_range]],
                           lzz_place: train_lzi[perm[batch_range]]*train_lzi[perm[batch_range]],
                           lxy_place: train_lxi[perm[batch_range]]*train_lyi[perm[batch_range]],
                           lyz_place: train_lyi[perm[batch_range]]*train_lzi[perm[batch_range]],
                           lzx_place: train_lzi[perm[batch_range]]*train_lxi[perm[batch_range]],
                           lout_place: train_louti[perm[batch_range]]}
            rbatch_feed = {rx_place: train_rxi[perm[batch_range]],
                           ry_place: train_ryi[perm[batch_range]],
                           rz_place: train_rzi[perm[batch_range]],
                           rxx_place: train_rxi[perm[batch_range]]*train_rxi[perm[batch_range]],
                           ryy_place: train_ryi[perm[batch_range]]*train_ryi[perm[batch_range]],
                           rzz_place: train_rzi[perm[batch_range]]*train_rzi[perm[batch_range]],
                           rxy_place: train_rxi[perm[batch_range]]*train_ryi[perm[batch_range]],
                           ryz_place: train_ryi[perm[batch_range]]*train_rzi[perm[batch_range]],
                           rzx_place: train_rzi[perm[batch_range]]*train_rxi[perm[batch_range]],
                           rout_place: train_routi[perm[batch_range]]}
            bbatch_feed = {lx_place: train_lxi[perm[batch_range]],
                           ly_place: train_lyi[perm[batch_range]],
                           lz_place: train_lzi[perm[batch_range]],
                           lxx_place: train_lxi[perm[batch_range]]*train_lxi[perm[batch_range]],
                           lyy_place: train_lyi[perm[batch_range]]*train_lyi[perm[batch_range]],
                           lzz_place: train_lzi[perm[batch_range]]*train_lzi[perm[batch_range]],
                           lxy_place: train_lxi[perm[batch_range]]*train_lyi[perm[batch_range]],
                           lyz_place: train_lyi[perm[batch_range]]*train_lzi[perm[batch_range]],
                           lzx_place: train_lzi[perm[batch_range]]*train_lxi[perm[batch_range]],
                           rx_place: train_rxi[perm[batch_range]],
                           ry_place: train_ryi[perm[batch_range]],
                           rz_place: train_rzi[perm[batch_range]],
                           rxx_place: train_rxi[perm[batch_range]]*train_rxi[perm[batch_range]],
                           ryy_place: train_ryi[perm[batch_range]]*train_ryi[perm[batch_range]],
                           rzz_place: train_rzi[perm[batch_range]]*train_rzi[perm[batch_range]],
                           rxy_place: train_rxi[perm[batch_range]]*train_ryi[perm[batch_range]],
                           ryz_place: train_ryi[perm[batch_range]]*train_rzi[perm[batch_range]],
                           rzx_place: train_rzi[perm[batch_range]]*train_rxi[perm[batch_range]],
                           lout_place: train_louti[perm[batch_range]]}

            ix += batchsize
            sess.run(optimizer_left, feed_dict=lbatch_feed)
            sess.run(optimizer_right,feed_dict=rbatch_feed)
            sess.run(optimizer_both, feed_dict=bbatch_feed)
        if (e+1) % (epochs/5) == 0:
            print(e,"complex")
    loutputs_pred = sess.run(prediction_left,   feed_dict=train_ldata_feeder)
    loutputs_test = sess.run(prediction_left,   feed_dict= test_ldata_feeder)
    routputs_pred = sess.run(prediction_right,  feed_dict=train_rdata_feeder)
    routputs_test = sess.run(prediction_right,  feed_dict= test_rdata_feeder)
    boutputs_pred = sess.run(prediction_both,   feed_dict=train_bdata_feeder)
    boutputs_test = sess.run(prediction_both,   feed_dict= test_bdata_feeder)

    #calculate Xscore (train/test, Xconfusion matrix (train/test), and Xroc_auc (train/test)
    #for left, right, and both
    ls = accuracy_score(train_ldata_feeder[lout_place].argmax(axis=1),loutputs_pred.argmax(axis=1))
    lst = accuracy_score(test_ldata_feeder[lout_place].argmax(axis=1),loutputs_test.argmax(axis=1))
    lc = confusion_matrix(train_ldata_feeder[lout_place].argmax(axis=1),loutputs_pred.argmax(axis=1))
    lct = confusion_matrix(test_ldata_feeder[lout_place].argmax(axis=1),loutputs_test.argmax(axis=1))
    lconfusion_sums += lc
    lconfusion_sumst += lct
    #try:
        #lra = roc_auc_score(train_ldata_feeder[k][lout_place].argmax(axis=1),loutputs_pred.argmax(axis=1))
    #except ValueError:
        #lra = -1
    #try:
        #lrat = roc_auc_score(test_ldata_feeder[k][lout_place].argmax(axis=1),loutputs_test.argmax(axis=1))
    #except ValueError:
        #lrat = -1

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
        print_write("ltrain accuracy score = %f" % (ls))
        print_write("ltest  accuracy score = %f" % (lst))
        print_write("ltrain confusion matrix =\n%s" % (str(lc)))
        print_write("ltest  confusion matrix =\n%s" % (str(lct)))
        print_write("rtrain accuracy score = %f" % (rs))
        print_write("rtest  accuracy score = %f" % (rst))
        print_write("rtrain confusion matrix =\n%s" % (str(rc)))
        print_write("rtest  confusion matrix =\n%s" % (str(rct)))
        print_write("btrain accuracy score = %f" % (bs))
        print_write("btest  accuracy score = %f" % (bst))
        print_write("btrain confusion matrix =\n%s" % (str(bc)))
        print_write("btest  confusion matrix =\n%s" % (str(bct)))
        #print_write("ltrain roc auc score = %.6f" % (lra))
        #print_write("ltest  roc auc score = %.6f" % (lrat))
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
